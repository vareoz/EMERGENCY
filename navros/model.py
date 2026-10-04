"""Transformer decodificador de NAVROS, con crecimiento que preserva la función.

El modelo puede crecer en profundidad (capas nuevas), en cabezas de atención,
en anchura del MLP y en vocabulario. Cada operación de crecimiento inicializa
los pesos nuevos de modo que la salida del modelo **no cambia** en el momento
de crecer (Net2Net / identidad por residuo): lo aprendido se conserva y la
capacidad extra se aprovecha en el entrenamiento siguiente.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DIGIT_IDS = (ord("0"), ord("9"))  # el tokenizador nunca fusiona dígitos
EQUALS_ID = ord("=")


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 32
    mlp_hidden: int = 512
    max_seq_len: int = 256
    pos: str = "rope"  # "rope" | "nope"
    abacus: int = 0  # tamaño de la tabla de posiciones numéricas; 0 = desactivado
    abacus_offset: int = 4  # desplazamiento aleatorio máximo al entrenar
    dropout: float = 0.0
    quantum_qubits: int = 0  # 0 = sin adaptador cuántico
    quantum_layers: int = 2


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rope(head_dim: int, max_len: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_len).float(), inv)
    return ang.cos(), ang.sin()


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


def _expand_linear(lin: nn.Linear, add_out: int = 0, add_in: int = 0,
                   zero_out: bool = False, zero_in: bool = True, std: float = 0.02) -> nn.Linear:
    """Copia ``lin`` añadiendo filas (salidas) y/o columnas (entradas)."""
    out_f, in_f = lin.out_features + add_out, lin.in_features + add_in
    new = nn.Linear(in_f, out_f, bias=False).to(lin.weight.device, lin.weight.dtype)
    with torch.no_grad():
        w = torch.zeros_like(new.weight) if zero_out else torch.randn_like(new.weight) * std
        if add_in and zero_in:
            w[:, lin.in_features:] = 0
        w[: lin.out_features, : lin.in_features] = lin.weight
        new.weight.copy_(w)
    return new


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        inner = cfg.n_heads * cfg.head_dim
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    @property
    def n_heads(self) -> int:
        return self.q.out_features // self.head_dim

    def forward(self, x, rope):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q, k, v = (m(x).view(B, T, H, D).transpose(1, 2) for m in (self.q, self.k, self.v))
        if rope is not None:
            cos, sin = rope[0][:T], rope[1][:T]
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, H * D))

    def grow_heads(self, n_new: int) -> None:
        add = n_new * self.head_dim
        self.q = _expand_linear(self.q, add_out=add)
        self.k = _expand_linear(self.k, add_out=add)
        self.v = _expand_linear(self.v, add_out=add)
        self.o = _expand_linear(self.o, add_in=add)  # columnas nuevas = 0


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def grow(self, n_new: int) -> None:
        self.fc1 = _expand_linear(self.fc1, add_out=n_new)
        self.fc2 = _expand_linear(self.fc2, add_in=n_new)  # columnas nuevas = 0


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class NavrosLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.abacus = nn.Embedding(cfg.abacus, cfg.d_model) if cfg.abacus else None
        self.quantum = None
        self.apply(self._init)
        for b in self.blocks:
            for lin in (b.attn.o, b.mlp.fc2):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.quantum_qubits:
            self._build_quantum(zero=False)
        self._rope_key = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def _rope(self, device):
        if self.cfg.pos != "rope":
            return None
        key = (self.cfg.head_dim, self.cfg.max_seq_len, str(device))
        if self._rope_key != key:
            cos, sin = _rope(self.cfg.head_dim, self.cfg.max_seq_len)
            self._rope_cache = (cos.to(device), sin.to(device))
            self._rope_key = key
        return self._rope_cache

    def digit_positions(self, idx: torch.Tensor) -> torch.Tensor:
        """Posición numérica de cada token (0 = sin posición numérica).

        - Abacus (McLeish et al., 2024): cada dígito recibe su índice dentro de
          su número (1 = primer dígito escrito).
        - Acoplamiento (Cho et al., 2024): tras ``=``, cada token recibe el
          índice del dígito que va a *predecir* (``=`` → 1, el dígito j → j+1),
          así la atención solo tiene que emparejar índices iguales.
        - Al entrenar se suma un desplazamiento aleatorio (≤ ``abacus_offset``)
          por secuencia: se entrenan índices mayores que los vistos y el modelo
          puede resolver números algo más largos — su frontera de automejora.
        """
        is_d = (idx >= DIGIT_IDS[0]) & (idx <= DIGIT_IDS[1])
        is_eq = idx == EQUALS_ID
        pos = torch.arange(idx.shape[1], device=idx.device).expand_as(idx)
        last_break = torch.cummax(torch.where(is_d, torch.full_like(pos, -1), pos), dim=1).values
        run = torch.where(is_d, pos - last_break, torch.zeros_like(pos))
        after_eq = is_eq.gather(1, last_break.clamp(min=0)) & (last_break >= 0) & is_d
        run = torch.where(after_eq, run + 1, run)
        run = torch.where(is_eq, torch.ones_like(run), run)
        if self.training and self.cfg.abacus_offset > 0:
            off = torch.randint(0, self.cfg.abacus_offset + 1, (idx.shape[0], 1), device=idx.device)
            run = torch.where(run > 0, run + off, run)
        return run.clamp(max=self.cfg.abacus - 1)

    def forward(self, idx):
        x = self.emb(idx)
        if self.abacus is not None:
            x = x + self.abacus(self.digit_positions(idx))
        rope = self._rope(idx.device)
        for b in self.blocks:
            x = b(x, rope)
        if self.quantum is not None:
            x = x + self.quantum(x)
        return self.norm_f(x) @ self.emb.weight.T  # embeddings atados

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- crecimiento (todas preservan la función) ---------------------------
    def grow_depth(self, n_new: int = 1) -> None:
        """Añade capas al final; con proyecciones de salida a cero son identidad."""
        dev = self.emb.weight.device
        for _ in range(n_new):
            b = Block(self.cfg).to(dev)
            b.apply(self._init)
            nn.init.zeros_(b.attn.o.weight)
            nn.init.zeros_(b.mlp.fc2.weight)
            self.blocks.append(b)
        self.cfg.n_layers += n_new

    def grow_heads(self, n_new: int = 1) -> None:
        for b in self.blocks:
            b.attn.grow_heads(n_new)
        self.cfg.n_heads += n_new

    def grow_mlp(self, n_new: int) -> None:
        for b in self.blocks:
            b.mlp.grow(n_new)
        self.cfg.mlp_hidden += n_new

    def grow_vocab(self, new_tokens: list[tuple[int, tuple[int, int]]]) -> None:
        """Añade embeddings para tokens nuevos = media de sus dos componentes."""
        if not new_tokens:
            return
        old = self.emb.weight.data
        n = max(i for i, _ in new_tokens) + 1
        emb = nn.Embedding(n, self.cfg.d_model).to(old.device, old.dtype)
        with torch.no_grad():
            emb.weight[: old.shape[0]] = old
            for new_id, (a, b) in new_tokens:
                emb.weight[new_id] = 0.5 * (emb.weight[a] + emb.weight[b])
        self.emb = emb
        self.cfg.vocab_size = n

    def attach_quantum(self, n_qubits: int, n_layers: int = 2) -> None:
        """Conecta un adaptador cuántico variacional (residual, salida a cero)."""
        self.cfg.quantum_qubits, self.cfg.quantum_layers = n_qubits, n_layers
        self._build_quantum(zero=True)

    def _build_quantum(self, zero: bool) -> None:
        from .quantum import QuantumAdapter
        self.quantum = QuantumAdapter(self.cfg.d_model, self.cfg.quantum_qubits,
                                      self.cfg.quantum_layers).to(self.emb.weight.device)
        if zero:
            nn.init.zeros_(self.quantum.up.weight)

    # -- persistencia --------------------------------------------------------
    def checkpoint(self) -> dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "NavrosLM":
        m = cls(ModelConfig(**ckpt["config"]))
        m.load_state_dict(ckpt["state_dict"])
        return m


@torch.inference_mode()
def generate(model: NavrosLM, prompts: list[list[int]], max_new_tokens: int,
             temperature: float = 0.0, eos_id: int | None = None, pad_id: int = 0,
             generator: torch.Generator | None = None) -> list[list[int]]:
    """Genera continuaciones por lotes (relleno a la derecha; atención causal)."""
    model.eval()
    dev = model.emb.weight.device
    B = len(prompts)
    start = torch.tensor([len(p) for p in prompts], device=dev)
    lens = start.clone()
    total = min(int(lens.max()) + max_new_tokens, model.cfg.max_seq_len)
    buf = torch.full((B, total), pad_id, dtype=torch.long, device=dev)
    for i, p in enumerate(prompts):
        buf[i, : len(p)] = torch.tensor(p, device=dev)
    done = lens >= total
    rows = torch.arange(B, device=dev)
    for _ in range(max_new_tokens):
        if bool(done.all()):
            break
        logits = model(buf[:, : int(lens.max())])[rows, lens - 1].float()
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        else:
            nxt = logits.argmax(-1)
        hit_eos = (nxt == eos_id) if eos_id is not None else torch.zeros_like(done)
        write = ~done & ~hit_eos
        buf[rows[write], lens[write]] = nxt[write]
        lens = lens + write.long()
        done = done | hit_eos | (lens >= total)
    return [buf[i, start[i]:lens[i]].tolist() for i in range(B)]
