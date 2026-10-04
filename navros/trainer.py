"""Datos, bucle de entrenamiento y evaluación."""

from __future__ import annotations

import contextlib
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import NavrosLM, generate
from .skills import Skill
from .tokenizer import BPETokenizer, EOS, PAD


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup: int = 50
    grad_clip: float = 1.0


Example = tuple[list[int], list[int]]  # (ids, máscara de pérdida 0/1)


def best_device() -> torch.device:
    """CUDA (Azure / Modal / Kaggle) > Apple MPS > CPU."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(os.cpu_count() or 1)
    return torch.device("cpu")


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 en A100/H100/L4/A10; fp16 en T4/P100 (Kaggle); nada en CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device):
    dt = amp_dtype(device)
    return torch.autocast(device.type, dtype=dt) if dt else contextlib.nullcontext()


def skill_example(tok: BPETokenizer, skill: Skill, problem: dict, completion: str) -> Example:
    """<bos> pregunta respuesta <eos>; la pérdida se calcula solo sobre la respuesta."""
    p = tok.encode(skill.prompt(problem), bos=True)
    a = tok.encode(completion, eos=True)
    return p + a, [0] * len(p) + [1] * len(a)


def text_examples(tok: BPETokenizer, texts: list[str], seq_len: int) -> list[Example]:
    out = []
    for t in texts:
        ids = tok.encode(t, bos=True, eos=True)
        for i in range(0, max(len(ids) - 1, 1), seq_len):
            chunk = ids[i : i + seq_len + 1]
            if len(chunk) > 1:
                out.append((chunk, [1] * len(chunk)))
    return out


def collate(batch: list[Example], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), T), PAD, dtype=torch.long)
    m = torch.zeros((len(batch), T))
    for i, (ids, mask) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        m[i, : len(mask)] = torch.tensor(mask, dtype=torch.float)
    # entrada = x[:, :-1], objetivo = x[:, 1:], máscara alineada con el objetivo
    return x[:, :-1].to(device), x[:, 1:].to(device), m[:, 1:].to(device)


class Mixture:
    """Muestrea lotes de varias fuentes de ejemplos según pesos."""

    def __init__(self, sources: list[tuple[list[Example], float]], rng: random.Random):
        self.sources = [(s, w) for s, w in sources if s and w > 0]
        self.rng = rng

    def batch(self, n: int) -> list[Example]:
        data = self.rng.choices([s for s, _ in self.sources], [w for _, w in self.sources])[0]
        return [self.rng.choice(data) for _ in range(n)]


def masked_ce(logits, y, m) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), reduction="none")
    return (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)


def make_optimizer(model: NavrosLM, lr: float, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "emb" in n]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.98))


def train(model: NavrosLM, mixture: Mixture, steps: int, cfg: TrainConfig,
          lr: float | None = None, log=None, log_every: int = 100, loss_fn=None) -> float:
    """Entrena ``steps`` pasos con AdamW y coseno; devuelve la pérdida media final.

    ``loss_fn(forward, x, y, m)`` permite otras pérdidas (p. ej. destilación).
    Con ``NAVROS_COMPILE=1`` usa ``torch.compile`` (vale la pena en GPU).
    """
    lr = lr or cfg.lr
    dev = model.emb.weight.device
    opt = make_optimizer(model, lr, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(dev) == torch.float16)
    forward = torch.compile(model) if os.environ.get("NAVROS_COMPILE") == "1" else model
    loss_fn = loss_fn or (lambda fwd, x, y, m: masked_ce(fwd(x), y, m))
    model.train()
    recent, t0 = [], time.time()
    for step in range(steps):
        warm = min(1.0, (step + 1) / max(cfg.warmup, 1))
        cos = 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
        for g in opt.param_groups:
            g["lr"] = lr * warm * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cos)
        x, y, m = collate(mixture.batch(cfg.batch_size), dev)
        with autocast(dev):
            loss = loss_fn(forward, x, y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        recent = (recent + [loss.item()])[-50:]
        if log and (step + 1) % log_every == 0:
            log(f"  paso {step + 1}/{steps}  pérdida {sum(recent) / len(recent):.4f}  "
                f"({time.time() - t0:.0f}s)")
    return sum(recent) / max(len(recent), 1)


def complete(model: NavrosLM, tok: BPETokenizer, skill: Skill, problems: list[dict],
             temperature: float = 0.0, batch_size: int | None = None,
             generator: torch.Generator | None = None) -> list[str]:
    batch_size = batch_size or (4096 if model.emb.weight.device.type == "cuda" else 512)
    outs: list[str] = []
    for i in range(0, len(problems), batch_size):
        chunk = problems[i : i + batch_size]
        prompts = [tok.encode(skill.prompt(p), bos=True) for p in chunk]
        max_new = max(skill.max_answer_tokens(p["level"]) for p in chunk)
        with autocast(model.emb.weight.device):
            gen = generate(model, prompts, max_new, temperature, eos_id=EOS, pad_id=PAD, generator=generator)
        outs.extend(tok.decode(g) for g in gen)
    return outs


def accuracy(model: NavrosLM, tok: BPETokenizer, skill: Skill, level: int, n: int, seed: int) -> float:
    """Exactitud (decodificación voraz) contra la verdad de referencia."""
    rng = random.Random(seed)
    problems = [skill.make_problem(rng, level) for _ in range(n)]
    comps = complete(model, tok, skill, problems)
    return sum(skill.verify(p, c) for p, c in zip(problems, comps)) / n
