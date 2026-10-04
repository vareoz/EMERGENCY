"""Módulo cuántico de NAVROS: circuitos variacionales listos para hardware real.

Qué hay aquí, sin exageraciones:

- Un simulador de vector de estado diferenciable (PyTorch, números complejos)
  para circuitos de pocos qubits (~ hasta 12–14 en CPU).
- ``VariationalCircuit``: codificación por ángulos + capas RY/RZ + anillo de
  CNOT, cuyas salidas son los valores esperados ⟨Z_i⟩.
- ``QuantumAdapter``: capa híbrida clásico-cuántica que se conecta de forma
  residual al transformer (``NavrosLM.attach_quantum``).
- Exportación a OpenQASM 2.0 y ejecución sobre cualquier *backend* que reciba
  QASM y devuelva conteos (simulador local o un proveedor de hardware).
- Gradientes por *parameter-shift*, la regla que se usa en un QPU real, donde
  no existe retropropagación.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import torch
import torch.nn as nn

Op = tuple[str, tuple[int, ...], float | None]  # (puerta, qubits, ángulo)


# -- simulador ----------------------------------------------------------------
def _gate(name: str, theta: torch.Tensor | float | None, batch: int):
    """Matrices 2x2 (complejas) por lote: tensor (B, 2, 2)."""
    if name == "h":
        g = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
        return g.expand(batch, 2, 2)
    th = torch.as_tensor(theta, dtype=torch.float32).reshape(-1).expand(batch) / 2
    c, s = torch.cos(th), torch.sin(th)
    z = torch.zeros_like(c)
    if name == "ry":
        re = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        return torch.complex(re, torch.zeros_like(re))
    if name == "rx":
        re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
        im = torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2)
        return torch.complex(re, im)
    if name == "rz":
        re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
        im = torch.stack([torch.stack([-s, z], -1), torch.stack([z, s], -1)], -2)
        return torch.complex(re, im)
    raise ValueError(f"puerta desconocida: {name}")


class Statevector:
    """Estado de n qubits por lote. Convención: el qubit 0 es el bit más significativo."""

    def __init__(self, n: int, batch: int):
        self.n, self.batch = n, batch
        self.amp = torch.zeros(batch, 2 ** n, dtype=torch.complex64)
        self.amp[:, 0] = 1

    def apply_1q(self, gate: torch.Tensor, q: int) -> None:
        s = self.amp.reshape(self.batch, 2 ** q, 2, 2 ** (self.n - q - 1))
        self.amp = torch.einsum("bij,bajc->baic", gate, s).reshape(self.batch, -1)

    def apply_cnot(self, control: int, target: int) -> None:
        idx = torch.arange(2 ** self.n)
        cbit = (idx >> (self.n - 1 - control)) & 1
        perm = torch.where(cbit == 1, idx ^ (1 << (self.n - 1 - target)), idx)
        self.amp = self.amp[:, perm]

    def probabilities(self) -> torch.Tensor:
        return self.amp.real ** 2 + self.amp.imag ** 2

    def expect_z(self) -> torch.Tensor:
        """⟨Z_i⟩ para cada qubit: tensor (B, n)."""
        idx = torch.arange(2 ** self.n)
        bits = torch.stack([(idx >> (self.n - 1 - q)) & 1 for q in range(self.n)], -1)
        return self.probabilities() @ (1 - 2 * bits).float()


def simulate(n: int, ops: list[tuple[str, tuple[int, ...], torch.Tensor | float | None]],
             batch: int = 1) -> Statevector:
    sv = Statevector(n, batch)
    for name, qubits, theta in ops:
        if name == "cx":
            sv.apply_cnot(*qubits)
        else:
            sv.apply_1q(_gate(name, theta, batch), qubits[0])
    return sv


# -- circuito variacional -------------------------------------------------------
class VariationalCircuit(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__()
        self.n, self.layers = n_qubits, n_layers
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

    def ops(self, x: torch.Tensor, theta: torch.Tensor | None = None) -> list:
        """Lista de puertas. ``x``: (B, n) ángulos de entrada."""
        theta = self.theta if theta is None else theta
        ops = [("ry", (q,), x[:, q]) for q in range(self.n)]
        for layer in range(self.layers):
            for q in range(self.n):
                ops.append(("ry", (q,), theta[layer, q, 0]))
                ops.append(("rz", (q,), theta[layer, q, 1]))
            if self.n > 1:
                for q in range(self.n):
                    ops.append(("cx", (q, (q + 1) % self.n), None))
        return ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return simulate(self.n, self.ops(x), batch=x.shape[0]).expect_z()

    # -- preparación para hardware ---------------------------------------------
    def to_qasm(self, x: torch.Tensor) -> str:
        """Circuito para UNA muestra (x: (n,)) en OpenQASM 2.0 con medición."""
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
                 f"qreg q[{self.n}];", f"creg c[{self.n}];"]
        for name, qubits, theta in self.ops(x.reshape(1, -1).detach(), self.theta.detach()):
            if name == "cx":
                lines.append(f"cx q[{qubits[0]}],q[{qubits[1]}];")
            else:
                lines.append(f"{name}({float(torch.as_tensor(theta).reshape(-1)[0]):.10f}) q[{qubits[0]}];")
        lines.append("measure q -> c;")
        return "\n".join(lines) + "\n"

    def run_on_backend(self, x: torch.Tensor, backend: "QuantumBackend", shots: int = 1024) -> torch.Tensor:
        """Inferencia en un backend que ejecuta QASM y devuelve conteos de bits."""
        rows = [expectations_from_counts(backend.run(self.to_qasm(xi), shots), self.n) for xi in x]
        return torch.stack(rows)

    @torch.no_grad()
    def parameter_shift_grad(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """d(Σ weights·⟨Z⟩)/dθ por parameter-shift (exacto para rotaciones Pauli)."""
        grad = torch.zeros_like(self.theta)
        base = self.theta.detach()
        for i in range(base.numel()):
            shift = torch.zeros_like(base).view(-1)
            shift[i] = math.pi / 2
            shift = shift.view_as(base)
            f = lambda t: (simulate(self.n, self.ops(x, t), x.shape[0]).expect_z() * weights).sum()
            grad.view(-1)[i] = 0.5 * (f(base + shift) - f(base - shift))
        return grad


class QuantumAdapter(nn.Module):
    """h -> up(VQC(π·tanh(down(h)))) — capa híbrida residual para NavrosLM."""

    def __init__(self, d_model: int, n_qubits: int, n_layers: int = 2):
        super().__init__()
        self.down = nn.Linear(d_model, n_qubits, bias=False)
        self.vqc = VariationalCircuit(n_qubits, n_layers)
        self.up = nn.Linear(n_qubits, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shape = h.shape
        angles = math.pi * torch.tanh(self.down(h)).reshape(-1, self.vqc.n)
        z = self.vqc(angles.float()).to(h.dtype)
        return self.up(z).reshape(*shape[:-1], -1)


# -- backends -----------------------------------------------------------------
class QuantumBackend(Protocol):
    def run(self, qasm: str, shots: int) -> dict[str, int]:
        """Ejecuta un circuito OpenQASM 2.0 y devuelve conteos {bits: veces}.

        Las claves deben tener el qubit 0 primero (q0 q1 ... q{n-1}).
        """
        ...


def expectations_from_counts(counts: dict[str, int], n: int) -> torch.Tensor:
    shots = sum(counts.values())
    z = torch.zeros(n)
    for bits, c in counts.items():
        for q in range(n):
            z[q] += c * (1 if bits[q] == "0" else -1)
    return z / max(shots, 1)


def _parse_qasm(qasm: str) -> tuple[int, list[Op]]:
    n, ops = 0, []
    for line in qasm.splitlines():
        line = line.strip().rstrip(";")
        if line.startswith("qreg"):
            n = int(line.split("[")[1].split("]")[0])
        elif line.startswith("cx"):
            a, b = (int(t.split("[")[1].rstrip("]")) for t in line[2:].split(","))
            ops.append(("cx", (a, b), None))
        elif line[:2] in ("rx", "ry", "rz") or line.startswith("h "):
            name = line.split("(")[0].split(" ")[0]
            theta = float(line.split("(")[1].split(")")[0]) if "(" in line else None
            q = int(line.split("[")[-1].rstrip("]"))
            ops.append((name, (q,), theta))
    return n, ops


class SimulatorBackend:
    """Backend local: interpreta el QASM exportado y muestrea mediciones."""

    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)

    def run(self, qasm: str, shots: int) -> dict[str, int]:
        n, ops = _parse_qasm(qasm)
        probs = simulate(n, ops).probabilities()[0]
        samples = torch.multinomial(probs, shots, replacement=True, generator=self.gen)
        counts: dict[str, int] = {}
        for s in samples.tolist():
            key = format(s, f"0{n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts


class CallableBackend:
    """Adapta cualquier función ``f(qasm, shots) -> counts`` (p. ej. un SDK de hardware)."""

    def __init__(self, fn: Callable[[str, int], dict[str, int]], qubit0_last: bool = False):
        self.fn, self.qubit0_last = fn, qubit0_last

    def run(self, qasm: str, shots: int) -> dict[str, int]:
        counts = self.fn(qasm, shots)
        # Qiskit y otros SDK devuelven el qubit 0 al final de la cadena.
        return {k[::-1]: v for k, v in counts.items()} if self.qubit0_last else counts
