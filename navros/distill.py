"""Asimilación de otros modelos: destilación de conocimiento y fusión de pesos.

Tres vías, de más a menos íntima:

1. **Fusión de pesos** (``merge_state_dicts``): promedia modelos NAVROS con la
   misma arquitectura (*model soup*). Combina lo aprendido por linajes distintos.
2. **Destilación de logits** (``logit_distill_loss``): el estudiante imita la
   distribución completa de probabilidades del maestro. Requiere el mismo
   tokenizador, así que se usa entre modelos NAVROS.
3. **Destilación de secuencias**: el maestro responde, las respuestas (filtradas
   por el verificador cuando existe) pasan a ser datos de entrenamiento. Sirve
   con cualquier modelo que puedas consultar.

Licencias: un maestro debe declarar ``allows_distillation=True``. Usa solo
modelos cuya licencia o términos permitan entrenar otro modelo con sus
salidas (p. ej. muchos modelos de pesos abiertos con licencia Apache-2.0/MIT,
o tus propios modelos). Los términos de Anthropic, OpenAI y Google prohíben
usar las salidas de sus modelos para desarrollar modelos competidores, por eso
NAVROS no incluye conectores hacia esas APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn.functional as F

from .model import NavrosLM


class Teacher(Protocol):
    name: str
    license: str
    allows_distillation: bool

    def complete(self, prompts: list[str]) -> list[str]:
        """Continúa cada prompt (en el formato de texto de NAVROS)."""
        ...


@dataclass
class FunctionTeacher:
    """Envuelve cualquier función ``prompts -> completions`` como maestro.

    Ejemplo: un modelo de pesos abiertos cargado con ``transformers``, cuya
    licencia permita destilación.
    """

    fn: Callable[[list[str]], list[str]]
    name: str
    license: str
    allows_distillation: bool = False
    meta: dict = field(default_factory=dict)

    def complete(self, prompts: list[str]) -> list[str]:
        return self.fn(prompts)


def check_license(teacher: Teacher) -> None:
    if not getattr(teacher, "allows_distillation", False):
        raise PermissionError(
            f"El maestro '{teacher.name}' (licencia: {teacher.license}) no declara "
            "allows_distillation=True. Verifica que su licencia permita entrenar "
            "otro modelo con sus salidas antes de habilitarlo.")


def logit_distill_loss(teacher: NavrosLM, temperature: float = 2.0, alpha: float = 0.5):
    """Pérdida = α·KL(maestro‖estudiante)·T² + (1-α)·entropía cruzada con los datos."""
    teacher.eval()

    def loss_fn(forward, x, y, m):
        s = forward(x).float()
        with torch.no_grad():
            t = teacher(x).float()
        V = min(s.shape[-1], t.shape[-1])
        s_log = F.log_softmax(s[..., :V] / temperature, -1)
        t_prob = F.softmax(t[..., :V] / temperature, -1)
        kl = (t_prob * (t_prob.clamp_min(1e-9).log() - s_log)).sum(-1)
        ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), y.reshape(-1), reduction="none").view_as(kl)
        mask = m / m.sum().clamp(min=1)
        return ((alpha * temperature ** 2 * kl + (1 - alpha) * ce) * mask).sum()

    return loss_fn


def merge_state_dicts(states: list[dict], weights: list[float] | None = None) -> dict:
    """Promedio ponderado de pesos de modelos con formas idénticas."""
    weights = weights or [1.0 / len(states)] * len(states)
    total = sum(weights)
    keys = states[0].keys()
    for s in states[1:]:
        if s.keys() != keys or any(s[k].shape != states[0][k].shape for k in keys):
            raise ValueError("Solo se fusionan modelos con la misma arquitectura. "
                             "Haz crecer el menor (grow_*) hasta igualarlo, o usa destilación.")
    return {k: sum(w * s[k].float() for w, s in zip(weights, states)).div(total).to(states[0][k].dtype)
            for k in keys}
