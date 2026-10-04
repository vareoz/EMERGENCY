"""Habilidades verificables: tareas donde NAVROS puede medir y mejorar su desempeño.

Una habilidad define cómo generar problemas de dificultad creciente, cómo
plantearlos como texto, cómo leer una respuesta y —si existe— cómo verificarla.
La verificación automática es lo que permite automejora real sin humanos:
el modelo propone, el verificador filtra, el modelo aprende de lo correcto.
"""

from __future__ import annotations

import random
from typing import Protocol


class Skill(Protocol):
    name: str

    def make_problem(self, rng: random.Random, level: int) -> dict: ...
    def prompt(self, problem: dict) -> str: ...
    def target(self, problem: dict) -> str: ...
    def answer_of(self, completion: str) -> str | None: ...
    def verify(self, problem: dict, completion: str) -> bool: ...
    def render(self, problem: dict, completion: str) -> str: ...
    def max_answer_tokens(self, level: int) -> int: ...


class AdditionSkill:
    """Suma de enteros de ``level`` dígitos, escritos al revés (unidades primero).

    Escribir los dígitos al revés hace que el acarreo fluya en el mismo orden
    en que el modelo genera, lo que vuelve el algoritmo aprendible por un
    transformer pequeño y extensible a números más largos.
    """

    name = "suma"

    def make_problem(self, rng: random.Random, level: int) -> dict:
        def num(d):
            return rng.randint(10 ** (d - 1) if d > 1 else 0, 10 ** d - 1)
        a, b = num(level), num(rng.randint(1, level))
        if rng.random() < 0.5:
            a, b = b, a
        return {"a": a, "b": b, "level": level}

    def prompt(self, p: dict) -> str:
        return f"{str(p['a'])[::-1]}+{str(p['b'])[::-1]}="

    def target(self, p: dict) -> str:
        return str(p["a"] + p["b"])[::-1]

    def answer_of(self, completion: str) -> str | None:
        s = completion.strip()
        return s if s.isdigit() and (len(s) == 1 or s[-1] != "0") else None

    def verify(self, p: dict, completion: str) -> bool:
        return self.answer_of(completion) == self.target(p)

    def render(self, p: dict, completion: str) -> str:
        ans = self.answer_of(completion)
        shown = ans[::-1] if ans is not None else f"<inválida:{completion!r}>"
        mark = "✓" if self.verify(p, completion) else "✗"
        return f"{p['a']} + {p['b']} = {shown} {mark}"

    def max_answer_tokens(self, level: int) -> int:
        return level + 2

    @staticmethod
    def parse_human(expr: str) -> dict:
        a, b = (int(x) for x in expr.replace(" ", "").split("+"))
        return {"a": a, "b": b, "level": max(len(str(a)), len(str(b)))}


SKILLS = {"suma": AdditionSkill}
