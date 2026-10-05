"""Habilidades verificables: tareas donde NAVROS puede medir y mejorar su desempeño.

Una habilidad define cómo generar problemas de dificultad creciente, cómo
plantearlos como texto, cómo leer una respuesta y —si existe— cómo verificarla.
La verificación automática es lo que permite automejora real sin humanos:
el modelo propone, el verificador filtra, el modelo aprende de lo correcto.
"""

from __future__ import annotations

import random
from typing import Protocol

from .agent import ProtocolError, extract_action


class Skill(Protocol):
    name: str
    unit: str  # qué mide ``level`` (para los mensajes): "dígitos", "parámetros"…

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
    unit = "dígitos"

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


class HttpCallSkill:
    """Escribir la petición HTTP correcta para una orden en español (primer paso del agente).

    ``level`` = nº de parámetros de consulta. La respuesta se analiza con el mismo
    analizador que usa ``navros.agent`` y debe coincidir exactamente con la
    petición esperada (método, URL, sin cabeceras ni cuerpo de más): un verificador
    automático, como en la suma. Enseña el *formato*; leer el resultado y
    decidir el siguiente paso es una habilidad aparte (``docs/AGENTE.md``).

    El prompt es lo que ``Agent`` le da al modelo en el primer turno: la tarea + salto de línea.
    """

    name = "http"
    unit = "parámetros"
    GET_VERBS = ("obtén", "consulta", "pide")
    HEAD_VERBS = ("comprueba",)
    HOSTS = ("api.example.com", "datos.example.org", "svc.example.net", "tienda.example.com", "clima.example.org")
    PATHS = ("/v1/items", "/v1/usuarios", "/buscar", "/clima", "/precios", "/v2/pedidos", "/estado")
    NAMES = ("id", "q", "page", "lang", "tag", "sort", "limit", "city", "from", "to")
    VALUES = ("es", "en", "asc", "desc", "madrid", "lima", "rojo", "nuevo")

    def make_problem(self, rng: random.Random, level: int) -> dict:
        names = rng.sample(self.NAMES, min(level, len(self.NAMES)))
        names += [f"p{i}" for i in range(len(names), level)]  # más allá del vocabulario: p10, p11…
        params = [[n, str(rng.randint(0, 999)) if rng.random() < 0.6 else rng.choice(self.VALUES)] for n in names]
        head = rng.random() < 0.2
        return {"verb": rng.choice(self.HEAD_VERBS if head else self.GET_VERBS), "host": rng.choice(self.HOSTS),
                "path": rng.choice(self.PATHS), "params": params, "level": level}

    def prompt(self, p: dict) -> str:
        pairs = " ".join(f"{k}={v}" for k, v in p["params"])
        return f"{p['verb']} {p['host']}{p['path']} con {pairs}\n"

    def _method(self, p: dict) -> str:
        return "HEAD" if p["verb"] in self.HEAD_VERBS else "GET"

    def target(self, p: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in p["params"])
        return f"<http>\n{self._method(p)} https://{p['host']}{p['path']}?{query}\n</http>"

    def answer_of(self, completion: str) -> str | None:
        """La petición en forma canónica, o ``None`` si no es una petición bien formada.

        Debe ser a su vez una completion válida (el modo ``consensus`` de ``core`` guarda
        esta cadena como dato de entrenamiento): lo que sobra del texto del modelo se descarta.
        """
        s = completion.strip()
        if not s.startswith("<http>"):
            return None
        try:
            act = extract_action(s)
        except ProtocolError:
            return None
        if act is None or act.kind != "http":
            return None
        r = act.request
        lines = [f"{r.method} {r.url}", *(f"{k}: {v}" for k, v in sorted(r.headers.items()))]
        if r.body:
            lines += ["", r.body.decode("utf-8", "replace")]
        return "<http>\n" + "\n".join(lines) + "\n</http>"

    def verify(self, p: dict, completion: str) -> bool:
        ans = self.answer_of(completion)
        return ans is not None and ans == self.target(p)

    def render(self, p: dict, completion: str) -> str:
        ans = self.answer_of(completion)
        shown = ans.splitlines()[1] if ans is not None else f"<inválida:{completion!r}>"
        return f"{self.prompt(p).strip()} → {shown} {'✓' if self.verify(p, completion) else '✗'}"

    def max_answer_tokens(self, level: int) -> int:
        # Cota en bytes (un token nunca es menos de un byte): etiquetas + host + ruta + parámetros.
        per_param = max(map(len, self.NAMES)) + 3 + max(map(len, self.VALUES)) + 3  # nombre+pNN, =, valor, &
        fixed = len("<http>\nHEAD https://?\n</http>")
        return fixed + max(map(len, self.HOSTS)) + max(map(len, self.PATHS)) + level * per_param

    def parse_human(self, expr: str) -> dict:
        """Inverso de ``prompt``: «obtén api.example.com/v1/items con id=7 tag=x»."""
        verb, hostpath, con, *pairs = expr.split()
        host, slash, rest = hostpath.partition("/")
        if (verb not in self.GET_VERBS + self.HEAD_VERBS or con != "con" or not slash
                or not pairs or not all("=" in p for p in pairs)):
            raise ValueError("formato: «obtén host/ruta con clave=valor …»")
        return {"verb": verb, "host": host, "path": "/" + rest, "params": [p.split("=", 1) for p in pairs],
                "level": len(pairs)}


SKILLS = {"suma": AdditionSkill, "http": HttpCallSkill}
