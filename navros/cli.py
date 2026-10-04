"""Línea de comandos: ``python -m navros <comando>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RUN = "runs/navros"


def _corpus_default() -> list[str]:
    root = Path(__file__).resolve().parent.parent
    return [str(p) for p in (root / "README.md", root / "docs") if p.exists()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="navros", description="NAVROS — IA que se automejora")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--run", default=DEFAULT_RUN, help="directorio de la ejecución")
        return p

    p = cmd("init", "crea NAVROS: tokenizador propio, pesos iniciales, preentrenamiento")
    p.add_argument("--preset", default="small", choices=["tiny", "mini", "small", "base", "large"])
    p.add_argument("--corpus", nargs="*", default=None, help="textos para su tokenizador/corpus")
    p.add_argument("--mode", choices=["verifier", "consensus"], default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="sobrescribe una ejecución existente")

    p = cmd("improve", "rondas de automejora (reanudable)")
    p.add_argument("--rounds", type=int, default=5, help="0 = continuo hasta --hours o max_level")
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--mode", choices=["verifier", "consensus"], default=None)

    cmd("status", "estado, linaje e historial")

    p = cmd("solve", "resuelve una suma, p. ej. 12345+678")
    p.add_argument("expr")

    p = cmd("generate", "genera texto")
    p.add_argument("prompt")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)

    p = cmd("ingest", "asimila textos nuevos (amplía su tokenizador si hace falta)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--steps", type=int, default=500)

    p = cmd("grow", "crecimiento manual que preserva la función")
    p.add_argument("kind", choices=["depth", "mlp", "heads"])

    p = cmd("assimilate", "aprende de otro modelo NAVROS (destilación)")
    p.add_argument("--teacher", required=True, help="directorio de ejecución del maestro")
    p.add_argument("--problems", type=int, default=2048)

    p = cmd("merge", "fusiona pesos con otro NAVROS de igual arquitectura")
    p.add_argument("--with", dest="other", required=True)
    p.add_argument("--weight", type=float, default=0.5)

    p = cmd("quantum", "adaptador cuántico: demo del simulador o conectarlo al modelo")
    p.add_argument("action", choices=["demo", "attach"])
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)

    a = ap.parse_args(argv)

    if a.cmd == "quantum" and a.action == "demo":
        return _quantum_demo(a.qubits, a.layers)

    from .core import Navros, NavrosTeacher, preset_config

    if a.cmd == "init":
        if (Path(a.run) / "model.pt").exists() and not a.force:
            raise SystemExit(f"{a.run} ya existe (usa --force para sobrescribir o 'improve' para continuar)")
        cfg = preset_config(a.preset, seed=a.seed)
        if a.mode:
            cfg.improve.mode = a.mode
        nav = Navros.create(a.run, cfg, a.corpus if a.corpus is not None else _corpus_default())
        print(nav.status())
        return

    nav = Navros.load(a.run)
    if a.cmd == "improve":
        if a.mode:
            nav.cfg.improve.mode = a.mode
        nav.improve(a.rounds, a.hours)
        print("\n" + nav.status())
    elif a.cmd == "status":
        print(nav.status())
    elif a.cmd == "solve":
        print(nav.solve(a.expr))
    elif a.cmd == "generate":
        print(nav.generate_text(a.prompt, a.tokens, a.temperature))
    elif a.cmd == "ingest":
        print(json.dumps(nav.ingest(a.paths, steps=a.steps), indent=2, ensure_ascii=False))
    elif a.cmd == "grow":
        nav.grow(a.kind)
        nav.save()
        print(nav.status())
    elif a.cmd == "assimilate":
        info = nav.assimilate(NavrosTeacher(Navros.load(a.teacher)), n_problems=a.problems)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    elif a.cmd == "merge":
        nav.merge_with(Navros.load(a.other), a.weight)
        print(nav.status())
    elif a.cmd == "quantum":
        nav.attach_quantum(a.qubits, a.layers)
        print(nav.status())


def _quantum_demo(n_qubits: int, n_layers: int) -> None:
    import torch

    from .quantum import SimulatorBackend, VariationalCircuit

    torch.manual_seed(0)
    vqc = VariationalCircuit(n_qubits, n_layers)
    x = torch.rand(1, n_qubits) * 3.14
    exact = vqc(x)[0]
    sampled = vqc.run_on_backend(x, SimulatorBackend(seed=0), shots=4096)[0]
    w = torch.ones(1, n_qubits)
    vqc.zero_grad()
    (vqc(x) * w).sum().backward()
    ps = vqc.parameter_shift_grad(x, w)
    print(f"Circuito variacional: {n_qubits} qubits, {n_layers} capas, {vqc.theta.numel()} parámetros")
    print("⟨Z⟩ exacto (vector de estado):", [round(v, 4) for v in exact.tolist()])
    print("⟨Z⟩ medido (4096 disparos)   :", [round(v, 4) for v in sampled.tolist()])
    print("máx |∇ autograd − ∇ parameter-shift| =", f"{(vqc.theta.grad - ps).abs().max().item():.2e}")
    print("\nOpenQASM 2.0 listo para hardware:\n")
    print(vqc.to_qasm(x[0]))
