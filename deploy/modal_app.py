"""NAVROS en Modal: cualquier GPU bajo demanda + volumen persistente.

Preparación (una vez):
    pip install modal && modal setup

Uso (desde la raíz del repositorio):
    NAVROS_GPU=A100 modal run deploy/modal_app.py --action init --preset base
    NAVROS_GPU=A100 modal run deploy/modal_app.py --action improve --rounds 20
    NAVROS_GPU=H100 modal run deploy/modal_app.py --action improve --rounds 0 --hours 23
    modal run deploy/modal_app.py --action status
    modal run deploy/modal_app.py --action solve --expr 98765+4321

Automejora continua programada (una sesión cada N horas, se reanuda sola):
    NAVROS_SCHEDULE_HOURS=6 NAVROS_GPU=A100 modal deploy deploy/modal_app.py

Descargar el modelo:
    modal volume get navros-runs navros ./runs/navros

GPU válidas: T4, L4, A10G, L40S, A100, A100-80GB, H100, H200, B200.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("NAVROS_GPU", "A10G")
SCHEDULE_HOURS = float(os.environ.get("NAVROS_SCHEDULE_HOURS", "0"))
RUN = "/runs/" + os.environ.get("NAVROS_RUN", "navros")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3", "numpy")
    # Las mismas variables dentro del contenedor, para que el módulo se defina igual.
    .env({"NAVROS_GPU": GPU, "NAVROS_SCHEDULE_HOURS": str(SCHEDULE_HOURS),
          "NAVROS_RUN": RUN.rsplit("/", 1)[1], "NAVROS_COMPILE": os.environ.get("NAVROS_COMPILE", "1")})
    .add_local_file(ROOT / "README.md", "/root/corpus/README.md")
    .add_local_dir(ROOT / "docs", "/root/corpus/docs")
    .add_local_dir(ROOT / "navros", "/root/navros")
)
app = modal.App("navros", image=image)
vol = modal.Volume.from_name("navros-runs", create_if_missing=True)
DAY = 24 * 3600


@app.function(gpu=GPU, volumes={"/runs": vol}, timeout=DAY)
def init(preset: str = "base", force: bool = False, mode: str = "") -> str:
    from navros.core import Navros, preset_config

    if Path(RUN, "model.pt").exists() and not force:
        return f"{RUN} ya existe; usa --force para recrearlo o --action improve para continuar."
    cfg = preset_config(preset)
    if mode:
        cfg.improve.mode = mode
    nav = Navros.create(RUN, cfg, corpus=["/root/corpus"])
    vol.commit()
    return nav.status()


def _improve(rounds: int, hours: float, mode: str = "") -> str:
    from navros.core import Navros

    vol.reload()
    nav = Navros.load(RUN)
    if mode:
        nav.cfg.improve.mode = mode
    nav.improve(rounds, max_hours=hours, on_round=lambda _: vol.commit())  # guarda cada ronda
    return nav.status()


@app.function(gpu=GPU, volumes={"/runs": vol}, timeout=DAY)
def improve(rounds: int = 10, hours: float = 23.0, mode: str = "") -> str:
    return _improve(rounds, hours, mode)


@app.function(volumes={"/runs": vol}, timeout=600)
def query(action: str, expr: str = "") -> str:
    from navros.core import Navros

    vol.reload()
    nav = Navros.load(RUN)
    if action == "solve":
        return nav.solve(expr)
    if action == "generate":
        return nav.generate_text(expr)
    return nav.status()


if SCHEDULE_HOURS > 0:
    @app.function(gpu=GPU, volumes={"/runs": vol}, timeout=DAY,
                  schedule=modal.Period(minutes=int(SCHEDULE_HOURS * 60)))
    def improve_scheduled() -> str:
        return _improve(rounds=0, hours=min(SCHEDULE_HOURS * 0.9, 23.0))


@app.local_entrypoint()
def main(action: str = "status", preset: str = "base", rounds: int = 10, hours: float = 23.0,
         mode: str = "", expr: str = "", force: bool = False):
    if action == "init":
        print(init.remote(preset, force, mode))
    elif action == "improve":
        print(improve.remote(rounds, hours, mode))
    else:
        print(query.remote(action, expr))
