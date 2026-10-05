"""NAVROS practicando la habilidad «http» y manejando el bucle del agente (sin red)."""

import pytest

from navros.agent import Agent, NavrosAgentModel
from navros.core import Navros, preset_config
from navros.httptool import HttpPolicy, HttpTool


def offline(host, port):
    raise OSError("las pruebas no salen a Internet")


@pytest.fixture(scope="module")
def nav(tmp_path_factory):
    cfg = preset_config("tiny", skill="http")
    cfg.model.max_seq_len = 256  # una petición no cabe en los 64 tokens de «tiny»
    return Navros.create(tmp_path_factory.mktemp("http"), cfg, log=lambda *_: None)


def test_warns_when_the_window_cannot_fit_the_skill(tmp_path):
    logs = []
    Navros.create(tmp_path, preset_config("tiny", skill="http"), log=logs.append)
    assert any("AVISO" in m and "ventana" in m for m in logs)
    quiet = []
    Navros.create(tmp_path / "suma", preset_config("tiny"), log=quiet.append)
    assert not any("AVISO" in m for m in quiet)


def test_improve_round_with_the_http_skill(nav):
    entry = nav.improve_round()
    assert set(entry["oracle"]) == {str(i) for i in range(1, entry["level"] + 2)}
    status = nav.status()
    assert "parámetros" in status and "dígitos" not in status
    assert Navros.load(nav.run_dir, log=lambda *_: None).skill.name == "http"  # la habilidad se persiste


def test_solve_renders_the_request(nav):
    out = nav.solve("obtén api.example.com/v1/items con id=7")
    assert out.startswith("obtén api.example.com/v1/items con id=7 → ") and out[-1] in "✓✗"


def test_navros_model_adapter_returns_only_the_continuation(nav):
    model = NavrosAgentModel(nav)
    prompt = "obtén api.example.com/v1/items con id=7\n"
    out = model(prompt)
    assert isinstance(out, str) and not out.startswith("obtén")
    assert 0 < model.context_chars < nav.model.cfg.max_seq_len * 16


def test_agent_loop_runs_with_navros_and_never_reaches_the_network(nav, tmp_path):
    http = HttpTool(HttpPolicy(allow_hosts=("api.example.com",)), resolver=offline, log_path=tmp_path / "a.jsonl")
    res = Agent(NavrosAgentModel(nav), http, max_steps=3, max_invalid=1).run("obtén api.example.com/v1/items con id=7")
    # Un modelo sin entrenar casi seguro escribe basura: lo importante es que el bucle termina limpio.
    assert res.stop in {"final", "max_steps", "invalid"} and res.steps
    assert res.transcript.startswith("obtén api.example.com/v1/items con id=7\n")
