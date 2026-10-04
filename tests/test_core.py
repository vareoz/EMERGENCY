import random

import pytest

from navros.core import Navros, NavrosTeacher, preset_config
from navros.distill import FunctionTeacher
from navros.skills import AdditionSkill


def test_addition_skill():
    sk = AdditionSkill()
    p = {"a": 128, "b": 95, "level": 3}
    assert sk.prompt(p) == "821+59=" and sk.target(p) == "322"
    assert sk.verify(p, "322") and not sk.verify(p, "3220") and not sk.verify(p, "x")
    assert sk.render(p, "322") == "128 + 95 = 223 ✓"
    rng = random.Random(0)
    for lvl in range(1, 6):
        q = sk.make_problem(rng, lvl)
        assert max(len(str(q["a"])), len(str(q["b"]))) == lvl


@pytest.fixture(scope="module")
def nav(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    return Navros.create(run, preset_config("tiny"), corpus=[], log=lambda *_: None)


def test_create_save_load(nav):
    assert (nav.run_dir / "model.pt").exists() and (nav.run_dir / "tokenizer.json").exists()
    again = Navros.load(nav.run_dir, log=lambda *_: None)
    assert again.tok == nav.tok and again.state["version"] == nav.state["version"]
    assert again.model.num_params() == nav.model.num_params()


def test_improve_round_is_recorded(nav):
    v = nav.state["version"]
    entry = nav.improve_round()
    assert nav.state["version"] == v + 1
    assert set(entry["oracle"]) == {str(i) for i in range(1, entry["level"] + 2)}
    assert (nav.run_dir / "pool.jsonl").exists()
    assert "ronda" in [e["event"] for e in nav.state["lineage"]]


def test_consensus_mode_round(nav):
    nav.cfg.improve.mode = "consensus"
    try:
        entry = nav.improve_round()
    finally:
        nav.cfg.improve.mode = "verifier"
    assert entry["mode"] == "consensus"


def test_grow_respects_max_params(nav):
    n = nav.model.num_params()
    assert nav.grow("depth") == "depth" and nav.model.num_params() > n
    nav.cfg.improve.max_params = nav.model.num_params()
    assert nav.grow("mlp") is None


def test_ingest_extends_tokenizer(nav, tmp_path):
    f = tmp_path / "nuevo.txt"
    f.write_text("computación cuántica superposición entrelazamiento " * 200)
    v0 = nav.tok.vocab_size
    info = nav.ingest([f], steps=5, new_merges=8)
    assert info["new_tokens"] > 0 and nav.tok.vocab_size == v0 + info["new_tokens"]
    assert nav.model.cfg.vocab_size == nav.tok.vocab_size


def test_assimilate_requires_license(nav):
    blocked = FunctionTeacher(lambda ps: ["0"] * len(ps), name="x", license="desconocida")
    with pytest.raises(PermissionError):
        nav.assimilate(blocked)


def test_assimilate_from_navros_teacher(nav, tmp_path):
    teacher = Navros.create(tmp_path / "t", preset_config("tiny", seed=1), log=lambda *_: None)
    info = nav.assimilate(NavrosTeacher(teacher), n_problems=16, steps=3)
    # tokenizadores distintos → destilación de secuencias
    assert info["method"] == "secuencias" and info["license"] == "propio"


def test_merge_same_lineage(nav, tmp_path):
    clone = Navros.load(nav.run_dir, log=lambda *_: None)
    clone.run_dir = tmp_path
    clone.save()
    nav.merge_with(clone, 0.5)
    assert nav.state["lineage"][-1]["event"] == "fusión"
