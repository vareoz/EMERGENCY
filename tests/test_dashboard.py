import json
import os
import time

import pytest

from navros.dashboard import PAGE, Controller, _fetch_url


def _write_run(run):
    run.mkdir(parents=True, exist_ok=True)
    (run / "state.json").write_text(json.dumps({
        "version": 7, "level": 5, "lr": 3e-3, "k": 2, "plateau": 0, "growths": 1,
        "created": 1_700_000_000.0,
        "history": [
            {"version": 6, "level": 4, "score_after": 0.8, "accepted": True, "params": 180000,
             "grew": None, "oracle": {"1": 1.0, "2": 1.0, "3": 0.9, "4": 0.4}},
            {"version": 7, "level": 5, "score_after": 0.95, "accepted": True, "params": 230000,
             "grew": "depth", "oracle": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 0.95, "5": 0.5}},
        ],
    }), encoding="utf-8")
    (run / "pool.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")


def test_snapshot_reads_state(tmp_path):
    run = tmp_path / "navros"
    _write_run(run)
    snap = Controller(run).snapshot()
    assert snap["run_exists"] and snap["alive"] is False
    assert snap["version"] == 7 and snap["level"] == 5
    assert snap["params"] == 230000 and snap["growths"] == 1
    assert snap["pool_size"] == 3
    assert snap["latest_oracle"]["5"] == 0.5
    assert len(snap["history"]) == 2


def test_snapshot_missing_run_is_graceful(tmp_path):
    snap = Controller(tmp_path / "noexiste").snapshot()
    assert snap["run_exists"] is False and snap["alive"] is False
    assert snap["version"] is None and snap["pool_size"] == 0


def test_is_alive_false_without_pid(tmp_path):
    assert Controller(tmp_path / "r").is_alive() is False


def test_delete_removes_run(tmp_path):
    run = tmp_path / "navros"
    _write_run(run)
    ctrl = Controller(run)
    assert ctrl.snapshot()["run_exists"]
    ctrl.delete()
    assert not run.exists() and ctrl.snapshot()["run_exists"] is False


def test_external_active_detection(tmp_path):
    run = tmp_path / "navros"
    _write_run(run)
    ctrl = Controller(run)
    # state.json recién escrito y sin proceso propio -> entrenamiento externo
    assert ctrl.is_alive() is False
    assert ctrl.external_active() is True
    snap = ctrl.snapshot()
    assert snap["external"] is True and snap["alive"] is False
    # start() se niega a lanzar un segundo escritor
    assert "no arranco" in ctrl.start().lower()
    # con state.json antiguo ya no cuenta como activo
    old = time.time() - 10_000
    os.utime(run / "state.json", (old, old))
    assert ctrl.external_active() is False


def test_ingest_rejects_non_http():
    with pytest.raises(ValueError):
        _fetch_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        _fetch_url("ftp://example.com/x")


def test_page_is_self_contained():
    # Sin CDNs externos: debe funcionar offline en local.
    assert "<title>NAVROS" in PAGE
    assert "http://" not in PAGE and "https://" not in PAGE
    assert "/api/state" in PAGE and "/api/delete" in PAGE
