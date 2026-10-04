import math

import torch

from navros.quantum import (CallableBackend, SimulatorBackend, VariationalCircuit,
                            expectations_from_counts, simulate)


def test_single_qubit_rotation():
    for theta in (0.0, 0.7, math.pi / 2, 2.5):
        z = simulate(1, [("ry", (0,), theta)]).expect_z()
        assert abs(z.item() - math.cos(theta)) < 1e-5


def test_bell_state():
    sv = simulate(2, [("h", (0,), None), ("cx", (0, 1), None)])
    torch.testing.assert_close(sv.probabilities()[0], torch.tensor([0.5, 0, 0, 0.5]))


def test_rx_rz_and_cnot_direction():
    # |10⟩ con CNOT(0→1) → |11⟩ ; CNOT(1→0) sobre |10⟩ no cambia nada
    flip = [("rx", (0,), math.pi)]
    assert simulate(2, flip + [("cx", (0, 1), None)]).probabilities()[0].argmax() == 3
    assert simulate(2, flip + [("cx", (1, 0), None)]).probabilities()[0].argmax() == 2
    # RZ solo cambia fases: no altera probabilidades
    sv = simulate(1, [("h", (0,), None), ("rz", (0,), 1.3)])
    torch.testing.assert_close(sv.probabilities()[0], torch.tensor([0.5, 0.5]))


def test_parameter_shift_matches_autograd():
    torch.manual_seed(0)
    vqc = VariationalCircuit(3, 2)
    x = torch.rand(4, 3)
    w = torch.randn(4, 3)
    (vqc(x) * w).sum().backward()
    torch.testing.assert_close(vqc.parameter_shift_grad(x, w), vqc.theta.grad, atol=1e-4, rtol=1e-4)


def test_qasm_runs_on_backend():
    torch.manual_seed(1)
    vqc = VariationalCircuit(3, 2)
    x = torch.rand(2, 3)
    qasm = vqc.to_qasm(x[0])
    assert qasm.startswith("OPENQASM 2.0;") and "measure q -> c;" in qasm
    measured = vqc.run_on_backend(x, SimulatorBackend(seed=0), shots=20000)
    torch.testing.assert_close(measured, vqc(x).detach(), atol=0.03, rtol=0)


def test_counts_conventions():
    assert expectations_from_counts({"01": 10}, 2).tolist() == [1.0, -1.0]
    little_endian = CallableBackend(lambda q, s: {"10": s}, qubit0_last=True)
    assert little_endian.run("", 5) == {"01": 5}
