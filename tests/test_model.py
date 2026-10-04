import pytest
import torch

from navros.model import ModelConfig, NavrosLM, generate


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=300, d_model=32, n_layers=2, n_heads=2, head_dim=16, mlp_hidden=64, max_seq_len=32)
    cfg.update(kw)
    m = NavrosLM(ModelConfig(**cfg))
    # Pesos no triviales para que la prueba de preservación sea exigente.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.eval()


X = torch.randint(0, 300, (3, 20))


@pytest.mark.parametrize("pos", [dict(pos="rope"), dict(pos="nope", abacus=16)])
@pytest.mark.parametrize("grow", [
    lambda m: m.grow_depth(2),
    lambda m: m.grow_heads(1),
    lambda m: m.grow_mlp(32),
    lambda m: m.attach_quantum(3, 1),
])
def test_growth_preserves_function(pos, grow):
    m = _model(**pos)
    before = m(X)
    n0 = m.num_params()
    grow(m)
    assert m.num_params() > n0
    torch.testing.assert_close(m.eval()(X), before, atol=1e-5, rtol=1e-5)


def test_vocab_growth_keeps_old_logits():
    m = _model()
    before = m(X)
    m.grow_vocab([(300, (5, 6)), (301, (300, 7))])
    after = m(X)
    assert after.shape[-1] == 302
    torch.testing.assert_close(after[..., :300], before)
    torch.testing.assert_close(m.emb.weight[300], 0.5 * (m.emb.weight[5] + m.emb.weight[6]))


def test_checkpoint_roundtrip_after_growth():
    m = _model()
    m.grow_depth(1)
    m.grow_heads(1)
    m.attach_quantum(2, 1)
    m2 = NavrosLM.from_checkpoint(m.checkpoint()).eval()
    torch.testing.assert_close(m2(X), m(X))


def test_generate_respects_lengths_and_eos():
    m = _model()
    out = generate(m, [[1, 2, 3], [4]], max_new_tokens=5, eos_id=None)
    assert [len(o) for o in out] == [5, 5]
    # Coincide con decodificar uno a uno (el relleno a la derecha no interfiere).
    solo = generate(m, [[4]], max_new_tokens=5, eos_id=None)
    assert out[1] == solo[0]
    eos = out[0][0]
    assert generate(m, [[1, 2, 3]], max_new_tokens=5, eos_id=eos) == [[]]


def test_digit_positions_abacus_coupled():
    m = _model(abacus=16, abacus_offset=0)
    ids = torch.tensor([[256] + [ord(c) for c in "821+59=322"]])
    #                 bos  8  2  1  +  5  9  =  3  2  2
    assert m.digit_positions(ids).tolist() == [[0, 1, 2, 3, 0, 1, 2, 1, 2, 3, 4]]
    m.cfg.abacus_offset = 3
    m.train()
    shifted = m.digit_positions(ids)
    off = shifted[0, 1] - 1
    assert 0 <= off <= 3 and (shifted[0, [0, 4]] == 0).all()
    assert shifted.tolist() == [[0] + [v + int(off) if v else 0 for v in [1, 2, 3, 0, 1, 2, 1, 2, 3, 4]]]
