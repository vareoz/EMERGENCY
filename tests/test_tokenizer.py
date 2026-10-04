from navros.tokenizer import BOS, EOS, FIRST_MERGE_ID, BPETokenizer, pretokenize

CORPUS = ["El modelo aprende. El modelo crece. El modelo se mejora a sí mismo. " * 20,
          "Ñandú, cigüeña y pingüino: 你好 🚀 x_y = 12345"]


def test_pretokenize_covers_everything():
    for text in CORPUS + ["  a\n\n b\t_ __ ²³", ""]:
        assert "".join(pretokenize(text)) == text


def test_roundtrip_and_specials():
    tok = BPETokenizer.train(CORPUS, 60)
    assert tok.vocab_size == FIRST_MERGE_ID + len(tok.merges)
    for text in CORPUS:
        ids = tok.encode(text, bos=True, eos=True)
        assert ids[0] == BOS and ids[-1] == EOS
        assert tok.decode(ids) == text
    assert len(tok.encode(CORPUS[0])) < len(CORPUS[0].encode())  # comprime


def test_digits_are_single_tokens():
    tok = BPETokenizer.train(["1234 1234 1234 5678 5678"] * 50, 40)
    assert tok.encode("90817") == [ord(c) for c in "90817"]


def test_extend_keeps_existing_ids(tmp_path):
    tok = BPETokenizer.train(CORPUS[:1], 30)
    before = {t: tok.encode(t) for t in ["El", " modelo"]}
    merges_before = list(tok.merges)
    added = tok.extend(["cuántico cuántico cuántico computación computación"] * 10, 10)
    assert added and tok.merges[: len(merges_before)] == merges_before
    assert all(new_id >= FIRST_MERGE_ID + len(merges_before) for new_id, _ in added)
    assert {t: tok.encode(t) for t in before} == before
    tok.save(tmp_path / "t.json")
    assert BPETokenizer.load(tmp_path / "t.json") == tok
