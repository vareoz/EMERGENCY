"""Tokenizador BPE a nivel de bytes, entrenado por NAVROS desde cero.

- Vocabulario base: los 256 bytes (cualquier texto Unicode es representable).
- Tokens especiales fijos: <|bos|>, <|eos|>, <|pad|>.
- Fusiones BPE aprendidas del corpus propio; se pueden *extender* después
  (``extend``) sin cambiar los ids existentes, de modo que el modelo puede
  hacer crecer su vocabulario a medida que asimila texto nuevo.
- Los dígitos se separan siempre de uno en uno: la aritmética se aprende
  sobre dígitos individuales, lo que facilita generalizar a números largos.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>"]
BOS, EOS, PAD = 256, 257, 258
FIRST_MERGE_ID = 256 + len(SPECIALS)

# Pre-tokenización: dígitos sueltos, palabras (con espacio inicial opcional),
# puntuación, guiones bajos y espacios. Cubre cualquier carácter.
_PRETOKENIZE = re.compile(r"\d| ?[^\W\d_]+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> list[str]:
    return _PRETOKENIZE.findall(text)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = []
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, s in enumerate(SPECIALS):
            self.vocab[256 + i] = s.encode()
        self._cache: dict[str, list[int]] = {}
        for pair in merges or []:
            self._add_merge(pair)

    # -- propiedades -------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    def _add_merge(self, pair: tuple[int, int]) -> int:
        new_id = FIRST_MERGE_ID + len(self.merges)
        self.merges.append(pair)
        self.ranks[pair] = len(self.merges) - 1
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._cache.clear()
        return new_id

    # -- entrenamiento -----------------------------------------------------
    def _learn(self, texts: list[str], n_merges: int) -> list[tuple[int, tuple[int, int]]]:
        """Aprende ``n_merges`` fusiones nuevas sobre la tokenización actual."""
        chunk_freq = Counter(c for t in texts for c in pretokenize(t))
        words = {c: self._encode_chunk(c) for c in chunk_freq}
        learned = []
        for _ in range(n_merges):
            pairs: Counter = Counter()
            for c, ids in words.items():
                f = chunk_freq[c]
                for p in zip(ids, ids[1:]):
                    pairs[p] += f
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:
                break
            new_id = self._add_merge(best)
            learned.append((new_id, best))
            for c, ids in words.items():
                if len(ids) > 1:
                    words[c] = _merge(ids, best, new_id)
        return learned

    @classmethod
    def train(cls, texts: list[str], n_merges: int) -> "BPETokenizer":
        tok = cls()
        tok._learn(texts, n_merges)
        return tok

    def extend(self, texts: list[str], n_merges: int) -> list[tuple[int, tuple[int, int]]]:
        """Amplía el vocabulario con texto nuevo. Los ids previos no cambian.

        Devuelve ``[(nuevo_id, (id_a, id_b)), ...]`` para que el modelo pueda
        inicializar los embeddings nuevos a partir de sus componentes.
        """
        return self._learn(texts, n_merges)

    # -- codificación ------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 60))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, FIRST_MERGE_ID + self.ranks[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [BOS] if bos else []
        for chunk in pretokenize(text):
            ids.extend(self._encode_chunk(chunk))
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids if i not in (BOS, EOS, PAD))
        return data.decode("utf-8", errors="replace")

    def bytes_per_token(self, texts: list[str]) -> float:
        """Compresión: bytes de UTF-8 por token (más alto = vocabulario más afín)."""
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tok = sum(len(self.encode(t)) for t in texts)
        return n_bytes / max(n_tok, 1)

    # -- persistencia ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "navros-bpe", "merges": self.merges}))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in data["merges"]])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BPETokenizer) and self.merges == other.merges
