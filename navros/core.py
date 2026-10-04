"""NAVROS: el motor de automejora.

Ciclo de cada ronda (``Navros.improve``):

1. **Proponer**: genera problemas en su frontera de dificultad (más repaso).
2. **Resolver**: muestrea k soluciones por problema con sus propios pesos.
3. **Filtrar**: se queda con las soluciones que pasan el verificador
   (modo ``verifier``) o en las que sus muestras coinciden (modo
   ``consensus``, sin ninguna respuesta de referencia).
4. **Aprender**: entrena candidatos con distintas tasas de aprendizaje sobre
   sus propios datos filtrados + el conjunto ancla (para no olvidar).
5. **Seleccionar**: acepta el mejor candidato solo si no empeora (si no, revierte).
6. **Adaptar**: sube de nivel al dominar la frontera, ajusta k y la tasa de
   aprendizaje y, si se estanca, **crece** (capas, cabezas, MLP).

Todo queda registrado (historial, linaje, procedencia de cada dato) y cada
ronda guarda un checkpoint, así que el bucle se puede reanudar en otra máquina.
"""

from __future__ import annotations

import copy
import json
import random
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from .distill import Teacher, check_license, logit_distill_loss, merge_state_dicts
from .model import ModelConfig, NavrosLM, generate
from .skills import SKILLS, Skill
from .tokenizer import BPETokenizer, EOS, PAD
from .trainer import (Example, Mixture, TrainConfig, autocast, best_device, complete,
                      skill_example, text_examples, train)


@dataclass
class ImproveConfig:
    mode: str = "verifier"  # "verifier" | "consensus"
    problems_per_round: int = 2048
    samples_per_problem: int = 4
    temperature: float = 0.8
    consensus_threshold: float = 0.75
    frontier_fraction: float = 0.75
    steps_per_round: int = 500
    anchor_weight: float = 0.25
    text_weight: float = 0.05
    frontier_weight: float = 0.35  # datos propios del nivel frontera (el resto va a niveles previos)
    population: int = 2
    lr_mutation: float = 2.0
    advance_threshold: float = 0.9
    patience: int = 2
    eval_problems: int = 512
    max_params: int = 50_000_000
    max_level: int = 40
    max_pool: int = 200_000


@dataclass
class NavrosConfig:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    improve: ImproveConfig = field(default_factory=ImproveConfig)
    skill: str = "suma"
    tokenizer_merges: int = 1000
    seed_level: int = 3
    seed_examples: int = 50_000
    pretrain_steps: int = 2000
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NavrosConfig":
        def build(kind, data):
            names = {f.name for f in fields(kind)}
            return kind(**{k: v for k, v in data.items() if k in names})
        rest = {k: v for k, v in d.items() if k not in ("model", "train", "improve")}
        return cls(model=build(ModelConfig, d["model"]), train=build(TrainConfig, d["train"]),
                   improve=build(ImproveConfig, d["improve"]), **rest)


# Tamaños orientativos por hardware. Todos se pueden ajustar a mano.
_POS = dict(pos="nope", abacus_offset=4)  # NoPE + Abacus acoplado: generaliza a números más largos

PRESETS: dict[str, dict] = {
    # Pruebas unitarias en CPU (segundos). No aprende nada útil.
    "tiny": dict(model=dict(d_model=32, n_layers=2, n_heads=2, head_dim=16, mlp_hidden=64, max_seq_len=64,
                            abacus=32, **_POS),
                 train=dict(batch_size=32, warmup=5), tokenizer_merges=50, seed_level=2,
                 seed_examples=500, pretrain_steps=30,
                 improve=dict(problems_per_round=64, samples_per_problem=2, steps_per_round=20,
                              eval_problems=32, population=1, max_params=2_000_000)),
    # Validación en CPU (minutos): ~150k parámetros.
    "mini": dict(model=dict(d_model=64, n_layers=3, n_heads=4, head_dim=16, mlp_hidden=256, max_seq_len=96,
                            abacus=64, **_POS),
                 train=dict(batch_size=64), tokenizer_merges=200, seed_examples=20_000, pretrain_steps=3000,
                 improve=dict(problems_per_round=1024, samples_per_problem=4, steps_per_round=600,
                              eval_problems=200, population=1, max_params=5_000_000)),
    # Kaggle (T4 / P100, 16 GB) o una VM pequeña: ~3M parámetros.
    "small": dict(model=dict(d_model=256, n_layers=4, n_heads=8, head_dim=32, mlp_hidden=1024, max_seq_len=256,
                             abacus=128, **_POS),
                  train=dict(batch_size=256), pretrain_steps=4000,
                  improve=dict(problems_per_round=4096, steps_per_round=800, max_params=30_000_000)),
    # Modal A10G / L4 / A100, Azure NC A100: ~20M parámetros.
    "base": dict(model=dict(d_model=512, n_layers=6, n_heads=8, head_dim=64, mlp_hidden=2048, max_seq_len=512,
                            abacus=256, **_POS),
                 train=dict(batch_size=512, lr=1e-3), tokenizer_merges=4000, seed_examples=200_000,
                 pretrain_steps=8000,
                 improve=dict(problems_per_round=16384, samples_per_problem=8, steps_per_round=1500,
                              eval_problems=1024, population=3, max_params=200_000_000)),
    # Modal H100 / Azure ND H100: ~100M parámetros de partida, crecimiento hasta 1B.
    "large": dict(model=dict(d_model=1024, n_layers=8, n_heads=16, head_dim=64, mlp_hidden=4096, max_seq_len=1024,
                             abacus=512, **_POS),
                  train=dict(batch_size=1024, lr=6e-4, warmup=200), tokenizer_merges=16000,
                  seed_examples=1_000_000, pretrain_steps=20000,
                  improve=dict(problems_per_round=65536, samples_per_problem=8, steps_per_round=3000,
                               eval_problems=2048, population=4, max_params=1_000_000_000)),
}


def preset_config(name: str, **overrides) -> NavrosConfig:
    p = copy.deepcopy(PRESETS[name])
    model = ModelConfig(vocab_size=0, **p.pop("model", {}))
    train_cfg = TrainConfig(**p.pop("train", {}))
    improve = ImproveConfig(**p.pop("improve", {}))
    p.update(overrides)
    return NavrosConfig(model=model, train=train_cfg, improve=improve, **p)


SEED_TEXT = (
    "NAVROS es un modelo de lenguaje que aprende de sus propios intentos. "
    "Propone problemas, los resuelve, verifica sus respuestas y entrena con lo que acierta. "
    "Cuando deja de mejorar, crece: añade capas, cabezas de atención y neuronas. "
    "Su tokenizador también crece cuando asimila textos nuevos.\n"
)


def stable_seed(*parts) -> int:
    """Semilla reproducible entre procesos (``hash()`` de str es aleatorio)."""
    return zlib.crc32(repr(parts).encode())


def read_texts(paths: list[str | Path]) -> list[str]:
    texts = []
    for p in map(Path, paths):
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".rst", ".py", ".json", ".csv", ""}:
                try:
                    texts.append(f.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    pass
    return texts


class Navros:
    def __init__(self, run_dir: Path, cfg: NavrosConfig, tok: BPETokenizer, model: NavrosLM,
                 state: dict, log=print):
        self.run_dir, self.cfg, self.tok, self.model, self.state = run_dir, cfg, tok, model, state
        self.skill: Skill = SKILLS[cfg.skill]()
        self.device = model.emb.weight.device
        self.log = log
        self._anchor_cache: list[Example] | None = None
        self._text_cache: tuple | None = None
        self._own_cache: tuple | None = None

    # -- creación / persistencia ---------------------------------------------
    @classmethod
    def create(cls, run_dir: str | Path, cfg: NavrosConfig, corpus: list[str | Path] = (),
               log=print) -> "Navros":
        run_dir = Path(run_dir)
        (run_dir / "corpus").mkdir(parents=True, exist_ok=True)
        texts = read_texts(list(corpus)) or [SEED_TEXT]
        for i, t in enumerate(texts):
            (run_dir / "corpus" / f"semilla_{i:04d}.txt").write_text(t, encoding="utf-8")
        torch.manual_seed(cfg.seed)
        log(f"[NAVROS] entrenando su tokenizador BPE ({cfg.tokenizer_merges} fusiones, {len(texts)} textos)…")
        tok = BPETokenizer.train(texts, cfg.tokenizer_merges)
        cfg.model.vocab_size = tok.vocab_size
        model = NavrosLM(cfg.model).to(best_device())
        state = {"version": 0, "level": cfg.seed_level + 1, "lr": cfg.train.lr,
                 "k": cfg.improve.samples_per_problem, "plateau": 0, "growths": 0,
                 "history": [], "lineage": [], "created": time.time()}
        nav = cls(run_dir, cfg, tok, model, state, log)
        log(f"[NAVROS] modelo inicial: {model.num_params():,} parámetros en {nav.device}; "
            f"vocabulario {tok.vocab_size}")
        log(f"[NAVROS] preentrenamiento: suma de 1 a {cfg.seed_level} dígitos con respuestas de referencia")
        train(model, nav._mixture(pool=False), cfg.pretrain_steps, cfg.train, log=log,
              log_every=max(cfg.pretrain_steps // 10, 1))
        nav._record("creación", {"params": model.num_params(), "vocab": tok.vocab_size})
        nav.save()
        return nav

    @classmethod
    def load(cls, run_dir: str | Path, log=print) -> "Navros":
        run_dir = Path(run_dir)
        cfg = NavrosConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        tok = BPETokenizer.load(run_dir / "tokenizer.json")
        dev = best_device()
        ckpt = torch.load(run_dir / "model.pt", map_location=dev, weights_only=True)
        model = NavrosLM.from_checkpoint(ckpt).to(dev)
        cfg.model = model.cfg
        state = json.loads((run_dir / "state.json").read_text())
        return cls(run_dir, cfg, tok, model, state, log)

    def save(self) -> None:
        self.cfg.model = self.model.cfg
        (self.run_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        self.tok.save(self.run_dir / "tokenizer.json")
        tmp = self.run_dir / "model.pt.tmp"
        torch.save(self.model.checkpoint(), tmp)
        tmp.replace(self.run_dir / "model.pt")  # escritura atómica: resiste cortes
        (self.run_dir / "state.json").write_text(json.dumps(self.state, indent=2))

    def _record(self, event: str, info: dict) -> None:
        self.state["lineage"].append({"version": self.state["version"], "event": event,
                                      "time": time.time(), **info})

    # -- datos -----------------------------------------------------------------
    def _anchor(self) -> list[Example]:
        """Conjunto ancla: niveles 1..seed_level con respuestas de referencia."""
        if self._anchor_cache is None:
            rng = random.Random(self.cfg.seed)
            self._anchor_cache = []
            for _ in range(self.cfg.seed_examples):
                p = self.skill.make_problem(rng, rng.randint(1, self.cfg.seed_level))
                self._anchor_cache.append(skill_example(self.tok, self.skill, p, self.skill.target(p)))
        return self._anchor_cache

    def _texts(self) -> list[str]:
        return read_texts([self.run_dir / "corpus"])

    def _text_examples(self) -> list[Example]:
        key = (self.tok.vocab_size, len(list((self.run_dir / "corpus").iterdir())), self.model.cfg.max_seq_len)
        if self._text_cache is None or self._text_cache[0] != key:
            self._text_cache = (key, text_examples(self.tok, self._texts(), self.model.cfg.max_seq_len))
        return self._text_cache[1]

    def _tokenize_item(self, it: dict) -> tuple[int, Example]:
        return it["problem"]["level"], skill_example(self.tok, self.skill, it["problem"], it["completion"])

    def _own_examples(self) -> tuple[list[Example], list[Example]]:
        """Datos autogenerados (y de maestros): (nivel frontera, resto). Caché incremental."""
        if self._own_cache is None or self._own_cache[0] != self.tok.vocab_size:
            self._own_cache = (self.tok.vocab_size, [self._tokenize_item(it) for it in self._pool()])
        items, L = self._own_cache[1][-self.cfg.improve.max_pool:], self.state["level"]
        return [e for lvl, e in items if lvl == L], [e for lvl, e in items if lvl != L]

    def _pool(self) -> list[dict]:
        path = self.run_dir / "pool.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-self.cfg.improve.max_pool:]
        return [json.loads(line) for line in lines]

    def _add_to_pool(self, items: list[dict]) -> None:
        with open(self.run_dir / "pool.jsonl", "a") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        if self._own_cache is not None:
            self._own_cache[1].extend(self._tokenize_item(it) for it in items)

    def _mixture(self, pool: bool = True, extra: list[Example] = ()) -> Mixture:
        ic = self.cfg.improve
        sources = [(self._anchor(), ic.anchor_weight if pool else 1.0), (self._text_examples(), ic.text_weight)]
        if pool:
            frontier, rest = self._own_examples()
            sources.append((frontier + list(extra), ic.frontier_weight))
            sources.append((rest, max(1.0 - ic.anchor_weight - ic.text_weight - ic.frontier_weight, 0.0)))
        return Mixture(sources, random.Random(self.cfg.seed + self.state["version"]))

    def _generator(self, salt: int = 0) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            self.cfg.seed * 7919 + self.state["version"] * 104729 + salt)

    # -- evaluación ----------------------------------------------------------------
    def _problems(self, level: int, n: int, salt: int) -> list[dict]:
        rng = random.Random(stable_seed(self.cfg.seed, self.state["version"], level, salt))
        return [self.skill.make_problem(rng, level) for _ in range(n)]

    def _agreement(self, model: NavrosLM, problems: list[dict], k: int) -> tuple[float, list]:
        """Fracción de problemas donde ≥ umbral de k muestras coinciden; y la mayoría."""
        rep = [p for p in problems for _ in range(k)]
        comps = complete(model, self.tok, self.skill, rep, self.cfg.improve.temperature,
                         generator=self._generator(len(problems)))
        majority, hits = [], 0
        for i in range(len(problems)):
            answers = [self.skill.answer_of(c) for c in comps[i * k:(i + 1) * k]]
            counts = Counter(a for a in answers if a is not None)
            if counts:
                ans, c = counts.most_common(1)[0]
                if c / k >= self.cfg.improve.consensus_threshold:
                    majority.append(ans)
                    hits += 1
                    continue
            majority.append(None)
        return hits / max(len(problems), 1), majority

    def oracle(self, model: NavrosLM | None = None, levels: range | None = None) -> dict[int, float]:
        """Exactitud real por nivel (medición; en modo consensus no decide nada)."""
        model = model or self.model
        levels = levels or range(1, self.state["level"] + 2)
        n = self.cfg.improve.eval_problems
        out = {}
        for lvl in levels:
            ps = self._problems(lvl, n, salt=1)
            comps = complete(model, self.tok, self.skill, ps)
            out[lvl] = sum(self.skill.verify(p, c) for p, c in zip(ps, comps)) / n
        return out

    def score(self, model: NavrosLM) -> tuple[float, dict]:
        """Señal de selección. Niveles ancla: verificador; frontera: según el modo."""
        ic, L = self.cfg.improve, self.state["level"]
        per_level = {}
        for lvl in range(1, L + 1):
            ps = self._problems(lvl, ic.eval_problems, salt=2)
            if ic.mode == "consensus" and lvl > self.cfg.seed_level:
                per_level[lvl] = self._agreement(model, ps, min(self.state["k"], 8))[0]
            else:
                comps = complete(model, self.tok, self.skill, ps)
                per_level[lvl] = sum(self.skill.verify(p, c) for p, c in zip(ps, comps)) / len(ps)
        return sum(per_level.values()) / len(per_level), per_level

    # -- automejora ------------------------------------------------------------------
    def improve(self, rounds: int = 1, max_hours: float | None = None, on_round=None) -> None:
        """``rounds=0`` = continuar hasta ``max_hours`` o hasta ``max_level``.

        ``on_round(entry)`` se llama tras cada ronda ya guardada (p. ej. para
        sincronizar un volumen de Modal o subir el checkpoint a la nube).
        """
        t0, done = time.time(), 0
        while rounds == 0 or done < rounds:
            if max_hours and (time.time() - t0) / 3600 >= max_hours:
                self.log("[NAVROS] presupuesto de tiempo agotado; estado guardado.")
                break
            if self.state["level"] > self.cfg.improve.max_level:
                self.log("[NAVROS] alcanzó max_level; súbelo en config.json para seguir.")
                break
            entry = self.improve_round()
            if on_round:
                on_round(entry)
            done += 1

    def improve_round(self) -> dict:
        ic, st, sk = self.cfg.improve, self.state, self.skill
        L, k, t0 = st["level"], st["k"], time.time()
        self.log(f"\n[NAVROS] ronda v{st['version'] + 1} · nivel frontera {L} dígitos · "
                 f"modo {ic.mode} · k={k} · lr={st['lr']:.2e} · {self.model.num_params():,} params")
        base_score, base_levels = self.score(self.model)

        # 1-3. proponer, resolver, filtrar
        rng = random.Random(stable_seed(self.cfg.seed, st["version"], "proponer"))
        problems = [sk.make_problem(rng, L if rng.random() < ic.frontier_fraction or L == 1
                                    else rng.randint(1, L - 1)) for _ in range(ic.problems_per_round)]
        if ic.mode == "verifier":
            rep = [p for p in problems for _ in range(k)]
            comps = complete(self.model, self.tok, sk, rep, ic.temperature, generator=self._generator())
            chosen = []
            for i, p in enumerate(problems):
                ok = [c for c in comps[i * k:(i + 1) * k] if sk.verify(p, c)]
                chosen.append(ok[0] if ok else None)
        else:
            _, chosen = self._agreement(self.model, problems, k)
        new_items = [{"problem": p, "completion": c, "source": f"self:{ic.mode}",
                      "version": st["version"] + 1} for p, c in zip(problems, chosen) if c is not None]
        yield_rate = len(new_items) / len(problems)
        n_front = sum(p["level"] == L for p in problems)
        front_yield = sum(it["problem"]["level"] == L for it in new_items) / max(n_front, 1)
        self.log(f"  autogeneró {len(problems) * k} intentos → {len(new_items)} aceptados "
                 f"({yield_rate:.1%}; en la frontera {front_yield:.1%})")
        self._add_to_pool(new_items)

        # 4-5. candidatos con distintas tasas de aprendizaje; selección con reversión
        lrs = [st["lr"], st["lr"] * ic.lr_mutation, st["lr"] / ic.lr_mutation][: max(ic.population, 1)]
        best = (base_score, None, None, base_levels)
        for lr in lrs:
            cand = copy.deepcopy(self.model)
            train(cand, self._mixture(), ic.steps_per_round, self.cfg.train, lr=lr)  # mismos datos para todos
            s, lv = self.score(cand)
            self.log(f"  candidato lr={lr:.2e}: puntuación {s:.3f}  (actual {base_score:.3f})")
            if s > best[0] or (best[1] is None and s >= base_score):
                best = (s, cand, lr, lv)
        accepted = best[1] is not None
        if accepted:
            self.model, st["lr"] = best[1], best[2]
        improved = best[0] > base_score + 0.005
        st["plateau"] = 0 if improved else st["plateau"] + 1
        levels = best[3]

        # 6. adaptación: nivel, k, crecimiento
        if levels.get(L, 0) >= ic.advance_threshold:
            st["level"] = L + 1
            st["plateau"] = 0
            self.log(f"  ★ dominó {L} dígitos → nueva frontera: {L + 1}")
        # k se adapta al rendimiento EN LA FRONTERA (el repaso casi siempre acierta)
        if front_yield < 0.15:
            st["k"] = min(k * 2, 64)
        elif front_yield > 0.6:
            st["k"] = max(k // 2, 2)
        grew = None
        if st["plateau"] >= ic.patience:
            grew = self.grow()
            st["plateau"] = 0

        st["version"] += 1
        oracle = self.oracle()
        entry = {"version": st["version"], "level": L, "mode": ic.mode, "accepted": accepted,
                 "score_before": base_score, "score_after": best[0], "yield": yield_rate,
                 "frontier_yield": front_yield,
                 "lr": st["lr"], "k": k, "params": self.model.num_params(), "grew": grew,
                 "oracle": {str(a): b for a, b in oracle.items()}, "seconds": time.time() - t0}
        st["history"].append(entry)
        self._record("ronda", {"accepted": accepted, "score": best[0], "grew": grew})
        self.save()
        self.log("  " + ("aceptado" if accepted else "rechazado (se conserva la versión anterior)")
                 + f" · exactitud real por nivel: "
                 + " ".join(f"{a}d={b:.0%}" for a, b in oracle.items()))
        return entry

    # -- crecimiento -------------------------------------------------------------------
    def grow(self, kind: str | None = None) -> str | None:
        """Crece preservando la función. Sin ``kind`` alterna capas → MLP → cabezas."""
        kinds = ["depth", "mlp", "heads"]
        kind = kind or kinds[self.state["growths"] % len(kinds)]
        trial = copy.deepcopy(self.model)
        if kind == "depth":
            trial.grow_depth(1)
        elif kind == "mlp":
            trial.grow_mlp(max(trial.cfg.mlp_hidden // 2, 1))
        elif kind == "heads":
            trial.grow_heads(max(trial.cfg.n_heads // 2, 1))
        else:
            raise ValueError(kind)
        if trial.num_params() > self.cfg.improve.max_params:
            self.log(f"  límite de parámetros ({self.cfg.improve.max_params:,}) alcanzado: "
                     "sube improve.max_params o usa una GPU mayor.")
            return None
        before = self.model.num_params()
        self.model = trial
        self.state["growths"] += 1
        self._record("crecimiento", {"kind": kind, "params_before": before,
                                     "params_after": trial.num_params()})
        self.log(f"  ↑ creció ({kind}): {before:,} → {trial.num_params():,} parámetros")
        return kind

    # -- conocimiento nuevo ----------------------------------------------------------------
    def ingest(self, paths: list[str | Path], steps: int = 500, min_gain: float = 0.05,
               new_merges: int | None = None) -> dict:
        """Asimila textos: los añade al corpus, amplía el tokenizador si la compresión
        empeora y entrena sobre ellos."""
        texts = read_texts(paths)
        if not texts:
            raise ValueError("no se encontraron textos legibles")
        old_bpt = self.tok.bytes_per_token(self._texts()[:200])
        new_bpt = self.tok.bytes_per_token(texts[:200])
        corpus = self.run_dir / "corpus"
        n0 = len(list(corpus.glob("*.txt")))
        for i, t in enumerate(texts):
            (corpus / f"ingesta_{n0 + i:05d}.txt").write_text(t, encoding="utf-8")
        added = []
        if new_bpt < old_bpt * (1 - min_gain):
            n = new_merges or max(self.cfg.tokenizer_merges // 4, 16)
            added = self.tok.extend(texts, n)
            self.model.grow_vocab(added)
            self._anchor_cache = None
            self.log(f"[NAVROS] tokenizador ampliado con {len(added)} tokens "
                     f"({new_bpt:.2f} → {self.tok.bytes_per_token(texts[:200]):.2f} bytes/token)")
        mixture = Mixture([(text_examples(self.tok, texts, self.model.cfg.max_seq_len), 0.7),
                           (self._anchor(), 0.3)], random.Random(self.state["version"]))
        loss = train(self.model, mixture, steps, self.cfg.train, lr=self.state["lr"], log=self.log,
                     log_every=max(steps // 5, 1))
        self.state["version"] += 1
        self._record("ingesta", {"texts": len(texts), "new_tokens": len(added), "loss": loss})
        self.save()
        return {"texts": len(texts), "new_tokens": len(added), "loss": loss}

    # -- asimilación de otros modelos -----------------------------------------------------
    def assimilate(self, teacher: Teacher, n_problems: int = 2048, steps: int | None = None) -> dict:
        """Aprende de otro modelo. NavrosTeacher compatible → logits; si no → secuencias."""
        check_license(teacher)
        steps = steps or self.cfg.improve.steps_per_round
        before, _ = self.score(self.model)
        rng = random.Random(stable_seed("asimilar", self.cfg.seed, self.state["version"]))
        top = max(self.state["level"], getattr(teacher, "level", 0))
        problems = [self.skill.make_problem(rng, rng.randint(1, top)) for _ in range(n_problems)]
        prompts = [self.skill.prompt(p) for p in problems]
        answers = [c[len(pr):] if c.startswith(pr) else c for pr, c in zip(prompts, teacher.complete(prompts))]
        kept = [{"problem": p, "completion": a.strip(), "source": f"teacher:{teacher.name}",
                 "version": self.state["version"] + 1}
                for p, a in zip(problems, answers) if self.skill.verify(p, a)]
        self.log(f"[NAVROS] {teacher.name}: {len(kept)}/{n_problems} respuestas verificadas")
        self._add_to_pool(kept)
        cand = copy.deepcopy(self.model)
        t_model = getattr(teacher, "model", None)
        same_tok = getattr(teacher, "tok", None) == self.tok
        loss_fn = logit_distill_loss(t_model) if t_model is not None and same_tok else None
        train(cand, self._mixture(), steps, self.cfg.train, lr=self.state["lr"], loss_fn=loss_fn)
        after, _ = self.score(cand)
        accepted = after >= before
        if accepted:
            self.model = cand
        if getattr(teacher, "level", 0) > self.state["level"] and accepted:
            self.state["level"] = teacher.level
        self.state["version"] += 1
        info = {"teacher": teacher.name, "license": teacher.license, "kept": len(kept),
                "method": "logits" if loss_fn else "secuencias", "score_before": before,
                "score_after": after, "accepted": accepted}
        self._record("asimilación", info)
        self.save()
        self.log(f"  puntuación {before:.3f} → {after:.3f} · " + ("aceptado" if accepted else "rechazado"))
        return info

    def attach_quantum(self, n_qubits: int = 4, n_layers: int = 2) -> None:
        """Añade el adaptador cuántico híbrido (no cambia la salida hasta entrenar)."""
        self.model.attach_quantum(n_qubits, n_layers)
        self.state["version"] += 1
        self._record("cuántico", {"qubits": n_qubits, "layers": n_layers})
        self.save()

    def merge_with(self, other: "Navros", weight: float = 0.5) -> None:
        if other.tok != self.tok:
            raise ValueError("los tokenizadores difieren: usa assimilate() en su lugar")
        merged = merge_state_dicts([self.model.state_dict(), other.model.state_dict()],
                                   [1 - weight, weight])
        self.model.load_state_dict(merged)
        self.state["level"] = max(self.state["level"], other.state["level"])
        self.state["version"] += 1
        self._record("fusión", {"with": str(other.run_dir), "weight": weight})
        self.save()

    # -- uso -------------------------------------------------------------------------------
    def solve(self, expr: str) -> str:
        p = self.skill.parse_human(expr)
        return self.skill.render(p, complete(self.model, self.tok, self.skill, [p])[0])

    def generate_text(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8) -> str:
        max_new_tokens = min(max_new_tokens, self.model.cfg.max_seq_len // 2)
        ids = self.tok.encode(prompt, bos=True)[-(self.model.cfg.max_seq_len - max_new_tokens):]
        with autocast(self.device):
            out = generate(self.model, [ids], max_new_tokens, temperature, eos_id=EOS, pad_id=PAD,
                           generator=self._generator())
        return prompt + self.tok.decode(out[0])

    def status(self) -> str:
        st, m = self.state, self.model.cfg
        lines = [f"NAVROS · {self.run_dir}",
                 f"  versión {st['version']} · frontera {st['level']} dígitos · modo {self.cfg.improve.mode}",
                 f"  {self.model.num_params():,} parámetros · capas {m.n_layers} · cabezas {m.n_heads} · "
                 f"MLP {m.mlp_hidden} · d_model {m.d_model} · vocab {m.vocab_size}"
                 + (f" · adaptador cuántico {m.quantum_qubits} qubits" if m.quantum_qubits else ""),
                 f"  lr {st['lr']:.2e} · k {st['k']} · crecimientos {st['growths']} · "
                 f"datos propios {len(self._pool()):,}"]
        for h in st["history"][-10:]:
            orc = " ".join(f"{a}d={b:.0%}" for a, b in h["oracle"].items())
            lines.append(f"  v{h['version']:>3} {'✓' if h['accepted'] else '·'} "
                         f"{h['score_before']:.3f}→{h['score_after']:.3f} frontera {h.get('frontier_yield', 0):.0%} | {orc}"
                         + (f" | creció {h['grew']}" if h.get("grew") else ""))
        return "\n".join(lines)


class NavrosTeacher:
    """Otro modelo NAVROS como maestro (misma licencia: tuyo)."""

    license = "propio"
    allows_distillation = True

    def __init__(self, nav: Navros):
        self.nav, self.model, self.tok = nav, nav.model, nav.tok
        self.name = f"navros:{nav.run_dir.name}"
        self.level = nav.state["level"]

    def complete(self, prompts: list[str]) -> list[str]:
        ids = [self.tok.encode(p, bos=True) for p in prompts]
        out = []
        for i in range(0, len(ids), 512):
            with autocast(self.nav.device):
                gen = generate(self.model, ids[i:i + 512], self.nav.skill.max_answer_tokens(self.level + 1),
                               eos_id=EOS, pad_id=PAD)
            out.extend(self.tok.decode(g) for g in gen)
        return out
