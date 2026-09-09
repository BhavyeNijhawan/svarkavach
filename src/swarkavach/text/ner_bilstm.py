"""BiLSTM sequence tagger for fraud entities, with an optional CRF output layer.

This is the neural counterpart to `ner_crf.CRFTagger`, and it exists to make the
comparison in the results table an honest one: same tokeniser, same label space,
same entity-level scoring, same train and test calls. The only thing that
changes is how the tagger scores a labelling.

Two design choices are worth stating.

The input is not words alone. A corpus of a few hundred short turns is far too
small to learn Hinglish word vectors from scratch, and no pretrained embedding
may be downloaded here, so each token is represented by its word embedding
concatenated with small embeddings of the same symbolic features the CRF uses:
coarse POS, gazetteer membership, language tag and a digit flag. That is what
lets a randomly initialised network reach a usable score on this corpus size.

The output layer is a CRF by default. A softmax over labels at each position
would let the network emit `I-OTP` straight after `O`, which is not a decodable
sequence. The transition parameters plus the same BIO constraint mask used in
`crf.py` make invalid output impossible, and the log-space forward algorithm
gives the exact sequence likelihood to train against.

Constraint scores here are a large negative constant rather than -inf. Autograd
propagates NaN through a logsumexp whose inputs are all -inf, which is the kind
of failure that shows up only after twenty epochs of silent damage.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import SETTINGS
from ..corpus.lexicon import gazetteer_type, is_digit_run
from ..schema import BIO_LABELS, Call, Turn, decode_bio, tokenize
from .langid import tag_languages
from .ner_crf import bio_constraints, entity_prf
from .pos import POS_TAGS, pos_tag

__all__ = ["BiLSTMTagger", "majority_baseline", "most_frequent_tag_baseline"]

_NEG = -1.0e4          # stands in for -inf, see the module docstring
_PAD, _UNK = 0, 1


def _lazy_torch():
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    return torch, nn


# --------------------------------------------------------------------------
# Baselines to compare against
# --------------------------------------------------------------------------


def majority_baseline(calls: Sequence[Call]) -> Dict[str, Any]:
    """Tag everything with the most frequent label, which is always O.

    Reported because it is the number a token-level accuracy figure hides
    behind: most tokens in any NER corpus are outside every entity, so a
    do-nothing tagger already looks accurate and scores exactly zero F1.
    """
    turns = [t for c in calls for t in c.turns if t.tokens]
    gold = [list(t.bio) for t in turns]
    pred = [["O"] * len(t.tokens) for t in turns]
    out = entity_prf(gold, pred)
    out["model"] = "majority"
    out["n_turns"] = len(turns)
    return out


def most_frequent_tag_baseline(train: Sequence[Call], test: Sequence[Call]) -> Dict[str, Any]:
    """Per-word most frequent gold tag, O for unseen words.

    A stronger baseline than majority class and the honest bar a sequence model
    has to clear: it captures every entity that is decided by the word alone.
    """
    from collections import Counter, defaultdict

    table: Dict[str, Counter] = defaultdict(Counter)
    for c in train:
        for t in c.turns:
            for w, b in zip(t.tokens, t.bio):
                table[w.lower()][b] += 1
    turns = [t for c in test for t in c.turns if t.tokens]
    gold = [list(t.bio) for t in turns]
    pred = [
        [table[w.lower()].most_common(1)[0][0] if w.lower() in table else "O" for w in t.tokens]
        for t in turns
    ]
    out = entity_prf(gold, pred)
    out["model"] = "most_frequent_tag"
    out["n_turns"] = len(turns)
    return out


# --------------------------------------------------------------------------
# Tagger
# --------------------------------------------------------------------------


class BiLSTMTagger:
    """Embedding, bidirectional LSTM, linear projection, optional CRF.

    Same `fit` / `predict_turn` / `evaluate` surface as `ner_crf.CRFTagger`.
    """

    name = "bilstm"

    def __init__(
        self,
        emb_dim: int = 64,
        hidden: int = 96,
        layers: int = 1,
        dropout: float = 0.25,
        lr: float = 5e-3,
        weight_decay: float = 1e-5,
        epochs: int = 40,
        batch_size: int = 16,
        use_crf: bool = True,
        min_freq: int = 1,
        seed: Optional[int] = None,
        max_seconds: float = 110.0,
        verbose: bool = False,
    ) -> None:
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_crf = use_crf
        self.min_freq = min_freq
        self.seed = SETTINGS.pipeline.seed if seed is None else seed
        self.max_seconds = max_seconds
        self.verbose = verbose

        self.labels_: List[str] = list(BIO_LABELS)
        self.label_index_ = {l: i for i, l in enumerate(self.labels_)}
        self.vocab_: Dict[str, int] = {}
        self.gaz_: Dict[str, int] = {}
        self.pos_: Dict[str, int] = {t: i for i, t in enumerate(POS_TAGS)}
        self.lang_: Dict[str, int] = {"hi": 0, "en": 1, "univ": 2}
        self.model = None
        self.history_: Dict[str, Any] = {}

    # -- encoding --------------------------------------------------------

    def _build_vocab(self, turns: Sequence[Turn]) -> None:
        from collections import Counter

        counts = Counter(w.lower() for t in turns for w in t.tokens)
        self.vocab_ = {"<pad>": _PAD, "<unk>": _UNK}
        for w, c in sorted(counts.items()):
            if c >= self.min_freq:
                self.vocab_[w] = len(self.vocab_)
        gaz_types = sorted({gazetteer_type(w) or "NONE" for t in turns for w in t.tokens})
        if "NONE" not in gaz_types:
            gaz_types.append("NONE")
        self.gaz_ = {g: i for i, g in enumerate(gaz_types)}

    def _encode(self, tokens: Sequence[str], langs: Optional[Sequence[str]] = None) -> np.ndarray:
        """Columns: word id, POS id, gazetteer id, language id, digit flag."""
        toks = list(tokens)
        pos = pos_tag(toks)
        if langs is None or len(langs) != len(toks):
            langs = tag_languages(toks)
        rows = np.zeros((len(toks), 5), dtype=np.int64)
        for i, w in enumerate(toks):
            low = w.lower()
            rows[i, 0] = self.vocab_.get(low, _UNK)
            rows[i, 1] = self.pos_.get(pos[i], self.pos_["X"])
            rows[i, 2] = self.gaz_.get(gazetteer_type(w) or "NONE", self.gaz_.get("NONE", 0))
            rows[i, 3] = self.lang_.get(langs[i], 2)
            rows[i, 4] = 2 if is_digit_run(w) else (1 if low.isdigit() else 0)
        return rows

    # -- network ---------------------------------------------------------

    def _build(self):
        torch, nn = _lazy_torch()
        L = len(self.labels_)
        a_trans, a_start, a_stop = bio_constraints(self.labels_)
        outer = self

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.w_emb = nn.Embedding(len(outer.vocab_), outer.emb_dim, padding_idx=_PAD)
                self.p_emb = nn.Embedding(len(outer.pos_), 12)
                self.g_emb = nn.Embedding(len(outer.gaz_), 12)
                self.l_emb = nn.Embedding(3, 6)
                self.d_emb = nn.Embedding(3, 4)
                in_dim = outer.emb_dim + 12 + 12 + 6 + 4
                self.lstm = nn.LSTM(
                    in_dim, outer.hidden, num_layers=outer.layers,
                    bidirectional=True, batch_first=True,
                    dropout=outer.dropout if outer.layers > 1 else 0.0,
                )
                self.drop = nn.Dropout(outer.dropout)
                self.proj = nn.Linear(2 * outer.hidden, L)
                self.transitions = nn.Parameter(torch.zeros(L, L))
                self.start = nn.Parameter(torch.zeros(L))
                self.stop = nn.Parameter(torch.zeros(L))
                self.register_buffer("m_trans", torch.where(
                    torch.from_numpy(a_trans), torch.zeros(L, L), torch.full((L, L), _NEG)))
                self.register_buffer("m_start", torch.where(
                    torch.from_numpy(a_start), torch.zeros(L), torch.full((L,), _NEG)))
                self.register_buffer("m_stop", torch.where(
                    torch.from_numpy(a_stop), torch.zeros(L), torch.full((L,), _NEG)))

            def emissions(self, x):
                e = torch.cat(
                    [self.w_emb(x[:, :, 0]), self.p_emb(x[:, :, 1]), self.g_emb(x[:, :, 2]),
                     self.l_emb(x[:, :, 3]), self.d_emb(x[:, :, 4])],
                    dim=-1,
                )
                h, _ = self.lstm(self.drop(e))
                return self.proj(self.drop(h))

            # -- CRF output layer ----------------------------------------

            def _eff(self):
                return (self.transitions + self.m_trans,
                        self.start + self.m_start,
                        self.stop + self.m_stop)

            def log_partition(self, em, mask):
                trans, start, stop = self._eff()
                B, T, L_ = em.shape
                alpha = start.unsqueeze(0) + em[:, 0]
                for t in range(1, T):
                    cand = em[:, t] + torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1)
                    m = mask[:, t].unsqueeze(1)
                    alpha = torch.where(m, cand, alpha)
                return torch.logsumexp(alpha + stop.unsqueeze(0), dim=1)

            def gold_score(self, em, tags, mask):
                trans, start, stop = self._eff()
                B, T, _ = em.shape
                idx = torch.arange(B, device=em.device)
                score = start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
                for t in range(1, T):
                    step = trans[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
                    score = score + step * mask[:, t].to(em.dtype)
                lengths = mask.sum(dim=1).long() - 1
                last = tags[idx, lengths]
                return score + stop[last]

            def nll(self, x, tags, mask):
                em = self.emissions(x)
                return (self.log_partition(em, mask) - self.gold_score(em, tags, mask)).mean()

            @torch.no_grad()
            def decode(self, x, mask):
                em = self.emissions(x)
                trans, start, stop = self._eff()
                B, T, L_ = em.shape
                delta = start.unsqueeze(0) + em[:, 0]
                back = torch.zeros(B, T, L_, dtype=torch.long)
                for t in range(1, T):
                    scores = delta.unsqueeze(2) + trans.unsqueeze(0)
                    best, arg = scores.max(dim=1)
                    back[:, t] = arg
                    cand = em[:, t] + best
                    m = mask[:, t].unsqueeze(1)
                    delta = torch.where(m, cand, delta)
                delta = delta + stop.unsqueeze(0)
                lengths = mask.sum(dim=1).long()
                paths = []
                best_last = delta.argmax(dim=1)
                for b in range(B):
                    n = int(lengths[b])
                    j = int(best_last[b])
                    seq = [j]
                    for t in range(n - 1, 0, -1):
                        j = int(back[b, t, j])
                        seq.append(j)
                    seq.reverse()
                    paths.append(seq)
                return paths

            @torch.no_grad()
            def decode_softmax(self, x, mask):
                """Independent per-token argmax, used when use_crf is False.

                It can emit sequences no BIO decoder accepts (I- with no B-
                before it), which is the cost of dropping the transition layer
                and is exactly what the comparison is meant to show.
                """
                arg = self.emissions(x).argmax(dim=-1)
                lengths = mask.sum(dim=1).long()
                return [arg[b, :int(lengths[b])].tolist() for b in range(x.shape[0])]

        return _Net()

    # -- batching --------------------------------------------------------

    def _batch(self, encoded: Sequence[np.ndarray], tags: Optional[Sequence[List[int]]] = None):
        torch, _ = _lazy_torch()
        B = len(encoded)
        T = max(len(e) for e in encoded)
        x = np.zeros((B, T, 5), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        y = np.zeros((B, T), dtype=np.int64)
        for i, e in enumerate(encoded):
            n = len(e)
            x[i, :n] = e
            mask[i, :n] = True
            if tags is not None:
                y[i, :n] = tags[i]
        return (torch.from_numpy(x), torch.from_numpy(mask),
                torch.from_numpy(y) if tags is not None else None)

    # -- api -------------------------------------------------------------

    def fit(self, calls: Sequence[Call]) -> "BiLSTMTagger":
        torch, nn = _lazy_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 31))
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        turns = [t for c in calls for t in c.turns if t.tokens]
        if not turns:
            raise ValueError("no non-empty turns to train on")
        self._build_vocab(turns)
        enc = [self._encode(t.tokens, t.lang) for t in turns]
        tags = [[self.label_index_[b] for b in t.bio] for t in turns]

        self.model = self._build()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss(ignore_index=-100)

        order = np.arange(len(enc))
        rng = np.random.default_rng(self.seed)
        t0 = time.time()
        losses: List[float] = []
        epochs_run = 0
        for epoch in range(self.epochs):
            rng.shuffle(order)
            self.model.train()
            total = 0.0
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                x, mask, y = self._batch([enc[i] for i in idx], [tags[i] for i in idx])
                opt.zero_grad()
                if self.use_crf:
                    loss = self.model.nll(x, y, mask)
                else:
                    em = self.model.emissions(x)
                    target = y.masked_fill(~mask, -100)
                    loss = ce(em.reshape(-1, len(self.labels_)), target.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                total += float(loss.detach()) * len(idx)
            losses.append(total / len(order))
            epochs_run = epoch + 1
            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch + 1:3d}  loss {losses[-1]:.4f}  {time.time() - t0:.1f}s")
            if time.time() - t0 > self.max_seconds:
                break
        self.model.eval()
        self.history_ = {
            "epochs": epochs_run,
            "seconds": float(time.time() - t0),
            "final_loss": float(losses[-1]) if losses else float("nan"),
            "vocab": len(self.vocab_),
            "n_params": int(sum(p.numel() for p in self.model.parameters())),
            "use_crf": self.use_crf,
            "n_turns": len(turns),
        }
        return self

    def predict_turn(self, tokens: Sequence[str], langs: Optional[Sequence[str]] = None) -> List[str]:
        if self.model is None:
            raise RuntimeError("BiLSTM is not fitted")
        toks = list(tokens)
        if not toks:
            return []
        return self.predict_batch([toks], [langs] if langs is not None else None)[0]

    def predict_batch(
        self,
        token_lists: Sequence[Sequence[str]],
        lang_lists: Optional[Sequence[Optional[Sequence[str]]]] = None,
    ) -> List[List[str]]:
        if self.model is None:
            raise RuntimeError("BiLSTM is not fitted")
        keep = [i for i, t in enumerate(token_lists) if len(t) > 0]
        out: List[List[str]] = [[] for _ in token_lists]
        if not keep:
            return out
        enc = [
            self._encode(token_lists[i], None if lang_lists is None else lang_lists[i])
            for i in keep
        ]
        x, mask, _ = self._batch(enc)
        paths = self.model.decode(x, mask) if self.use_crf else self.model.decode_softmax(x, mask)
        for i, path in zip(keep, paths):
            out[i] = [self.labels_[j] for j in path]
        return out

    def predict_call(self, call: Call) -> List[List[str]]:
        return self.predict_batch([t.tokens for t in call.turns], [t.lang for t in call.turns])

    def entities(self, tokens: Sequence[str], turn_index: int = 0):
        return decode_bio(list(tokens), self.predict_turn(tokens), turn_index=turn_index)

    def evaluate(self, calls: Sequence[Call]) -> Dict[str, Any]:
        turns = [t for c in calls for t in c.turns if t.tokens]
        gold = [list(t.bio) for t in turns]
        pred = self.predict_batch([t.tokens for t in turns], [t.lang for t in turns])
        out = entity_prf(gold, pred)
        out["model"] = self.name
        out["n_turns"] = len(turns)
        out["train_seconds"] = float(self.history_.get("seconds", 0.0))
        return out

    def save(self, path) -> None:
        torch, _ = _lazy_torch()
        torch.save(
            {
                "state": self.model.state_dict(),
                "vocab": self.vocab_, "gaz": self.gaz_, "labels": self.labels_,
                "cfg": {
                    "emb_dim": self.emb_dim, "hidden": self.hidden, "layers": self.layers,
                    "dropout": self.dropout, "use_crf": self.use_crf,
                },
                "history": self.history_,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "BiLSTMTagger":
        torch, _ = _lazy_torch()
        d = torch.load(path, weights_only=False)
        m = cls(**d["cfg"])
        m.vocab_ = d["vocab"]
        m.gaz_ = d["gaz"]
        m.labels_ = d["labels"]
        m.label_index_ = {l: i for i, l in enumerate(m.labels_)}
        m.model = m._build()
        m.model.load_state_dict(d["state"])
        m.model.eval()
        m.history_ = d.get("history", {})
        return m


if __name__ == "__main__":
    from .ner_crf import demo_calls

    calls = demo_calls(n_repeat=3)
    train = [c for c in calls if c.split == "train"]
    test = [c for c in calls if c.split == "test"]
    print(f"{len(train)} train calls, {len(test)} test calls, "
          f"{sum(len(c.turns) for c in train)} training turns")

    base = majority_baseline(test)
    freq = most_frequent_tag_baseline(train, test)
    print(f"majority (all O)  F1={base['f1']:.3f}  token acc={base['token_accuracy']:.3f}")
    print(f"most-frequent tag F1={freq['f1']:.3f}  token acc={freq['token_accuracy']:.3f}")

    tagger = BiLSTMTagger(epochs=12, verbose=True)
    tagger.fit(train)
    print("history:", tagger.history_)
    res = tagger.evaluate(test)
    print(f"bilstm+crf        F1={res['f1']:.3f}  P={res['precision']:.3f} "
          f"R={res['recall']:.3f}  token acc={res['token_accuracy']:.3f}")

    demo = tokenize("sir aapka account 10 minute me band ho jayega otp 445566 abhi bataiye")
    print(" ".join(f"{w}/{b}" for w, b in zip(demo, tagger.predict_turn(demo))))
    assert res["f1"] > base["f1"]
