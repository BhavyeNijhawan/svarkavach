"""TF-IDF scam-intent classifier, turn level and call level.

Word n-grams alone underperform badly on romanised Hindi. There is no spelling
standard, so "bataiye", "batayiye", "bataye" and "batao" are four separate
vocabulary items to a word model, and each one is rare. Character n-grams share
substrings across all of them, and they also survive the substitutions an ASR
makes on telephone-band audio. Both views are used here: word 1-2 grams give
the interpretable lexical evidence, character 3-5 grams absorb the spelling
variation. The union is what the classifier sees.

`char_wb` is used rather than plain `char` so n-grams do not straddle word
boundaries, which would otherwise manufacture features out of whitespace.

Call-level scoring is a second stage, not a re-run of the turn model on
concatenated text. A scam call is not uniformly scammy: the opening turns are
indistinguishable from a real bank call and the pressure arrives late. Pooling
therefore keeps both the maximum caller-turn score (did any single turn cross
the line) and the mean (how much of the call was like that), and adds entity
counts, since an OTP request plus an authority claim is evidence a bag of words
spreads too thinly to see. The turn scores that train the pooling stage come
from cross-validated predictions, so the pooler is not fitted on the turn
model's own overconfident in-sample output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..corpus.lexicon import URGENCY_TERMS, canonical
from ..schema import Call, EntitySpan, tokenize
from .normalize import normalize_text

__all__ = ["TfidfIntent", "ENTITY_FEATURE_MAP"]

#: Which fusion feature each entity type feeds. MONEY_AMOUNT has no slot of its
#: own in `schema.FUSION_FEATURES`, so it is folded into the payment count.
ENTITY_FEATURE_MAP: Dict[str, str] = {
    "OTP": "ent_otp",
    "AUTHORITY_CLAIM": "ent_authority",
    "THREAT_DEADLINE": "ent_threat",
    "PAYMENT_HANDLE": "ent_payment",
    "PERSONAL_INFO_REQ": "ent_personal",
    "BANK_ENTITY": "ent_bank",
    # MONEY_AMOUNT deliberately absent. Folding it into ent_payment here while
    # fusion/featurize.ENT_FEATURE left it out meant one feature name carried
    # two different quantities inside a single feature vector, depending on
    # which half of the code computed it.
}

_ENT_FEATURES: Tuple[str, ...] = (
    "ent_otp", "ent_authority", "ent_threat", "ent_payment", "ent_personal", "ent_bank",
)

_URGENCY_TOKENS = {t for term in URGENCY_TERMS for t in term.split() if len(t) > 2}


def _urgency_density(tokens: Sequence[str]) -> float:
    """Urgency words per 100 tokens, on the scale the fusion contract asks for."""
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t.lower() in _URGENCY_TOKENS or canonical(t) in _URGENCY_TOKENS)
    return 100.0 * hits / len(tokens)


class TfidfIntent:
    """Turn-level scam classifier plus a call-level pooling stage."""

    name = "tfidf"

    def __init__(
        self,
        model: str = "logreg",
        word_ngrams: Tuple[int, int] = (1, 2),
        char_ngrams: Tuple[int, int] = (3, 5),
        C: float = 4.0,
        min_df: int = 1,
        seed: Optional[int] = None,
        caller_only: bool = True,
    ) -> None:
        self.model_kind = model
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.C = float(C)
        self.min_df = int(min_df)
        self.seed = SETTINGS.pipeline.seed if seed is None else int(seed)
        self.caller_only = caller_only
        self.pipeline_ = None
        self.pooler_ = None
        self.history_: Dict[str, Any] = {}

    # -- data ------------------------------------------------------------

    def _turn_text(self, tokens: Sequence[str]) -> str:
        return normalize_text(" ".join(tokens))

    def _training_turns(self, calls: Sequence[Call]) -> Tuple[List[str], List[int]]:
        texts: List[str] = []
        labels: List[int] = []
        for c in calls:
            for t in c.turns:
                if self.caller_only and t.speaker != "caller":
                    continue
                if not t.tokens:
                    continue
                texts.append(self._turn_text(t.tokens))
                labels.append(int(c.label_scam))
        return texts, labels

    def _build_pipeline(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC

        union = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=self.word_ngrams, min_df=self.min_df,
                sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\b\w+\b")),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=self.char_ngrams,
                min_df=max(2, self.min_df), sublinear_tf=True, lowercase=True)),
        ])
        if self.model_kind == "svc":
            clf = CalibratedClassifierCV(
                LinearSVC(C=self.C, random_state=self.seed), method="sigmoid", cv=3
            )
        else:
            clf = LogisticRegression(
                C=self.C, max_iter=3000, class_weight="balanced", random_state=self.seed
            )
        return Pipeline([("feats", union), ("clf", clf)])

    # -- fitting ---------------------------------------------------------

    def fit(self, calls: Sequence[Call]) -> "TfidfIntent":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        texts, labels = self._training_turns(calls)
        if len(set(labels)) < 2:
            raise ValueError("need both scam and benign calls to fit the intent model")
        y = np.asarray(labels)
        self.pipeline_ = self._build_pipeline()

        # Out-of-fold turn scores, so the pooling stage below sees the kind of
        # scores it will meet at test time rather than memorised ones.
        n_splits = int(min(5, max(2, np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(
            self._build_pipeline(), texts, y, cv=cv, method="predict_proba"
        )[:, 1]
        self.pipeline_.fit(texts, y)

        # Map the out-of-fold turn scores back onto their calls.
        pos = 0
        rows: List[List[float]] = []
        call_y: List[int] = []
        for c in calls:
            n = sum(
                1 for t in c.turns
                if t.tokens and (not self.caller_only or t.speaker == "caller")
            )
            scores = list(oof[pos:pos + n])
            pos += n
            # No spans here on purpose: this is the fit, and reading the
            # gold annotation off the training calls is what training is.
            rows.append(self._pool_features(c, scores))
            call_y.append(int(c.label_scam))
        self.pooler_ = LogisticRegression(
            C=2.0, max_iter=2000, class_weight="balanced", random_state=self.seed
        ).fit(np.asarray(rows), np.asarray(call_y))

        self.history_ = {
            "n_calls": len(calls),
            "n_turns": len(texts),
            "n_features": int(self.pipeline_.named_steps["feats"].transform(texts[:1]).shape[1]),
            "model": self.model_kind,
            "pool_features": self.pool_feature_names(),
        }
        return self

    # -- pooling ---------------------------------------------------------

    @staticmethod
    def pool_feature_names() -> List[str]:
        return ["max", "mean", "top2_mean", "frac_above_half", "log_n_turns"] + list(_ENT_FEATURES)

    def _entity_counts(self, call: Call,
                       spans: Optional[Sequence[EntitySpan]] = None) -> Dict[str, float]:
        """Entity counts per 100 tokens.

        `spans` is what the NER tagger predicted. Pass it at inference. Leave
        it None only while fitting, where reading the gold annotation off
        `Turn.bio` is legitimate because that is what training data is.

        This used to always read `Turn.bio`, with a docstring asserting that
        at inference the tagger would have overwritten it. Nothing ever did:
        `CRFTagger.tag_call` exists and is called from nowhere. So six of the
        eleven pooler features were gold annotations on held-out calls, and
        since the pooler's output overwrites `intent_score`, the single
        highest weighted fusion feature was a function of the answer key.
        That is the same leak `featurize.tag_entities` was fixed for, arriving
        through a different door, and it moved the mean scam score on the test
        split from 0.741 to 0.960.
        """
        counts = {k: 0.0 for k in _ENT_FEATURES}
        n_tokens = sum(len(t.tokens) for t in call.turns)
        if spans is None:
            spans = [sp for t in call.turns for sp in t.entities()]
        for span in spans:
            key = ENTITY_FEATURE_MAP.get(span.type)
            if key:
                counts[key] += 1.0
        scale = 100.0 / max(1, n_tokens)
        return {k: v * scale for k, v in counts.items()}

    def _pool_features(self, call: Call, turn_scores: Sequence[float],
                       spans: Optional[Sequence[EntitySpan]] = None) -> List[float]:
        s = np.asarray(list(turn_scores), dtype=float)
        if s.size == 0:
            s = np.zeros(1)
        top2 = np.sort(s)[-2:].mean()
        ent = self._entity_counts(call, spans)
        return [
            float(s.max()),
            float(s.mean()),
            float(top2),
            float((s > 0.5).mean()),
            float(np.log1p(s.size)),
        ] + [float(ent[k]) for k in _ENT_FEATURES]

    # -- scoring ---------------------------------------------------------

    def score_turn(self, tokens: Sequence[str]) -> float:
        if self.pipeline_ is None:
            raise RuntimeError("TfidfIntent is not fitted")
        if not list(tokens):
            return 0.0
        return float(self.pipeline_.predict_proba([self._turn_text(tokens)])[0, 1])

    def score_text(self, text: str) -> float:
        return self.score_turn(tokenize(text))

    def score_turns(self, token_lists: Sequence[Sequence[str]]) -> List[float]:
        if self.pipeline_ is None:
            raise RuntimeError("TfidfIntent is not fitted")
        keep = [i for i, t in enumerate(token_lists) if len(t) > 0]
        out = [0.0] * len(token_lists)
        if not keep:
            return out
        probs = self.pipeline_.predict_proba([self._turn_text(token_lists[i]) for i in keep])[:, 1]
        for i, p in zip(keep, probs):
            out[i] = float(p)
        return out

    def score_call(self, call: Call,
                   spans: Optional[Sequence[EntitySpan]] = None) -> Dict[str, float]:
        """Call-level score plus the intent-group fusion features.

        Keys match `schema.FUSION_FEATURES` for the intent group, so the fusion
        layer can merge the result without renaming anything.

        `spans` are the tagger's predictions. Callers that have them must pass
        them; see `_entity_counts` for what happens when they do not.
        """
        caller = [t for t in call.turns if t.speaker == "caller" and t.tokens]
        turns = caller if (self.caller_only and caller) else [t for t in call.turns if t.tokens]
        scores = self.score_turns([t.tokens for t in turns])
        feats = self._pool_features(call, scores, spans)
        if self.pooler_ is not None:
            call_score = float(self.pooler_.predict_proba(np.asarray([feats]))[0, 1])
        else:
            call_score = float(np.mean(scores)) if scores else 0.0
        all_caller_tokens = [w for t in turns for w in t.tokens]
        ent = self._entity_counts(call, spans)
        out = {
            "intent_score": call_score,
            "intent_max_turn": float(max(scores) if scores else 0.0),
            "urgency_density": float(_urgency_density(all_caller_tokens)),
        }
        out.update({k: float(ent[k]) for k in _ENT_FEATURES})
        out["per_turn"] = [float(s) for s in scores]
        out["turn_index"] = [int(t.index) for t in turns]
        out["backend"] = self.name
        return out

    # -- evaluation ------------------------------------------------------

    def evaluate(self, calls: Sequence[Call]) -> Dict[str, Any]:
        from sklearn.metrics import roc_auc_score

        texts, labels = self._training_turns(calls)
        turn_scores = np.asarray(
            self.pipeline_.predict_proba(texts)[:, 1]
        ) if texts else np.zeros(0)
        y = np.asarray(labels)
        turn_acc = float(((turn_scores >= 0.5).astype(int) == y).mean()) if y.size else 0.0

        call_scores = np.asarray([self.score_call(c)["intent_score"] for c in calls])
        call_y = np.asarray([int(c.label_scam) for c in calls])
        call_acc = float(((call_scores >= 0.5).astype(int) == call_y).mean()) if call_y.size else 0.0

        def _auc(s, t):
            return float(roc_auc_score(t, s)) if len(set(t.tolist())) > 1 else float("nan")

        return {
            "model": self.name,
            "turn_accuracy": turn_acc,
            "turn_auc": _auc(turn_scores, y),
            "call_accuracy": call_acc,
            "call_auc": _auc(call_scores, call_y),
            "n_turns": int(y.size),
            "n_calls": int(call_y.size),
        }

    def top_terms(self, k: int = 12) -> Dict[str, List[Tuple[str, float]]]:
        """Highest and lowest weighted word features, for the report."""
        if self.pipeline_ is None:
            return {"scam": [], "benign": []}
        clf = self.pipeline_.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {"scam": [], "benign": []}
        union = self.pipeline_.named_steps["feats"]
        names = union.get_feature_names_out()
        coef = clf.coef_[0]
        order = np.argsort(coef)
        word_only = [i for i in order if names[i].startswith("word__")]
        return {
            "benign": [(names[i].split("__", 1)[1], float(coef[i])) for i in word_only[:k]],
            "scam": [(names[i].split("__", 1)[1], float(coef[i])) for i in word_only[::-1][:k]],
        }

    # -- persistence -----------------------------------------------------

    def save(self, path) -> None:
        import joblib

        joblib.dump(
            {"pipeline": self.pipeline_, "pooler": self.pooler_, "history": self.history_,
             "cfg": {"model": self.model_kind, "word_ngrams": self.word_ngrams,
                     "char_ngrams": self.char_ngrams, "C": self.C, "min_df": self.min_df,
                     "seed": self.seed, "caller_only": self.caller_only}},
            path,
        )

    @classmethod
    def load(cls, path) -> "TfidfIntent":
        import joblib

        from ..compat import install_pickle_aliases
        install_pickle_aliases()
        d = joblib.load(path)
        m = cls(**d["cfg"])
        m.pipeline_ = d["pipeline"]
        m.pooler_ = d["pooler"]
        m.history_ = d.get("history", {})
        return m


if __name__ == "__main__":
    import json

    from .ner_crf import demo_calls

    calls = demo_calls(n_repeat=5)
    train = [c for c in calls if c.split == "train"]
    test = [c for c in calls if c.split == "test"]
    model = TfidfIntent().fit(train)
    print("history:", json.dumps({k: v for k, v in model.history_.items() if k != "pool_features"}))

    res = model.evaluate(test)
    print(f"held-out  turn acc={res['turn_accuracy']:.3f} auc={res['turn_auc']:.3f}   "
          f"call acc={res['call_accuracy']:.3f} auc={res['call_auc']:.3f}   "
          f"({res['n_turns']} caller turns, {res['n_calls']} calls)")

    top = model.top_terms(8)
    print("scam-leaning words :", [t for t, _ in top["scam"]])
    print("benign-leaning     :", [t for t, _ in top["benign"]])

    probe = [
        "otp turant bataiye warna account band ho jayega",
        "hum kabhi otp nahi mangte aap chahe to branch aa sakte hain",
    ]
    for p in probe:
        print(f"  {model.score_text(p):.3f}  {p}")

    c = test[0]
    sc = model.score_call(c)
    print(f"\ncall {c.call_id} ({'scam' if c.label_scam else 'benign'}): "
          f"intent={sc['intent_score']:.3f} max_turn={sc['intent_max_turn']:.3f} "
          f"urgency/100tok={sc['urgency_density']:.2f} "
          f"ent_otp={sc['ent_otp']:.2f} ent_personal={sc['ent_personal']:.2f}")
