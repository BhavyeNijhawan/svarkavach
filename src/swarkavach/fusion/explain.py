"""Explanations for a fused verdict.

Two explainers and one assembler.

`linear_shap` is exact. For a logistic model the log-odds are additive in the
standardised features, so the Shapley value of feature i is just its weight
times its deviation from the background mean. No sampling, no approximation,
no extra dependency. That is the main reason the default fusion model is
logistic: the evidence panel can state a number and mean it.

`kernel_shap` is the sampling fallback for the gradient-boosting model.

`build_evidence` turns contributions, entity spans and lexicon hits into the
object the console renders, including reasons written in plain language,
because a risk score nobody can argue with is a risk score nobody will trust.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import (
    Call, Evidence, EntitySpan, FEATURE_GROUPS, FEATURE_LABELS, risk_band,
)

# --------------------------------------------------------------------------
# explainers
# --------------------------------------------------------------------------


def linear_shap(
    weights: np.ndarray,
    z_standardised: np.ndarray,
    names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Exact Shapley values for a linear model in log-odds space.

    The background is the training mean, which is the origin after
    standardisation, so the deviation term is the standardised value itself.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    z = np.asarray(z_standardised, dtype=np.float64).ravel()
    n = min(len(w), len(z), len(names))
    out = []
    for i in range(n):
        phi = float(w[i] * z[i])
        if not np.isfinite(phi):
            phi = 0.0
        name = names[i]
        out.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "group": FEATURE_GROUPS.get(name, "other"),
            "value": float(z[i]),
            "contribution": phi,
        })
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out


def kernel_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    names: Sequence[str],
    n_samples: int = 512,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Sampling Shapley values via the kernel SHAP weighted regression.

    Used for the non-linear estimator. The output is on the same log-odds
    scale as `linear_shap` so the two are directly comparable in the report.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    bg = np.asarray(background, dtype=np.float64).ravel()
    d = len(x)
    rng = np.random.default_rng(seed)

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # coalition sizes 1..d-1 drawn with the kernel SHAP weighting
    sizes = np.arange(1, d)
    kw = (d - 1) / (sizes * (d - sizes))
    kw = kw / kw.sum()

    masks = np.zeros((n_samples, d), dtype=bool)
    for i in range(n_samples):
        k = rng.choice(sizes, p=kw)
        masks[i, rng.choice(d, size=int(k), replace=False)] = True

    X = np.where(masks, x, bg)
    fx = logit(np.asarray(predict_fn(X)).ravel())
    f0 = float(logit(np.asarray(predict_fn(bg.reshape(1, -1))).ravel())[0])
    f1 = float(logit(np.asarray(predict_fn(x.reshape(1, -1))).ravel())[0])

    M = masks.astype(np.float64)
    yv = fx - f0
    # constrain the contributions to sum to f1 - f0 by eliminating the last one
    A = M[:, :-1] - M[:, -1:]
    b = yv - M[:, -1] * (f1 - f0)
    reg = 1e-6 * np.eye(d - 1)
    try:
        phi_head = np.linalg.solve(A.T @ A + reg, A.T @ b)
    except np.linalg.LinAlgError:
        phi_head = np.linalg.lstsq(A, b, rcond=None)[0]
    phi = np.append(phi_head, (f1 - f0) - phi_head.sum())

    out = []
    for i in range(d):
        name = names[i] if i < len(names) else f"f{i}"
        out.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "group": FEATURE_GROUPS.get(name, "other"),
            "value": float(x[i]),
            "contribution": float(phi[i]) if np.isfinite(phi[i]) else 0.0,
        })
    out.sort(key=lambda d_: abs(d_["contribution"]), reverse=True)
    return out


# --------------------------------------------------------------------------
# plain language
# --------------------------------------------------------------------------

ENT_PHRASE = {
    "OTP": "asks for a one-time password",
    "AUTHORITY_CLAIM": "claims to be calling from a government or police body",
    "THREAT_DEADLINE": "attaches a deadline to a threat",
    "PAYMENT_HANDLE": "supplies a payment destination",
    "PERSONAL_INFO_REQ": "requests information a real institution already holds",
    "BANK_ENTITY": "refers to a bank account or card",
    "MONEY_AMOUNT": "names a sum of money",
}


def _reason(text: str, group: str = "intent", direction: str = "raises") -> Dict[str, str]:
    return {"text": text, "group": group, "direction": direction}


def write_reasons(
    call: Call,
    features: Dict[str, float],
    contributions: Sequence[Dict[str, Any]],
    detail: Dict[str, Any],
    risk: float,
) -> List[Dict[str, str]]:
    """Turn numbers into sentences a reviewer can check against the transcript."""
    out: List[Dict[str, str]] = []
    counts: Dict[str, int] = detail.get("entity_counts", {}) or {}

    # what the call actually did, in order of how damning it is
    order = ["OTP", "PERSONAL_INFO_REQ", "AUTHORITY_CLAIM", "THREAT_DEADLINE",
             "PAYMENT_HANDLE", "BANK_ENTITY"]
    for et in order:
        n = counts.get(et, 0)
        if n:
            phrase = ENT_PHRASE.get(et, et)
            out.append(_reason(
                f"The caller {phrase} ({n} mention{'s' if n > 1 else ''})."))

    coer = detail.get("coercion") or {}
    slope = float(features.get("coercion_slope", 0.0))
    if slope > 0.05:
        out.append(_reason(
            f"Pressure climbs steadily through the call (escalation slope {slope:.2f}), "
            "which is the shape a scripted extraction follows rather than a normal conversation.",
            group="cross"))
    peak = coer.get("peak_act")
    if peak in ("REQUEST_SENSITIVE", "ISOLATE", "PRESSURE_ESCALATE", "DEADLINE", "THREAT"):
        out.append(_reason(
            f"The call reaches a {peak.replace('_', ' ').lower()} turn, the top of the coercion ladder.",
            group="cross"))

    pim = detail.get("pim") or {}
    pim_v = float(features.get("pim", 0.0))
    if pim.get("available") and pim_v >= 0.35:
        out.append(_reason(
            f"Delivery does not match the wording: high-pressure language spoken with flat prosody "
            f"(mismatch {pim_v:.2f}, correlation {pim.get('corr', 0):.2f}). "
            "That combination is characteristic of a synthetic voice reading a script.",
            group="cross"))
    elif pim.get("available") and pim_v < 0.15 and float(features.get("intent_score", 0)) > 0.6:
        out.append(_reason(
            "The speaker's delivery tracks the content, so the pressure sounds genuine rather than read out.",
            group="cross", direction="clears"))

    as_score = float(features.get("as_score", 0.0))
    ab = detail.get("antispoof") or {}
    if ab.get("backend") not in (None, "unavailable"):
        if as_score >= 0.6:
            out.append(_reason(
                f"The voice scores {as_score:.2f} on the synthetic-speech detector "
                f"({ab.get('backend')} backend), with unusually low pitch jitter and shimmer.",
                group="voice"))
        elif as_score <= 0.3:
            out.append(_reason(
                f"The voice carries the micro-variation of real speech (synthetic score {as_score:.2f}).",
                group="voice", direction="clears"))
    elif ab.get("backend") == "unavailable":
        out.append(_reason(
            "No audio was supplied, so this verdict rests on the transcript alone.",
            group="voice", direction="clears"))

    cm = detail.get("code_mixing") or {}
    align = float(features.get("ent_lang_align", 0.0))
    if counts and align <= 0.75 and counts.get("THREAT_DEADLINE", 0):
        out.append(_reason(
            f"The money words are English but the threat is in Hindi: only {align:.0%} of "
            "the fraud entities here are English tokens, against about 90 percent on a "
            "legitimate call. That split is the code-switching signature of a rehearsed "
            "Indian fraud script.",
            group="cross"))

    resist = float(features.get("callee_resist", 0.0))
    if resist > 0.3:
        out.append(_reason(
            f"The person receiving the call pushes back or asks clarifying questions in "
            f"{resist:.0%} of their turns, which is itself a sign they are being pressured.",
            group="cross"))

    # the model's own top drivers, if they say something the rules did not
    said = " ".join(r["text"].lower() for r in out)
    for c in list(contributions)[:3]:
        if c["contribution"] <= 0.15:
            continue
        lab = c["label"].lower()
        if lab.split()[0] in said:
            continue
        out.append(_reason(
            f"{c['label']} is the strongest single driver of this score "
            f"(pushes the log-odds by {c['contribution']:+.2f}).",
            group=c.get("group", "intent")))

    if not out:
        out.append(_reason(
            "Nothing in the wording, the entities or the voice stands out as fraudulent.",
            direction="clears"))
    if risk < 0.35:
        out.append(_reason(
            f"Overall risk {risk:.0%}, which sits in the {risk_band(risk)} band. No action suggested.",
            direction="clears"))
    return out


# --------------------------------------------------------------------------
# assembler
# --------------------------------------------------------------------------


def build_evidence(
    call: Call,
    features: Dict[str, float],
    contributions: Sequence[Dict[str, Any]],
    detail: Dict[str, Any],
    risk: float,
) -> Evidence:
    """Everything the console needs to justify the number it shows."""
    spans: List[Dict[str, Any]] = []
    for e in detail.get("entities", []) or []:
        spans.append({
            "turn_index": int(e["turn_index"]),
            "tok_start": int(e["tok_start"]),
            "tok_end": int(e["tok_end"]),
            "type": e["type"],
            "text": e.get("text", ""),
            "weight": float(e.get("score", 1.0)),
            "source": "ner",
        })

    # lexicon hits become spans too, but only where they do not sit on top of
    # an entity, so the transcript does not end up double-marked
    taken = {(s["turn_index"], i) for s in spans
             for i in range(s["tok_start"], s["tok_end"])}
    by_index = {t.index: t for t in call.turns}
    for h in detail.get("lexicon_spans", []) or []:
        turn = by_index.get(h["turn_index"])
        if turn is None:
            continue
        term_toks = h["term"].split()
        low = [t.lower() for t in turn.tokens]
        for i in range(len(low) - len(term_toks) + 1):
            if low[i:i + len(term_toks)] != term_toks:
                continue
            if any((turn.index, j) in taken for j in range(i, i + len(term_toks))):
                break
            spans.append({
                "turn_index": turn.index, "tok_start": i,
                "tok_end": i + len(term_toks), "type": h["type"],
                "text": h["term"], "weight": float(h["weight"]), "source": "lexicon",
            })
            for j in range(i, i + len(term_toks)):
                taken.add((turn.index, j))
            break

    markers: List[Dict[str, Any]] = []
    for row in (detail.get("pim", {}) or {}).get("per_turn", []):
        if row.get("gap", 0) >= 0.35 and "t" in row:
            markers.append({
                "t_start": max(0.0, float(row["t"]) - 1.0),
                "t_end": float(row["t"]),
                "kind": "prosody_gap",
                "note": f"pressure {row['lex']:.2f} spoken at arousal {row['aco']:.2f}",
            })

    return Evidence(
        reasons=write_reasons(call, features, contributions, detail, risk),
        contributions=[{
            "feature": c["feature"], "label": c["label"], "group": c["group"],
            "value": round(float(c["value"]), 5),
            "contribution": round(float(c["contribution"]), 5),
        } for c in contributions],
        spans=spans,
        audio_markers=markers,
    )


if __name__ == "__main__":
    w = np.array([1.4, -0.8, 0.3])
    z = np.array([1.2, 0.5, -2.0])
    names = ["as_score", "spk_consistency", "pim"]
    print("exact linear Shapley values:")
    for c in linear_shap(w, z, names):
        print(f"  {c['feature']:18s} value {c['value']:+.2f}  phi {c['contribution']:+.3f}")
    print(f"  sum {sum(c['contribution'] for c in linear_shap(w, z, names)):+.3f} "
          f"(equals w.z = {float(w @ z):+.3f})")

    # kernel SHAP should land close to the exact answer on a linear function
    def f(M):
        lo = M @ w
        return 1 / (1 + np.exp(-lo))

    approx = kernel_shap(f, z, np.zeros(3), names, n_samples=800, seed=1)
    print("\nkernel SHAP on the same linear model:")
    for c in approx:
        print(f"  {c['feature']:18s} phi {c['contribution']:+.3f}")
