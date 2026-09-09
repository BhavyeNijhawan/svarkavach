"""A small TDNN speaker embedder, used here to measure within-call stability.

The usual reason to embed a speaker is to compare two recordings. That is not
what this is for. The fusion vector wants `spk_consistency`: how much the
speaker embedding moves BETWEEN segments of one call. The reasoning:

- A real person recorded over a few seconds drifts. Their pitch settles, they
  lean toward or away from the handset, they get louder on a stressed word.
  Segment to segment cosine similarity is high but not pinned to 1.
- A single text-to-speech voice reading a script does not drift. Every segment
  comes from the same fixed speaker vector through the same vocoder, so the
  embeddings are unnaturally stable, closer to 1 than any human recording.
- A spliced call, or one where only part of the audio was cloned (the common
  real attack: a genuine greeting followed by a cloned instruction), swings the
  other way. Two different voices in one call means unnaturally low similarity.

So the feature is informative at BOTH ends and the relationship to fraud is not
monotonic. The scorer keeps the raw consistency in the fusion vector, because
that is what the contract asks for and the fusion model can learn the shape, and
uses a "distance from the normal human band" transform in its own heuristic.

This works without training. The network is a fixed, seeded random projection of
the cepstral trajectory, so it is a deterministic nonlinear function of the
audio: unrelated voices land in different places, the same voice twice lands in
the same place. `fit` trains the trunk with a speaker classification objective,
and the pipeline never calls it, for a reason worth writing down: a speaker
classifier is trained to be invariant to within-speaker variation, which is
exactly the drift this feature measures. On the demo signals below, training
reaches 95 percent speaker accuracy and cuts the human versus synthetic
separation from d = 3.3 to d = 0.7. Whether that also happens on a real corpus
is an experiment for the report to run, not something to assume either way.

One step is not optional, trained or not: centering. An untrained trunk adds a
large constant vector to every embedding, so raw cosines between any two calls
come out above 0.999 and the measure is dead on arrival. Subtracting a reference
mean before length normalisation is the standard x-vector post-processing step
and it is what makes the cosines mean anything. With no training data the
reference is estimated from seeded random feature matrices that share the first
and second order statistics of CMVN'd cepstra; `fit` replaces it with the mean
over the training embeddings.

Measured on the demo signals, six six-second calls of each kind, four segments
per call: untrained, human calls score -0.090 +- 0.064 and synthetic ones
+0.081 +- 0.036, and a signal against a copy of itself scores +0.998 against
-0.652 for two different voices. So the ordering is there before any training.
The absolute level is not meaningful with an untrained trunk, which is why the
fusion layer gets the raw number and decides for itself.

Run the self-test with:
    python -m swarkavach.antispoof.embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import SETTINGS, TARGET_SR, artifact_path
from .features import frame_features

__all__ = [
    "TORCH_AVAILABLE",
    "SpeakerEmbedder",
    "cosine_consistency",
    "EMBED_DIM",
]

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    TORCH_AVAILABLE = False

#: x-vector systems settled on 192 or 512. 192 is the smaller of the two and
#: plenty for a within-call comparison.
EMBED_DIM = 192

#: Segments per call for the consistency measure, and the shortest segment worth
#: embedding. Below about 0.6 s the statistics pooling has too few frames and the
#: embedding wobbles for reasons that have nothing to do with the speaker.
DEFAULT_SEGMENTS = 4
MIN_SEGMENT_S = 0.6


def cosine_consistency(embs) -> float:
    """Mean pairwise cosine similarity of a set of embeddings, in [-1, 1].

    One embedding (or none) means there is nothing to disagree with, so the
    answer is 1.0: a call too short to split cannot be called inconsistent.

    Not squashed into [0, 1], because with an untrained trunk the useful part of
    the range sits around zero and clipping at zero would throw away exactly the
    difference this feature exists to report.
    """
    arr = np.atleast_2d(np.asarray(embs, dtype=np.float64))
    if arr.shape[0] < 2:
        return 1.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-9)
    sim = arr @ arr.T
    iu = np.triu_indices(arr.shape[0], k=1)
    vals = sim[iu]
    if vals.size == 0:
        return 1.0
    return float(np.clip(np.nan_to_num(np.mean(vals), nan=0.0), -1.0, 1.0))


if TORCH_AVAILABLE:

    class _TDNNLayer(nn.Module):
        """One time-delay layer: a dilated 1-D convolution, ReLU, batch norm."""

        def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel,
                                  dilation=dilation, padding=0)
            self.act = nn.ReLU()
            self.bn = nn.BatchNorm1d(c_out)

        def forward(self, x):
            return self.bn(self.act(self.conv(x)))

    class _TDNN(nn.Module):
        """Five time-delay layers, statistics pooling, one linear bottleneck.

        Statistics pooling (concatenate the mean and the standard deviation over
        time) is what turns a variable length call into a fixed vector, and the
        standard deviation half is why the embedding notices a voice that is
        unnaturally steady.
        """

        def __init__(self, n_in: int, dim: int = EMBED_DIM, width: int = 96) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                _TDNNLayer(n_in, width, 5, 1),
                _TDNNLayer(width, width, 3, 2),
                _TDNNLayer(width, width, 3, 3),
                _TDNNLayer(width, width, 1, 1),
                _TDNNLayer(width, 2 * width, 1, 1),
            )
            self.proj = nn.Linear(4 * width, dim)
            # Receptive field: 4 + 4 + 6 = 14 frames of context, so a segment
            # needs at least that many frames to produce anything at all.
            self.min_frames = 16

        def forward(self, x):
            """(batch, n_frames, n_in) in, (batch, dim) out."""
            h = self.layers(x.transpose(1, 2))
            mean = h.mean(dim=2)
            std = h.std(dim=2, unbiased=False)
            return self.proj(torch.cat([mean, std], dim=1))

else:  # pragma: no cover - only when torch is missing

    class _TDNN:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise RuntimeError("the TDNN embedder needs torch")


class SpeakerEmbedder:
    """192-dimensional speaker embedding, and the within-call stability measure.

    Without torch this degrades to a deterministic random projection of the same
    pooled statistics, which is enough to keep `cosine_consistency` meaningful
    and the pipeline running.
    """

    def __init__(
        self,
        feature_set: str = "mfcc",
        dim: int = EMBED_DIM,
        width: int = 96,
        seed: Optional[int] = None,
    ) -> None:
        self.feature_set = str(feature_set).lower()
        self.dim = int(dim)
        self.width = int(width)
        self.seed = int(SETTINGS.pipeline.seed if seed is None else seed)
        self.n_in: Optional[int] = None
        self.net = None
        self.trained = False
        self.meta_: Dict[str, Any] = {}
        self.center: Optional[np.ndarray] = None
        self._fallback: Optional[np.ndarray] = None

    # -- construction -----------------------------------------------------

    def _build(self, n_in: int) -> None:
        if (self.net is not None or self._fallback is not None) and self.n_in == n_in:
            return
        self.n_in = int(n_in)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            self.net = _TDNN(self.n_in, self.dim, self.width)
            self.net.eval()
        else:  # deterministic random projection of the pooled statistics
            rng = np.random.default_rng(self.seed)
            self._fallback = rng.standard_normal((2 * self.n_in, self.dim)) / np.sqrt(self.n_in)
        if self.center is None or self.center.size != self.dim:
            self.center = self._probe_center()

    def _probe_center(self) -> np.ndarray:
        """Reference mean, estimated without any data.

        Eight seeded standard normal frame matrices of different lengths. They
        match what the front end actually produces (CMVN leaves every column at
        zero mean and unit variance) closely enough to estimate the constant the
        network adds to everything, which is all this has to do.
        """
        rng = np.random.default_rng(self.seed)
        probes = [rng.standard_normal((n, self.n_in))
                  for n in (64, 96, 128, 200, 300, 64, 128, 256)]
        return np.mean(np.stack([self._raw(p) for p in probes]), axis=0)

    def n_parameters(self) -> int:
        if self.net is None:
            return 0 if self._fallback is None else int(self._fallback.size)
        return int(sum(p.numel() for p in self.net.parameters()))

    # -- embedding --------------------------------------------------------

    def _raw(self, feat: np.ndarray) -> np.ndarray:
        """Uncentred, unnormalised network output for one frame matrix."""
        feat = np.atleast_2d(np.asarray(feat, dtype=np.float32))
        if self.net is None:  # numpy fallback
            pooled = np.concatenate([feat.mean(axis=0), feat.std(axis=0)])
            vec = np.tanh(pooled @ self._fallback)
        else:
            need = getattr(self.net, "min_frames", 16)
            if feat.shape[0] < need:  # pad by repeating, never by zeros
                reps = int(np.ceil(need / max(feat.shape[0], 1)))
                feat = np.tile(feat, (reps, 1))[:need]
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32))[None]
                vec = self.net(t)[0].cpu().numpy()
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def _embed_matrix(self, feat: np.ndarray) -> np.ndarray:
        """Frame matrix (n_frames, d) to a centred, unit norm embedding."""
        feat = np.atleast_2d(np.asarray(feat, dtype=np.float32))
        if feat.shape[0] == 0:
            feat = np.zeros((1, max(self.n_in or 1, 1)), dtype=np.float32)
        self._build(feat.shape[1])
        vec = self._raw(feat)
        if self.center is not None and self.center.shape == vec.shape:
            vec = vec - self.center
        return vec / max(float(np.linalg.norm(vec)), 1e-9)

    def embed(self, x, sr: int = TARGET_SR) -> np.ndarray:
        """One 192-dimensional unit norm vector for a whole signal."""
        feat = frame_features(x, sr, self.feature_set)
        return self._embed_matrix(feat)

    def embed_segments(
        self,
        x,
        sr: int = TARGET_SR,
        n_segments: int = DEFAULT_SEGMENTS,
        min_segment_s: float = MIN_SEGMENT_S,
    ) -> np.ndarray:
        """Embed equal slices of one call, shape (n, 192).

        The front end runs once over the whole call and the slicing happens on
        the frame matrix, which matters: CMVN is then computed over the entire
        call, so the segments share one channel estimate and any difference
        between them is the voice, not the normalisation.
        """
        feat = frame_features(x, sr, self.feature_set)
        n_frames = feat.shape[0]
        hop_s = SETTINGS.frame.hop_ms / 1000.0
        min_frames = max(int(min_segment_s / hop_s), 16)

        k = int(max(1, min(n_segments, n_frames // min_frames)))
        if k < 2:
            return self._embed_matrix(feat)[None, :]
        bounds = np.linspace(0, n_frames, k + 1).astype(int)
        return np.stack([
            self._embed_matrix(feat[a:b]) for a, b in zip(bounds[:-1], bounds[1:])
        ])

    def consistency(self, x, sr: int = TARGET_SR,
                    n_segments: int = DEFAULT_SEGMENTS) -> float:
        """`cosine_consistency` of this call's segment embeddings."""
        return cosine_consistency(self.embed_segments(x, sr, n_segments))

    # -- optional training ------------------------------------------------

    def fit(
        self,
        items: Sequence,
        speakers: Optional[Sequence] = None,
        sr: int = TARGET_SR,
        epochs: int = 8,
        batch_size: int = 8,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the trunk with a plain speaker classification objective.

        `items` is a list of Call objects (audio read from `audio_path`), of
        waveforms, or of (waveform, speaker_id) pairs. `speakers` supplies the
        labels when the items do not carry them.

        This is optional on purpose, and off by default. A head over two dozen
        simulated speakers pulls clips of one speaker together and pushes
        different speakers apart, which is the right objective for verification
        and not obviously the right one for `spk_consistency`: the same training
        signal tells the network to ignore the within-speaker drift that the
        consistency feature reads. Measure it on your corpus before turning it
        on. Returns a history dict, or {"error": ...} instead of raising.
        """
        if not TORCH_AVAILABLE:
            return {"error": "torch is not installed"}

        signals, labels = _unpack(items, speakers, sr)
        if not signals:
            return {"error": "no usable audio"}
        uniq = sorted(set(labels))
        if len(uniq) < 2:
            return {"error": "speaker training needs at least two speakers"}
        lut = {s: i for i, s in enumerate(uniq)}
        y = np.array([lut[s] for s in labels], dtype=np.int64)

        feats = [frame_features(x, sr, self.feature_set).astype(np.float32)
                 for x in signals]
        self._build(feats[0].shape[1])
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        head = nn.Linear(self.dim, len(uniq))
        params = list(self.net.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        # Fixed length crops so a batch can be stacked into one tensor.
        crop = max(int(getattr(self.net, "min_frames", 16)) * 4, 64)

        history: Dict[str, Any] = {"loss": [], "acc": [], "n_speakers": len(uniq),
                                   "n_items": len(feats), "seed": self.seed}
        self.net.train()
        for epoch in range(int(epochs)):
            order = rng.permutation(len(feats))
            losses, correct, total = [], 0, 0
            for start in range(0, order.size, batch_size):
                chunk = order[start: start + batch_size]
                if chunk.size < 2:
                    continue
                batch = []
                for i in chunk:
                    f = feats[i]
                    if f.shape[0] <= crop:
                        reps = int(np.ceil(crop / max(f.shape[0], 1)))
                        f = np.tile(f, (reps, 1))[:crop]
                    else:
                        a = int(rng.integers(0, f.shape[0] - crop))
                        f = f[a: a + crop]
                    batch.append(f)
                xb = torch.from_numpy(np.stack(batch))
                yb = torch.from_numpy(y[chunk])
                opt.zero_grad()
                logits = head(self.net(xb))
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                correct += int((logits.argmax(dim=1) == yb).sum().item())
                total += int(yb.numel())
            history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
            history["acc"].append(correct / max(total, 1))
            if verbose:
                print(f"    epoch {epoch + 1:2d} loss={history['loss'][-1]:.4f} "
                      f"acc={history['acc'][-1]:.3f}")
        self.net.eval()
        # Now that there is real data, the reference mean stops being a guess.
        self.center = np.mean(np.stack([self._raw(f) for f in feats]), axis=0)
        self.trained = True
        self.meta_ = dict(history)
        return history

    # -- persistence ------------------------------------------------------

    def save(self, path: Union[str, Path] = None) -> str:
        path = Path(path or artifact_path("antispoof_embedder.pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "kind": "SpeakerEmbedder",
            "feature_set": self.feature_set,
            "dim": self.dim,
            "width": self.width,
            "seed": self.seed,
            "n_in": self.n_in,
            "trained": self.trained,
            "center": (None if self.center is None else np.asarray(self.center)),
            "meta": self.meta_,
            "state_dict": (self.net.state_dict()
                           if (TORCH_AVAILABLE and self.net is not None) else None),
        }
        if TORCH_AVAILABLE:
            torch.save(blob, path)
        else:  # pragma: no cover - environment dependent
            import joblib

            joblib.dump(blob, path)
        return str(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Optional["SpeakerEmbedder"]:
        """Load an embedder, or None if the file is missing or unreadable."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            if TORCH_AVAILABLE:
                blob = torch.load(p, map_location="cpu", weights_only=False)
            else:  # pragma: no cover - environment dependent
                import joblib

                blob = joblib.load(p)
            obj = cls(feature_set=blob.get("feature_set", "mfcc"),
                      dim=int(blob.get("dim", EMBED_DIM)),
                      width=int(blob.get("width", 96)),
                      seed=blob.get("seed"))
            n_in = blob.get("n_in")
            if n_in:
                obj._build(int(n_in))
            sd = blob.get("state_dict")
            if sd is not None and obj.net is not None:
                obj.net.load_state_dict(sd)
                obj.net.eval()
            centre = blob.get("center")
            if centre is not None:
                obj.center = np.asarray(centre, dtype=np.float64)
            obj.trained = bool(blob.get("trained", False))
            obj.meta_ = dict(blob.get("meta", {}))
            return obj
        except Exception:
            return None


def _unpack(items: Sequence, speakers: Optional[Sequence], sr: int):
    """Turn Call objects, arrays or pairs into (signals, speaker labels)."""
    from ..audioio import read_audio

    signals: List[np.ndarray] = []
    labels: List[str] = []
    for i, item in enumerate(items):
        spk = None
        sig = None
        if isinstance(item, tuple) and len(item) == 2:
            sig, spk = item[0], item[1]
        elif hasattr(item, "audio_path"):  # a Call
            if not item.audio_path or not Path(str(item.audio_path)).exists():
                continue
            sig, _ = read_audio(str(item.audio_path), sr)
            spk = getattr(item, "speaker_id", None)
        else:
            sig = item
        if speakers is not None and i < len(speakers):
            spk = speakers[i]
        if sig is None or spk is None:
            continue
        arr = np.asarray(sig, dtype=np.float32).ravel()
        if arr.size < sr // 4:
            continue
        signals.append(arr)
        labels.append(str(spk))
    return signals, labels


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    from .features import demo_bonafide, demo_spoof

    t0 = time.time()
    sr = TARGET_SR
    print("antispoof.embeddings self-test")

    emb = SpeakerEmbedder()
    v = emb.embed(demo_bonafide(sr, 2.0, seed=1), sr)
    print(f"  torch available: {TORCH_AVAILABLE}, embedding {v.shape}, "
          f"norm {np.linalg.norm(v):.4f}, parameters {emb.n_parameters():,}")

    a = demo_bonafide(sr, 2.0, seed=1, f0_base=120.0)
    b = demo_bonafide(sr, 2.0, seed=77, f0_base=190.0)
    same = np.concatenate([a, a])
    diff = np.concatenate([a, b])
    c_same = cosine_consistency(emb.embed_segments(same, sr, 2))
    c_diff = cosine_consistency(emb.embed_segments(diff, sr, 2))
    print(f"  consistency, a signal against itself: {c_same:.4f}")
    print(f"  consistency, two different voices:    {c_diff:.4f}")

    human = [emb.consistency(demo_bonafide(sr, 6.0, seed=s, f0_base=110.0 + 13.0 * s), sr)
             for s in range(6)]
    machine = [emb.consistency(demo_spoof(sr, 6.0, seed=s, f0_base=110.0 + 13.0 * s), sr)
               for s in range(6)]
    print(f"  4 segment consistency over 6 calls each: "
          f"human {np.mean(human):+.3f} +- {np.std(human):.3f}, "
          f"synthetic {np.mean(machine):+.3f} +- {np.std(machine):.3f}")
    print(f"    human   {np.round(human, 3)}")
    print(f"    synthetic {np.round(machine, 3)}")

    print(f"  determinism: repeat embed max |delta| "
          f"{np.max(np.abs(emb.embed(a, sr) - emb.embed(a, sr))):.2e}")
    print(f"  cosine_consistency of a single embedding: {cosine_consistency(v[None, :]):.1f}")

    pairs = []
    for spk in range(4):
        for take in range(5):
            pairs.append((demo_bonafide(sr, 2.5, seed=100 * spk + take,
                                        f0_base=100.0 + 30.0 * spk), f"spk_{spk}"))
    hist = emb.fit(pairs, sr=sr, epochs=15, batch_size=4)
    print(f"  optional fit: {hist.get('n_speakers')} speakers, {hist.get('n_items')} "
          f"clips, accuracy {hist['acc'][0]:.3f} -> {hist['acc'][-1]:.3f} "
          f"(chance {1.0 / hist['n_speakers']:.2f})")
    human2 = [emb.consistency(demo_bonafide(sr, 6.0, seed=s, f0_base=110.0 + 13.0 * s), sr)
              for s in range(6)]
    machine2 = [emb.consistency(demo_spoof(sr, 6.0, seed=s, f0_base=110.0 + 13.0 * s), sr)
                for s in range(6)]

    def sep(a, b):
        return (np.mean(b) - np.mean(a)) / max(np.sqrt(0.5 * (np.var(a) + np.var(b))), 1e-9)

    print(f"  human vs synthetic separation (Cohen's d): "
          f"untrained {sep(human, machine):.2f} -> trained {sep(human2, machine2):.2f}")
    print("    training buys speaker accuracy and costs consistency contrast, "
          "which is why the pipeline leaves the trunk untrained")

    path = artifact_path("selftest_embedder.pt")
    emb.save(path)
    again = SpeakerEmbedder.load(path)
    print(f"  save/load round trip: max |delta| "
          f"{np.max(np.abs(again.embed(a, sr) - emb.embed(a, sr))):.2e}")
    Path(path).unlink(missing_ok=True)
    print(f"  load on a missing file: {SpeakerEmbedder.load(artifact_path('nope.pt'))}")
    print(f"  done in {time.time() - t0:.1f} s")
