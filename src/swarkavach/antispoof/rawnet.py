"""A small RawNet-style end-to-end spoof detector.

The point of a raw waveform model here is that it does not commit to a front
end. LFCC throws away phase and smooths the spectrum into 20 numbers per frame;
whatever artefact a new vocoder leaves outside that view is invisible to it. A
network that learns its own filters can, in principle, find it.

The front end is a SincConv layer (Ravanelli and Bengio 2018, as used by RawNet2
for ASVspoof): each filter is a band-pass whose only free parameters are its low
and high cutoff, so 20 filters cost 40 parameters instead of the 2580 a free
convolution of the same kernel size would need. That is the whole idea. On a
few hundred training calls a free first layer overfits immediately, while a
parameterised filterbank can only move its pass-bands, and where those bands end
up after training is something you can plot and argue about.

After that: four residual blocks with batch norm and max pooling, a GRU over the
time axis, and a two-way head. About 90 thousand parameters, which keeps CPU
inference on a 30 second call under a second and makes a Colab training run on
ASVspoof a matter of minutes per epoch rather than hours.

Everything torch-related in this module is guarded. If torch is missing, or no
checkpoint has been trained, importing this module and calling `load_rawnet`
returns None and the pipeline falls through to the next back end.

Run the self-test with:
    python -m swarkavach.antispoof.rawnet
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..audioio import pad_or_trim, read_audio, resample, rms_normalize
from ..config import SETTINGS, TARGET_SR, artifact_path

__all__ = [
    "TORCH_AVAILABLE",
    "SincConv",
    "RawNetLite",
    "train_rawnet",
    "load_rawnet",
    "rawnet_path",
]

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

#: Segment length the network sees, in seconds. Four seconds at 8 kHz is 32000
#: samples, long enough to hold several syllables and short enough that a call
#: splits into a handful of segments.
SEGMENT_S = 4.0


def rawnet_path(name: str = "antispoof_rawnet.pt") -> Path:
    """Where the trained checkpoint lives."""
    return artifact_path(name)


def _segments(x: np.ndarray, n_samples: int, hop: Optional[int] = None) -> np.ndarray:
    """Cut a signal into fixed length segments, shape (n_segments, n_samples).

    A signal shorter than one segment is tiled rather than zero padded, because
    a long tail of zeros would tell the network more about the padding than
    about the voice.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)
    if x.size <= n_samples:
        return pad_or_trim(x, n_samples)[None, :]
    hop = int(hop or n_samples)
    starts = list(range(0, x.size - n_samples + 1, hop))
    if starts[-1] + n_samples < x.size:
        starts.append(x.size - n_samples)
    return np.stack([x[s: s + n_samples] for s in starts])


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


if TORCH_AVAILABLE:

    class SincConv(nn.Module):
        """Band-pass filterbank whose cutoffs are the only learnable parameters.

        Each filter is the difference of two ideal low-pass sincs, Hamming
        windowed. Two numbers per filter: the low cutoff and the bandwidth, both
        kept positive by taking absolute values, with a floor so a filter cannot
        collapse to zero width and produce a divide by zero.

        Initialised mel-spaced across the band, which just means the filters
        start where speech energy is and then move during training.
        """

        def __init__(
            self,
            out_channels: int = 20,
            kernel_size: int = 129,
            sr: int = TARGET_SR,
            stride: int = 1,
            min_low_hz: float = 30.0,
            min_band_hz: float = 30.0,
        ) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                kernel_size += 1  # symmetric filters need an odd length
            self.out_channels = int(out_channels)
            self.kernel_size = int(kernel_size)
            self.sr = int(sr)
            self.stride = int(stride)
            self.min_low_hz = float(min_low_hz)
            self.min_band_hz = float(min_band_hz)

            high = self.sr / 2.0 - (self.min_low_hz + self.min_band_hz)
            mel = np.linspace(_hz_to_mel(self.min_low_hz), _hz_to_mel(high),
                              self.out_channels + 1)
            hz = _mel_to_hz(mel)
            self.low_hz_ = nn.Parameter(
                torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
            self.band_hz_ = nn.Parameter(
                torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

            half = self.kernel_size // 2
            n_lin = torch.linspace(0, half - 1, steps=half)
            window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
            self.register_buffer("window_", window.view(1, -1), persistent=False)
            n = (self.kernel_size - 1) / 2.0
            self.register_buffer(
                "n_",
                (2 * math.pi * torch.arange(-n, 0, dtype=torch.float32) / self.sr).view(1, -1),
                persistent=False,
            )

        def filters(self) -> "torch.Tensor":
            low = self.min_low_hz + torch.abs(self.low_hz_)
            high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                               self.min_low_hz, self.sr / 2.0)
            band = (high - low)[:, 0]

            left = ((torch.sin(high @ self.n_) - torch.sin(low @ self.n_))
                    / (self.n_ / 2.0)) * self.window_
            centre = 2.0 * band.view(-1, 1)
            right = torch.flip(left, dims=[1])
            kernel = torch.cat([left, centre, right], dim=1)
            kernel = kernel / (2.0 * band[:, None])
            return kernel.view(self.out_channels, 1, self.kernel_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return F.conv1d(x, self.filters(), stride=self.stride,
                            padding=self.kernel_size // 2)

        def band_edges(self) -> np.ndarray:
            """(low, high) in Hz per filter, for the plot in the report."""
            with torch.no_grad():
                low = (self.min_low_hz + torch.abs(self.low_hz_)).cpu().numpy().ravel()
                band = (self.min_band_hz + torch.abs(self.band_hz_)).cpu().numpy().ravel()
            return np.stack([low, np.minimum(low + band, self.sr / 2.0)], axis=1)

    class ResBlock(nn.Module):
        """Pre-activation residual block, then max pool.

        Pre-activation (norm and nonlinearity before the convolution) keeps the
        shortcut a clean identity path, which is what lets a stack this deep
        train on a few hundred calls without careful initialisation.
        """

        def __init__(self, c_in: int, c_out: int, pool: int = 3) -> None:
            super().__init__()
            self.bn1 = nn.BatchNorm1d(c_in)
            self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(c_out)
            self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
            self.shortcut = (nn.Conv1d(c_in, c_out, kernel_size=1)
                             if c_in != c_out else nn.Identity())
            self.pool = nn.MaxPool1d(pool)
            self.act = nn.LeakyReLU(0.3)

        def forward(self, x):
            h = self.conv1(self.act(self.bn1(x)))
            h = self.conv2(self.act(self.bn2(h)))
            return self.pool(h + self.shortcut(x))

    class RawNetLite(nn.Module):
        """SincConv front end, residual trunk, GRU, two-way head."""

        def __init__(
            self,
            sr: int = TARGET_SR,
            n_filters: int = 20,
            kernel_size: int = 129,
            channels: Sequence[int] = (32, 32, 64, 64),
            gru_hidden: int = 64,
            segment_s: float = SEGMENT_S,
            seed: Optional[int] = None,
        ) -> None:
            super().__init__()
            seed = int(SETTINGS.pipeline.seed if seed is None else seed)
            torch.manual_seed(seed)

            self.sr = int(sr)
            self.segment_s = float(segment_s)
            self.n_samples = int(round(self.sr * self.segment_s))
            self.seed = seed

            self.sinc = SincConv(n_filters, kernel_size, sr)
            self.pool0 = nn.MaxPool1d(3)
            self.bn0 = nn.BatchNorm1d(n_filters)
            self.act0 = nn.LeakyReLU(0.3)

            blocks = []
            c_prev = n_filters
            for c in channels:
                blocks.append(ResBlock(c_prev, c))
                c_prev = c
            self.blocks = nn.Sequential(*blocks)

            self.bn_gru = nn.BatchNorm1d(c_prev)
            self.gru = nn.GRU(c_prev, gru_hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden),
                nn.LeakyReLU(0.3),
                nn.Linear(gru_hidden, 2),
            )

        # -- forward ------------------------------------------------------

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """(batch, n_samples) or (batch, 1, n_samples) in, logits (batch, 2) out."""
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = self.sinc(x)
            # Absolute value after the band-pass: what matters is how much
            # energy each band carries, not its sign. This is the standard
            # SincNet arrangement and it halves what the next layer has to learn.
            h = self.act0(self.bn0(self.pool0(torch.abs(h))))
            h = self.blocks(h)
            h = self.bn_gru(h)
            out, _ = self.gru(h.transpose(1, 2))
            return self.head(out[:, -1, :])

        def n_parameters(self) -> int:
            return int(sum(p.numel() for p in self.parameters()))

        # -- inference ----------------------------------------------------

        @torch.no_grad()
        def score(self, x, sr: int = TARGET_SR, hop: Optional[int] = None) -> float:
            """P(spoof) for a whole call.

            The call is cut into segments, each scored, and the probabilities
            averaged. Averaging probabilities rather than logits keeps one
            strange segment (a beep, a moment of silence) from dominating the
            verdict for the whole call.
            """
            probs = self.score_segments(x, sr, hop)
            return float(np.mean(probs)) if probs.size else 0.5

        @torch.no_grad()
        def score_segments(self, x, sr: int = TARGET_SR,
                           hop: Optional[int] = None) -> np.ndarray:
            """P(spoof) per segment, which is what the dashboard timeline shows."""
            self.eval()
            sig = np.asarray(x, dtype=np.float32).ravel()
            if sig.size == 0:
                return np.array([0.5], dtype=np.float64)
            if int(sr) != self.sr:
                sig = resample(sig, int(sr), self.sr)
            sig = rms_normalize(sig)
            segs = _segments(sig, self.n_samples, hop)
            batch = torch.from_numpy(np.ascontiguousarray(segs, dtype=np.float32))
            logits = self(batch)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            return np.nan_to_num(p.astype(np.float64), nan=0.5)

        # -- persistence --------------------------------------------------

        def config(self) -> Dict[str, Any]:
            return {
                "sr": self.sr,
                "n_filters": self.sinc.out_channels,
                "kernel_size": self.sinc.kernel_size,
                "channels": [b.conv2.out_channels for b in self.blocks],
                "gru_hidden": self.gru.hidden_size,
                "segment_s": self.segment_s,
                "seed": self.seed,
            }

        def save(self, path: Union[str, Path] = None, meta: Optional[Dict] = None) -> str:
            path = Path(path or rawnet_path())
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"kind": "RawNetLite", "config": self.config(),
                 "state_dict": self.state_dict(), "meta": meta or {}},
                path,
            )
            return str(path)

else:  # pragma: no cover - only when torch is missing

    class SincConv:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise RuntimeError("SincConv needs torch, which is not installed")

    class RawNetLite:  # type: ignore[no-redef]
        """Placeholder so imports and isinstance checks keep working."""

        def __init__(self, *a, **kw):
            raise RuntimeError("RawNetLite needs torch, which is not installed")


# --------------------------------------------------------------------------
# Training and loading
# --------------------------------------------------------------------------


def _load_signal(item, sr: int) -> np.ndarray:
    """A path or an array in, a float32 waveform at `sr` out."""
    if isinstance(item, (str, Path)):
        sig, _ = read_audio(str(item), sr)
        return np.asarray(sig, dtype=np.float32)
    arr = np.asarray(item, dtype=np.float32).ravel()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def train_rawnet(
    X_paths_or_arrays: Sequence,
    labels: Sequence[int],
    sr: int = TARGET_SR,
    epochs: int = 12,
    batch_size: int = 8,
    lr: float = 1e-3,
    segment_s: float = SEGMENT_S,
    val_split: float = 0.2,
    seed: Optional[int] = None,
    out_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    model: Optional["RawNetLite"] = None,
) -> Tuple[Optional["RawNetLite"], Dict[str, Any]]:
    """Train the network on CPU. Returns (model, history).

    Accepts file paths or in-memory arrays, mixed freely. Each epoch takes a
    fresh random crop of every training call, which is the cheapest useful
    augmentation for this task: it stops the network from memorising where in
    the file each utterance starts, and it multiplies a small corpus by the
    number of distinct crops.

    Returns (None, {"error": ...}) rather than raising when torch is missing or
    the data is unusable, because callers treat this back end as optional.
    """
    if not TORCH_AVAILABLE:
        return None, {"error": "torch is not installed"}

    y = np.asarray(labels).ravel().astype(int)
    if y.size != len(X_paths_or_arrays):
        return None, {"error": "labels and inputs have different lengths"}
    if np.unique(y).size < 2:
        return None, {"error": "training needs both classes"}

    seed = int(SETTINGS.pipeline.seed if seed is None else seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    signals = [rms_normalize(_load_signal(item, sr)) for item in X_paths_or_arrays]
    n = len(signals)
    n_samples = int(round(sr * segment_s))

    # Held-out split for the epoch-by-epoch validation curve. Stratified, so a
    # small corpus cannot end up with a validation set of one class.
    idx = np.arange(n)
    val_idx: List[int] = []
    for cls in (0, 1):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        take = int(round(val_split * cls_idx.size))
        val_idx.extend(cls_idx[:take].tolist())
    val_set = set(val_idx)
    train_idx = np.array([i for i in idx if i not in val_set], dtype=int)
    val_idx_arr = np.array(sorted(val_set), dtype=int)
    if train_idx.size == 0:
        train_idx = idx

    net = model if model is not None else RawNetLite(sr=sr, segment_s=segment_s, seed=seed)
    net.train()

    counts = np.bincount(y[train_idx], minlength=2).astype(np.float64)
    weight = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) / 2.0,
                          dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    def crop(i: int, deterministic: bool = False) -> np.ndarray:
        sig = signals[i]
        if sig.size <= n_samples:
            return pad_or_trim(sig, n_samples)
        start = 0 if deterministic else int(rng.integers(0, sig.size - n_samples))
        return sig[start: start + n_samples]

    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_acc": [],
                               "n_train": int(train_idx.size),
                               "n_val": int(val_idx_arr.size),
                               "n_parameters": net.n_parameters(), "seed": seed}

    for epoch in range(int(epochs)):
        net.train()
        order = train_idx.copy()
        rng.shuffle(order)
        losses = []
        for start in range(0, order.size, batch_size):
            chunk = order[start: start + batch_size]
            if chunk.size < 2:  # batch norm needs more than one example
                continue
            xb = torch.from_numpy(np.stack([crop(i) for i in chunk]))
            yb = torch.from_numpy(y[chunk]).long()
            opt.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))

        if val_idx_arr.size >= 2:
            net.eval()
            with torch.no_grad():
                xb = torch.from_numpy(np.stack([crop(i, True) for i in val_idx_arr]))
                yb = torch.from_numpy(y[val_idx_arr]).long()
                logits = net(xb)
                history["val_loss"].append(float(criterion(logits, yb).item()))
                pred = logits.argmax(dim=1)
                history["val_acc"].append(float((pred == yb).float().mean().item()))
        if verbose:
            tail = (f" val_loss={history['val_loss'][-1]:.4f}"
                    f" val_acc={history['val_acc'][-1]:.3f}"
                    if history["val_loss"] else "")
            print(f"    epoch {epoch + 1:2d} train_loss={history['train_loss'][-1]:.4f}{tail}")

    net.eval()
    if out_path is not None:
        history["path"] = net.save(out_path, meta=history)
    return net, history


def load_rawnet(path: Optional[Union[str, Path]] = None) -> Optional["RawNetLite"]:
    """Load a checkpoint, or return None if there is not one to load.

    Never raises. A missing file, a missing torch, a checkpoint written by an
    older layout: all of them mean "this back end is unavailable", and the
    scorer moves on to the next one.
    """
    if not TORCH_AVAILABLE:
        return None
    p = Path(path or rawnet_path())
    if not p.exists():
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        cfg = dict(blob.get("config", {}))
        cfg.pop("kind", None)
        net = RawNetLite(**cfg)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return net
    except Exception:
        return None


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    from .features import demo_bonafide, demo_spoof

    t0 = time.time()
    print("antispoof.rawnet self-test")
    if not TORCH_AVAILABLE:
        print("  torch is not installed, the module still imports and "
              "load_rawnet() returns None")
        raise SystemExit(0)

    sr = TARGET_SR
    net = RawNetLite(sr=sr)
    print(f"  parameters: {net.n_parameters():,} (budget 300,000)")
    edges = net.sinc.band_edges()
    print(f"  sinc filters at init: {edges.shape[0]} bands, "
          f"first {edges[0, 0]:.0f}-{edges[0, 1]:.0f} Hz, "
          f"last {edges[-1, 0]:.0f}-{edges[-1, 1]:.0f} Hz")

    call30 = np.tile(demo_bonafide(sr, 3.0, seed=1), 10)
    t1 = time.time()
    p = net.score(call30, sr)
    dt = time.time() - t1
    print(f"  untrained score on a 30 s call: {p:.3f} in {dt:.2f} s "
          f"({call30.size / sr:.0f} s of audio, budget 1.0 s)")

    n = 10
    train = ([demo_bonafide(sr, 2.5, seed=i, f0_base=105 + 7 * (i % 5)) for i in range(n)]
             + [demo_spoof(sr, 2.5, seed=400 + i, f0_base=105 + 7 * (i % 5)) for i in range(n)])
    y = [0] * n + [1] * n
    t1 = time.time()
    net, hist = train_rawnet(train, y, sr=sr, epochs=8, batch_size=4,
                             segment_s=2.0, verbose=True)
    print(f"  trained {hist['n_train']} calls in {time.time() - t1:.1f} s, "
          f"final val_acc={hist['val_acc'][-1] if hist['val_acc'] else float('nan'):.3f}")

    test_b = [demo_bonafide(sr, 2.5, seed=900 + i) for i in range(5)]
    test_s = [demo_spoof(sr, 2.5, seed=950 + i) for i in range(5)]
    pb = [net.score(x, sr) for x in test_b]
    ps = [net.score(x, sr) for x in test_s]
    print(f"  held-out P(spoof): bona fide {np.mean(pb):.3f}, spoofed {np.mean(ps):.3f}")

    path = rawnet_path("selftest_rawnet.pt")
    net.save(path)
    again = load_rawnet(path)
    print(f"  save/load round trip: |delta| "
          f"{abs(again.score(test_b[0], sr) - pb[0]):.2e}")
    Path(path).unlink(missing_ok=True)
    print(f"  load_rawnet on a missing file: {load_rawnet(rawnet_path('nope.pt'))}")
    print(f"  done in {time.time() - t0:.1f} s")
