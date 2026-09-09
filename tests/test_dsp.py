"""Tests for the signal processing layer: swarkavach.dsp and swarkavach.channel.

Runs two ways, because pytest is not in the local requirements list:

    .venv\\Scripts\\python.exe -m pytest tests/test_dsp.py
    .venv\\Scripts\\python.exe tests/test_dsp.py

Everything is a plain `assert` and a plain function, so nothing here depends on
pytest being importable. The block at the bottom collects every test_* function
in this module, runs it, and exits non zero if any of them fail.

What is checked and why it is worth checking:

- framing and windows, because every later number inherits the frame grid;
- ZCR against a square wave whose crossings can be counted by hand;
- the STFT against numpy.fft on a single frame, so a wrong window or a wrong
  n_fft cannot hide;
- no all-zero filterbank row, which is the silent killer in a hand written
  filterbank: an empty row turns a cepstral channel into a constant and the
  feature keeps working, just worse;
- the DCT against scipy, since this one is written from scratch;
- LPCC on silence, where the autocorrelation matrix is singular;
- prosody_summary on four very different signals, because the fusion layer
  cannot cope with a NaN in it;
- G.711 round trip SNR against the value the standard promises;
- the SNR after add_noise, measured back out of the signal.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import swarkavach  # noqa: F401
except ImportError:  # running from a checkout without the package installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy import fft as sfft
from scipy import signal as ssig

from swarkavach import channel
from swarkavach.config import CepstralConfig, FrameConfig, ProsodyConfig, SETTINGS, TARGET_SR
from swarkavach.dsp import cepstral, deltas, filterbanks, framing, prosody, spectral, vad

SR = TARGET_SR
SEED = SETTINGS.pipeline.seed


# --------------------------------------------------------------------------
# Signal fixtures, built by hand so the expected answers are known
# --------------------------------------------------------------------------


def sine(freq: float, dur_s: float = 2.0, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def square(freq: float, dur_s: float = 2.0, sr: int = SR) -> np.ndarray:
    """Exact square wave: sign flips every half period, no values near zero.

    Built with integer division rather than sign(sin(...)) on purpose. The sine
    version lands on exact zeros at every period boundary, and then the sign of
    a 1e-16 rounding error decides whether a crossing is counted.
    """
    half = int(round(sr / (2.0 * freq)))
    idx = np.arange(int(dur_s * sr))
    return np.where((idx // half) % 2 == 0, 1.0, -1.0)


def vowel(dur_s: float = 2.0, f0: float = 150.0, sr: int = SR,
          jitter_pct: float = 0.012, shimmer_pct: float = 0.05,
          breath: float = 0.02, seed: int = 3) -> np.ndarray:
    """Source-filter vowel: a jittered pulse train through three formants.

    Resonators run with lfilter, so this is linear in the duration. The jitter
    and shimmer are put in deliberately, which lets the prosody test check that
    the measured values land in the right neighbourhood instead of only
    checking that they are finite.
    """
    n = int(dur_s * sr)
    rng = np.random.default_rng(seed)
    src = np.zeros(n)
    pos = 0.0
    while pos < n:
        src[int(pos)] = 1.0 + shimmer_pct * rng.standard_normal()
        pos += max(sr / (f0 * (1.0 + jitter_pct * rng.standard_normal())), 2.0)
    y = np.zeros(n)
    for fc, bw, g in ((700.0, 90.0, 1.0), (1220.0, 110.0, 0.5), (2600.0, 170.0, 0.25)):
        r = np.exp(-np.pi * bw / sr)
        w = 2 * np.pi * fc / sr
        y += g * ssig.lfilter([1.0], [1.0, -2 * r * np.cos(w), r * r], src)
    y = 0.5 * y / max(np.max(np.abs(y)), 1e-9)
    return y + breath * rng.standard_normal(n) * np.abs(y)


def speech_and_silence(sr: int = SR, seed: int = 5):
    """Two spoken stretches with silence around and between them.

    Returns (signal, [(start, end)]) with the true speech spans in seconds.
    """
    rng = np.random.default_rng(seed)
    gap = lambda d: 0.0005 * rng.standard_normal(int(d * sr))
    a, b = vowel(0.8, 150.0, sr, seed=11), vowel(1.1, 175.0, sr, seed=12)
    x = np.concatenate([gap(0.5), a, gap(0.6), b, gap(0.4)])
    spans = [(0.5, 1.3), (1.9, 3.0)]
    return x + 0.0005 * rng.standard_normal(x.size), spans


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def test_framing_shapes():
    x = sine(220.0, 3.0)
    for frame_len, hop in ((200, 80), (256, 128), (400, 400), (200, 200)):
        f = framing.frame_signal(x, frame_len, hop, "hamming")
        expected = 1 + (x.size - frame_len) // hop
        assert f.shape == (expected, frame_len), (f.shape, expected, frame_len)
        assert f.dtype == np.float64
        assert framing.frame_count(x.size, frame_len, hop) == expected

    # A signal shorter than one frame still yields exactly one padded frame.
    short = framing.frame_signal(np.ones(37), 200, 80, "rect")
    assert short.shape == (1, 200)
    assert np.allclose(short[0, :37], 1.0) and np.allclose(short[0, 37:], 0.0)

    # A frame starts exactly at i * hop.
    ramp = np.arange(1000, dtype=np.float64)
    f = framing.frame_signal(ramp, 100, 50, "rect")
    for i in range(f.shape[0]):
        assert f[i, 0] == i * 50


def test_framing_reconstruction():
    """A rectangular window with hop == frame_len is a lossless partition."""
    x = sine(310.0, 1.0) + 0.1 * sine(1100.0, 1.0)
    f = framing.frame_signal(x, 160, 160, "rect")
    rebuilt = f.ravel()
    assert rebuilt.size == f.shape[0] * 160
    assert np.max(np.abs(rebuilt - x[: rebuilt.size])) < 1e-12

    # And with a window the frames are exactly signal times window.
    w = framing.get_window("hamming", 160)
    fw = framing.frame_signal(x, 160, 160, "hamming")
    assert np.max(np.abs(fw[3] - x[3 * 160: 4 * 160] * w)) < 1e-12


def test_window_sums():
    n = 256
    sums = {}
    for name in framing.WINDOWS:
        w = framing.get_window(name, n)
        assert w.shape == (n,)
        assert np.all(np.isfinite(w))
        assert w.sum() > 0.0
        assert np.max(np.abs(w)) <= 1.0 + 1e-12
        assert np.allclose(w, w[::-1], atol=1e-12), f"{name} is not symmetric"
        sums[name] = w.sum()

    assert np.isclose(sums["rect"], n)
    # Textbook coherent gains: hamming 0.54, hann 0.5, blackman 0.42.
    assert abs(sums["hamming"] / n - 0.54) < 0.01
    assert abs(sums["hann"] / n - 0.50) < 0.01
    assert abs(sums["blackman"] / n - 0.42) < 0.01
    assert sums["rect"] > sums["hamming"] > sums["hann"] > sums["blackman"]

    assert np.isclose(framing.get_window("hamming", n)[0], 0.08, atol=1e-6)
    assert np.isclose(framing.get_window("hann", n)[0], 0.0, atol=1e-12)
    assert np.array_equal(framing.get_window("rect", 16), np.ones(16))
    assert np.array_equal(framing.get_window("boxcar", 16), np.ones(16))

    try:
        framing.get_window("kaiser", 16)
        raise AssertionError("unknown window name should raise")
    except ValueError:
        pass


def test_zero_crossing_rate_square_wave():
    """A square wave has a crossing count that can be worked out by hand."""
    freq, frame_len, hop = 100.0, 200, 200
    half = int(round(SR / (2.0 * freq)))          # 40 samples
    x = square(freq, 2.0)
    frames = framing.frame_signal(x, frame_len, hop, "rect")
    z = framing.zero_crossing_rate(frames)

    # Frame 0 covers samples 0..199, and the sign flips at 40, 80, 120, 160.
    expected_crossings = len([k for k in range(1, frame_len) if k % half == 0])
    assert expected_crossings == 4
    assert np.isclose(z[0], expected_crossings / (frame_len - 1))
    assert np.allclose(z, z[0]), "every frame is aligned the same way here"

    # And it sits close to the analytic rate of 2 f / sr. The gap is pure
    # discretisation: a 200 sample frame holds 2.5 periods, so it can only
    # contain a whole number of crossings.
    assert abs(z.mean() - 2.0 * freq / SR) < 0.006

    # Twice the frequency, twice as many crossings, again countable by hand.
    half2 = int(round(SR / (2.0 * 200.0)))        # 20 samples
    z2 = framing.zero_crossing_rate(framing.frame_signal(square(200.0, 2.0), 200, 200, "rect"))
    assert np.isclose(z2[0], len([k for k in range(1, 200) if k % half2 == 0]) / 199)
    assert z2.mean() > z.mean() * 1.9

    # Ends of the scale: DC never crosses, alternating samples always do.
    assert framing.zero_crossing_rate(np.ones((1, 100)))[0] == 0.0
    alt = np.where(np.arange(100) % 2 == 0, 1.0, -1.0)[None, :]
    assert framing.zero_crossing_rate(alt)[0] == 1.0
    # White noise crosses on about half the sample pairs.
    noise = np.random.default_rng(0).standard_normal((20, 400))
    assert abs(framing.zero_crossing_rate(noise).mean() - 0.5) < 0.05


def test_energy_measures():
    x = sine(200.0, 1.0, amp=0.5)
    frames = framing.frame_signal(x, 200, 200, "rect")
    e = framing.short_time_energy(frames)
    assert e.shape == (frames.shape[0],)
    # Sum of squares of a 0.5 amplitude sine over 200 samples is 200 * 0.25 / 2.
    assert abs(e.mean() - 200 * 0.25 / 2) < 0.5
    assert np.allclose(framing.log_energy(frames), np.log(e), atol=1e-9)
    assert np.all(np.isfinite(framing.log_energy(np.zeros((4, 200)))))
    # RMS dB of a 0.5 amplitude sine is 20 log10(0.5 / sqrt(2)).
    assert abs(framing.rms_db(frames).mean() - 20 * math.log10(0.5 / math.sqrt(2))) < 0.05


# --------------------------------------------------------------------------
# spectral
# --------------------------------------------------------------------------


def test_stft_matches_numpy_fft_on_one_frame():
    cfg = FrameConfig(sr=SR, preemphasis=0.0)     # compare the transform alone
    x = sine(500.0, 1.0) + 0.3 * sine(1700.0, 1.0)
    n_fft = max(cfg.n_fft, cfg.frame_len)

    got = spectral.stft(x, cfg)
    frames = framing.frame_signal(x, cfg.frame_len, cfg.hop_len, cfg.window)
    assert got.shape == (frames.shape[0], n_fft // 2 + 1)

    for i in (0, 5, 37):
        ref = np.fft.rfft(frames[i], n=n_fft)
        assert np.max(np.abs(ref - got[i])) < 1e-10, i

    # Pre-emphasis is applied when the config asks for it, and only then.
    cfg_pre = FrameConfig(sr=SR, preemphasis=0.97)
    pre = spectral.stft(x, cfg_pre)
    frames_pre = framing.frame_signal(framing.preemphasis(x, 0.97),
                                      cfg_pre.frame_len, cfg_pre.hop_len, cfg_pre.window)
    assert np.max(np.abs(np.fft.rfft(frames_pre[5], n=n_fft) - pre[5])) < 1e-10
    assert np.max(np.abs(pre[5] - got[5])) > 1e-6

    # A 500 Hz tone peaks in the 500 Hz bin.
    freqs = spectral.fft_frequencies(cfg)
    peak = int(np.argmax(spectral.power_spectrum(sine(500.0, 1.0), cfg).mean(axis=0)))
    assert abs(freqs[peak] - 500.0) <= SR / n_fft


def test_spectral_descriptors():
    cfg = SETTINGS.frame
    tone = sine(400.0, 2.0)
    noise = 0.5 * np.random.default_rng(1).standard_normal(2 * SR)
    n_frames = framing.frame_count(2 * SR, cfg.frame_len, cfg.hop_len)

    for name, fn in (("centroid", spectral.spectral_centroid),
                     ("bandwidth", spectral.spectral_bandwidth),
                     ("rolloff", spectral.spectral_rolloff),
                     ("flatness", spectral.spectral_flatness),
                     ("flux", spectral.spectral_flux)):
        for sig in (tone, noise, np.zeros(2 * SR)):
            v = fn(sig, cfg)
            assert v.shape == (n_frames,), (name, v.shape)
            assert np.all(np.isfinite(v)), name

    # Noise is brighter and much flatter than a low tone.
    assert spectral.spectral_centroid(noise, cfg).mean() > \
        spectral.spectral_centroid(tone, cfg).mean() + 500
    assert spectral.spectral_flatness(noise, cfg).mean() > 0.1
    assert spectral.spectral_flatness(tone, cfg).mean() < 0.01
    assert np.all(spectral.spectral_flatness(noise, cfg) <= 1.0)
    assert spectral.spectral_flux(tone, cfg)[0] == 0.0

    # Passing a precomputed power spectrum gives the same answer.
    S = spectral.power_spectrum(tone, cfg)
    assert np.allclose(spectral.spectral_centroid(S, cfg),
                       spectral.spectral_centroid(tone, cfg))

    db = spectral.spectrogram_db(tone, cfg, top_db=80)
    assert db.shape == (n_frames, max(cfg.n_fft, cfg.frame_len) // 2 + 1)
    assert abs(db.max()) < 1e-9 and db.min() >= -80.0 - 1e-9


# --------------------------------------------------------------------------
# filterbanks
# --------------------------------------------------------------------------


def test_frequency_scale_round_trips():
    f = np.array([0.0, 60.0, 300.0, 1000.0, 3400.0, 3999.0])
    assert np.allclose(filterbanks.mel_to_hz(filterbanks.hz_to_mel(f)), f, atol=1e-6)
    assert np.allclose(filterbanks.erb_to_hz(filterbanks.hz_to_erb(f)), f, atol=1e-6)
    # The mel scale is near linear below 1 kHz by construction.
    assert abs(filterbanks.hz_to_mel(1000.0) - 1000.0) < 1.0
    # ERB at 1 kHz is about 133 Hz (Glasberg and Moore).
    assert abs(filterbanks.erb_bandwidth(1000.0) - 132.6) < 1.0
    assert filterbanks.hz_to_mel(3000.0) < 3000.0        # compressive above 1 kHz
    for scale in (filterbanks.hz_to_mel, filterbanks.hz_to_erb):
        assert np.all(np.diff(scale(np.linspace(0, 4000, 50))) > 0)


def test_filterbanks_have_no_empty_rows():
    """The bug this test exists for: an all-zero row silently kills a channel."""
    cep = SETTINGS.cepstral
    configs = [
        # sr, n_fft, n_filters, fmin, fmax
        (SR, 512, cep.n_filters, cep.fmin, cep.fmax),   # the project default
        (SR, 512, 20, 0.0, 4000.0),
        (SR, 512, 40, 300.0, 3400.0),                   # telephone passband only
        (SR, 256, 64, 20.0, 3900.0),                    # more filters than bins can resolve
        (SR, 1024, 40, 60.0, 3800.0),
        (16000, 512, 40, 50.0, 7800.0),                 # wideband, for the ASR side
    ]
    for kind in ("mel", "linear", "gammatone", "cqt"):
        for sr, n_fft, n_filters, fmin, fmax in configs:
            fb = filterbanks.get_filterbank(kind, sr, n_fft, n_filters, fmin, fmax)
            tag = f"{kind} sr={sr} n_fft={n_fft} n={n_filters} [{fmin},{fmax}]"

            assert fb.shape == (n_filters, n_fft // 2 + 1), tag
            assert np.all(np.isfinite(fb)), tag
            assert np.all(fb >= 0.0), tag

            sums = fb.sum(axis=1)
            assert np.all(sums > 0.0), f"{tag}: {int((sums <= 0).sum())} empty rows"
            assert np.count_nonzero(fb.sum(axis=1) == 0) == 0, tag
            # Every row must have at least one bin actually carrying weight.
            assert np.all(fb.max(axis=1) > 0.0), tag

            centres = filterbanks.filterbank_centres(fb, sr, n_fft)
            assert np.all(np.diff(centres) > 0), f"{tag}: centres not increasing"
            assert centres[0] >= 0.0 and centres[-1] <= sr / 2.0, tag


def test_filterbank_shapes_and_spacing():
    fb_mel = filterbanks.mel_filterbank(SR, 512, 40, 60.0, 3800.0)
    fb_lin = filterbanks.linear_filterbank(SR, 512, 40, 60.0, 3800.0)
    fb_gam = filterbanks.gammatone_filterbank(SR, 512, 40, 60.0, 3800.0)
    fb_cqt = filterbanks.cqt_like_filterbank(SR, 512, 40, 60.0, 3800.0)
    for fb in (fb_mel, fb_lin, fb_gam, fb_cqt):
        assert fb.shape == (40, 257)

    c_lin = filterbanks.filterbank_centres(fb_lin, SR, 512)
    spacing = np.diff(c_lin)
    assert np.std(spacing) / np.mean(spacing) < 0.05, "linear spacing is not uniform"

    c_cqt = filterbanks.filterbank_centres(fb_cqt, SR, 512)
    ratios = c_cqt[1:] / c_cqt[:-1]
    assert np.std(ratios) / np.mean(ratios) < 0.15, "constant-Q spacing is not geometric"

    # Mel and gammatone both crowd the low frequencies relative to linear.
    c_mel = filterbanks.filterbank_centres(fb_mel, SR, 512)
    c_gam = filterbanks.filterbank_centres(fb_gam, SR, 512)
    assert c_mel[19] < c_lin[19] and c_gam[19] < c_lin[19]

    # Peak normalised triangles, and an area normalised bank sums to one. The
    # apex is never exactly 1.0 because the triangles are sampled at the FFT
    # bin centres and a bin rarely lands exactly on a centre frequency.
    assert 0.9 < fb_mel.max() <= 1.0
    area = filterbanks.mel_filterbank(SR, 512, 40, 60.0, 3800.0, norm="area")
    assert np.allclose(area.sum(axis=1), 1.0)

    # get_filterbank hands back a cached, read only array.
    a = filterbanks.get_filterbank("mel", SR, 512, 40, 60.0, 3800.0)
    b = filterbanks.get_filterbank("mel", SR, 512, 40, 60.0, 3800.0)
    assert a is b and not a.flags.writeable


# --------------------------------------------------------------------------
# cepstral
# --------------------------------------------------------------------------


def test_dct2_matches_scipy():
    rng = np.random.default_rng(2)
    for n_in in (8, 20, 40, 64):
        x = rng.standard_normal((7, n_in))
        ref = sfft.dct(x, type=2, axis=-1, norm="ortho")
        assert np.max(np.abs(cepstral.dct2(x, n_in) - ref)) < 1e-10, n_in
        for n_out in (1, 13, n_in):
            got = cepstral.dct2(x, n_out)
            keep = min(n_out, n_in)
            assert got.shape == (7, n_out)
            assert np.max(np.abs(got[:, :keep] - ref[:, :keep])) < 1e-10, (n_in, n_out)
            assert np.allclose(got[:, keep:], 0.0), "padding must be zeros"

    # 1-D input stays 1-D.
    v = rng.standard_normal(40)
    assert cepstral.dct2(v, 20).shape == (20,)
    assert np.max(np.abs(cepstral.dct2(v, 40) - sfft.dct(v, type=2, norm="ortho"))) < 1e-10

    # Orthonormal means the matrix is orthogonal and energy is preserved.
    d = cepstral.dct_matrix(32, 32)
    assert np.max(np.abs(d @ d.T - np.eye(32))) < 1e-10
    assert abs(np.sum(cepstral.dct2(v, 40) ** 2) - np.sum(v ** 2)) < 1e-8

    # Asking for more coefficients than exist pads rather than raising.
    assert cepstral.dct2(rng.standard_normal((3, 5)), 9).shape == (3, 9)


def test_mfcc_shape_and_stability():
    cfg, cep = SETTINGS.frame, SETTINGS.cepstral
    x = vowel(2.0, 150.0)
    n_frames = framing.frame_count(x.size, cfg.frame_len, cfg.hop_len)

    m = cepstral.mfcc(x, cfg, cep)
    assert m.shape == (n_frames, cep.n_ceps)
    assert np.all(np.isfinite(m))

    # Deterministic.
    assert np.array_equal(m, cepstral.mfcc(x, cfg, cep))

    # Scaling the signal moves c0 (the energy term) and leaves the shape alone.
    loud = cepstral.mfcc(2.0 * x, cfg, cep)
    assert np.max(np.abs(loud[:, 1:] - m[:, 1:])) < 1e-6
    assert np.allclose(loud[:, 0] - m[:, 0], math.log(4.0), atol=1e-6)

    # A small perturbation gives a small change, not a different feature.
    noisy = x + 1e-4 * np.random.default_rng(4).standard_normal(x.size)
    diff = np.abs(cepstral.mfcc(noisy, cfg, cep) - m)
    assert diff.mean() < 0.1 * m.std(), (diff.mean(), m.std())

    # A different number of coefficients only truncates.
    fewer = cepstral.mfcc(x, cfg, CepstralConfig(n_ceps=13))
    assert fewer.shape == (n_frames, 13)
    assert np.max(np.abs(fewer - m[:, :13])) < 1e-9


def test_all_five_front_ends():
    cfg, cep = SETTINGS.frame, SETTINGS.cepstral
    x = vowel(2.0, 150.0)
    n_frames = framing.frame_count(x.size, cfg.frame_len, cfg.hop_len)

    assert set(cepstral.FEATURE_EXTRACTORS) == {"mfcc", "gfcc", "lfcc", "cqcc", "lpcc"}
    outputs = {}
    for name, fn in cepstral.FEATURE_EXTRACTORS.items():
        feat = fn(x, cfg, cep)
        assert feat.shape == (n_frames, cep.n_ceps), name
        assert np.all(np.isfinite(feat)), name
        assert feat.std() > 1e-6, f"{name} is constant, the filterbank may be dead"
        outputs[name] = feat
        assert np.array_equal(cepstral.extract(x, name, cfg, cep), feat)

    # The five front ends must actually differ from each other.
    names = sorted(outputs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(outputs[a], outputs[b]), f"{a} equals {b}"


def test_feature_extraction_speed():
    """A 30 second call at 8 kHz, all five front ends, under 3 seconds."""
    cfg, cep = SETTINGS.frame, SETTINGS.cepstral
    x = vowel(30.0, 140.0)
    t0 = time.time()
    for name in cepstral.FEATURE_EXTRACTORS:
        cepstral.extract(x, name, cfg, cep)
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"all five front ends took {elapsed:.2f} s"


def test_lpcc_finite_on_silence():
    cfg = SETTINGS.frame
    n_frames = framing.frame_count(SR, cfg.frame_len, cfg.hop_len)

    for name, sig in (
        ("zeros", np.zeros(SR)),
        ("dc", np.ones(SR)),
        ("tiny", np.full(SR, 1e-12)),
        ("one sample", np.zeros(1)),
        ("clipped", np.sign(sine(200.0, 1.0))),
        ("half silence", np.concatenate([np.zeros(SR // 2), vowel(0.5, 150.0)])),
    ):
        c = cepstral.lpcc(sig, cfg, order=16, n_ceps=20)
        assert np.all(np.isfinite(c)), f"lpcc produced NaN or inf on {name}"
        assert c.shape[1] == 20, name
        if name not in ("one sample",):
            assert c.shape[0] == framing.frame_count(sig.size, cfg.frame_len, cfg.hop_len)

    # Silence really is flagged: c0 sits at the energy floor.
    c_sil = cepstral.lpcc(np.zeros(SR), cfg)
    assert np.all(c_sil[:, 0] < -20.0)
    assert np.allclose(c_sil[:, 1:], 0.0)

    # On a real vowel it is not degenerate.
    c = cepstral.lpcc(vowel(1.0, 150.0), cfg)
    assert np.all(np.isfinite(c)) and c[:, 1:].std() > 1e-6
    assert c.shape == (n_frames, 20)


def test_levinson_durbin_recovers_a_known_filter():
    """Drive white noise through a known all-pole filter and read the poles back."""
    rng = np.random.default_rng(9)
    a_true = np.array([1.0, -1.6, 0.9])           # a stable two pole resonator
    x = ssig.lfilter([1.0], a_true, rng.standard_normal(20000))
    frames = x[None, :]
    r = cepstral.autocorrelation(frames, 2)
    a_est, err = cepstral.levinson_durbin(r, 2)
    assert np.max(np.abs(a_est[0] - a_true)) < 0.05, a_est[0]
    assert err[0] > 0

    # The autocorrelation itself matches a direct computation.
    seg = x[:512][None, :]
    direct = np.array([np.dot(seg[0, : 512 - k], seg[0, k:]) for k in range(9)])
    assert np.max(np.abs(cepstral.autocorrelation(seg, 8)[0] - direct)) < 1e-6

    # Every estimated synthesis filter is stable (all roots inside the circle).
    frames = framing.frame_signal(vowel(1.0, 150.0), 200, 80)
    a, e = cepstral.levinson_durbin(cepstral.autocorrelation(frames, 16), 16)
    assert np.all(e > 0)
    assert all(np.max(np.abs(np.roots(row))) < 1.0 for row in a[:20])


# --------------------------------------------------------------------------
# deltas
# --------------------------------------------------------------------------


def test_delta_shapes_and_values():
    rng = np.random.default_rng(6)
    feat = rng.standard_normal((120, 20))

    d = deltas.delta(feat, width=2)
    assert d.shape == feat.shape
    assert np.all(np.isfinite(d))

    full = deltas.add_deltas(feat, width=2, order=2)
    assert full.shape == (120, 60)
    assert np.array_equal(full[:, :20], feat)
    assert np.array_equal(full[:, 20:40], d)
    assert np.array_equal(full[:, 40:], deltas.delta(d, width=2))
    assert deltas.add_deltas(feat, order=1).shape == (120, 40)
    assert deltas.add_deltas(feat, order=0).shape == (120, 20)

    # A ramp has a constant slope, which the regression must return exactly
    # away from the replicated edges.
    slope = np.array([1.0, -2.0, 0.5])
    ramp = np.arange(40, dtype=np.float64)[:, None] * slope
    dr = deltas.delta(ramp, width=2)
    assert np.allclose(dr[5:35], slope, atol=1e-12)

    # A constant feature has zero derivative everywhere.
    assert np.allclose(deltas.delta(np.ones((30, 4))), 0.0)
    # One frame in, one frame out.
    assert deltas.delta(np.ones((1, 4))).shape == (1, 4)
    # Deltas of a real feature matrix stay finite.
    assert np.all(np.isfinite(deltas.add_deltas(cepstral.mfcc(vowel(1.0)))))


def test_cmvn():
    rng = np.random.default_rng(8)
    feat = rng.normal(loc=-4.0, scale=7.0, size=(300, 20))
    out = deltas.cmvn(feat)

    assert out.shape == feat.shape
    assert np.max(np.abs(out.mean(axis=0))) < 1e-10
    assert np.max(np.abs(out.std(axis=0) - 1.0)) < 1e-10
    assert np.all(np.isfinite(out))

    # Mean only leaves the scale alone.
    mean_only = deltas.cmvn(feat, var_norm=False)
    assert np.max(np.abs(mean_only.mean(axis=0))) < 1e-10
    assert np.allclose(mean_only.std(axis=0), feat.std(axis=0))

    # CMVN removes a constant channel offset, which is the point of it: a
    # convolutional channel is additive in the cepstral domain.
    shifted = feat + np.array([2.0] * 20)
    assert np.max(np.abs(deltas.cmvn(shifted) - out)) < 1e-9

    # A dead channel must not blow up.
    dead = feat.copy()
    dead[:, 3] = 1.5
    got = deltas.cmvn(dead)
    assert np.all(np.isfinite(got)) and np.allclose(got[:, 3], 0.0)
    assert np.all(np.isfinite(deltas.cmvn(np.zeros((10, 4)))))


# --------------------------------------------------------------------------
# prosody
# --------------------------------------------------------------------------


def test_f0_track_on_known_pitches():
    cfg = SETTINGS.prosody
    for want in (90.0, 150.0, 220.0, 330.0):
        x = vowel(1.5, want, jitter_pct=0.004, shimmer_pct=0.02, breath=0.01)
        f0, voiced = prosody.f0_track(x, SR, cfg)
        assert f0.shape == voiced.shape
        assert voiced.mean() > 0.7, f"{want} Hz: only {voiced.mean():.2f} voiced"
        got = float(np.median(f0[voiced]))
        assert abs(got - want) / want < 0.05, f"wanted {want} Hz, tracked {got:.1f} Hz"
        # No octave errors: nothing near half or double the true value.
        rel = f0[voiced] / want
        assert np.mean((rel > 0.9) & (rel < 1.1)) > 0.9

    # A pure tone is tracked too, and unvoiced material is not.
    f0, voiced = prosody.f0_track(sine(200.0, 1.5), SR, cfg)
    assert abs(float(np.median(f0[voiced])) - 200.0) < 5.0
    _, v_sil = prosody.f0_track(np.zeros(SR), SR, cfg)
    assert not v_sil.any()
    _, v_noise = prosody.f0_track(0.3 * np.random.default_rng(1).standard_normal(2 * SR), SR, cfg)
    assert v_noise.mean() < 0.15, f"white noise looks {v_noise.mean():.2f} voiced"
    # Unvoiced frames report exactly 0.0 Hz.
    f0, voiced = prosody.f0_track(np.concatenate([np.zeros(SR), vowel(1.0, 160.0)]), SR, cfg)
    assert np.all(f0[~voiced] == 0.0)


def test_voice_quality_measures():
    cfg = SETTINGS.prosody
    steady = vowel(2.0, 150.0, jitter_pct=0.002, shimmer_pct=0.01, breath=0.005)
    rough = vowel(2.0, 150.0, jitter_pct=0.030, shimmer_pct=0.15, breath=0.10)

    f0_s, v_s = prosody.f0_track(steady, SR, cfg)
    f0_r, v_r = prosody.f0_track(rough, SR, cfg)

    j_s, j_r = prosody.jitter(f0_s, v_s), prosody.jitter(f0_r, v_r)
    sh_s, sh_r = prosody.shimmer(steady, SR, f0_s, v_s), prosody.shimmer(rough, SR, f0_r, v_r)
    h_s, h_r = prosody.hnr(steady, SR, f0_s, v_s), prosody.hnr(rough, SR, f0_r, v_r)

    for value in (j_s, j_r, sh_s, sh_r, h_s, h_r):
        assert math.isfinite(value)
    assert 0.0 <= j_s < j_r, (j_s, j_r)
    assert 0.0 <= sh_s < sh_r, (sh_s, sh_r)
    assert h_s > h_r, (h_s, h_r)          # more noise, lower harmonic to noise

    # A perfect sine is the limiting case: no perturbation, HNR at the ceiling.
    pure = sine(200.0, 2.0)
    f0_p, v_p = prosody.f0_track(pure, SR, cfg)
    assert prosody.jitter(f0_p, v_p) < 0.01
    assert prosody.hnr(pure, SR, f0_p, v_p) > 25.0

    # Nothing to measure means 0.0, not a crash.
    empty = np.zeros(0, dtype=bool)
    assert prosody.jitter(np.zeros(0), empty) == 0.0
    assert prosody.shimmer(np.zeros(SR), SR) == 0.0
    assert prosody.hnr(np.zeros(SR), SR) == 0.0


def test_prosody_summary_always_finite():
    """The contract PIM depends on: every key, every time, always finite."""
    rng = np.random.default_rng(SEED)
    cases = {
        "silence": np.zeros(3 * SR),
        "white noise": 0.3 * rng.standard_normal(3 * SR),
        "pure sine": sine(200.0, 3.0),
        "synthetic vowel": vowel(3.0, 150.0),
        "vowel with pauses": speech_and_silence()[0],
        "dc": np.ones(2 * SR),
        "single sample": np.array([0.5]),
        "two samples": np.array([0.5, -0.5]),
        "empty": np.zeros(0),
        "clipped square": square(180.0, 2.0),
        "tiny": 1e-9 * rng.standard_normal(SR),
        "huge": 1e6 * rng.standard_normal(SR),
        "nan input": np.concatenate([vowel(1.0), np.array([np.nan, np.inf, -np.inf])]),
        "very short": vowel(0.05, 150.0),
        "one frame": vowel(0.025, 150.0),
    }
    for name, sig in cases.items():
        summary = prosody.prosody_summary(sig, SR)
        assert isinstance(summary, dict), name
        assert set(summary) == set(prosody.PROSODY_KEYS), (
            name, set(prosody.PROSODY_KEYS) ^ set(summary))
        for key, value in summary.items():
            assert isinstance(value, float), f"{name}/{key} is {type(value)}"
            assert math.isfinite(value), f"{name}/{key} = {value}"
        assert 0.0 <= summary["voiced_ratio"] <= 1.0, name
        assert 0.0 <= summary["pause_ratio"] <= 1.0, name
        assert summary["jitter"] >= 0.0 and summary["shimmer"] >= 0.0, name
        assert summary["f0_mean"] >= 0.0, name

    # Silence really does report nothing rather than noise.
    sil = prosody.prosody_summary(np.zeros(3 * SR), SR)
    assert sil["voiced_ratio"] == 0.0 and sil["f0_mean"] == 0.0
    assert sil["pause_ratio"] == 1.0

    # A real vowel reports a sensible pitch and a high voiced ratio.
    v = prosody.prosody_summary(vowel(3.0, 150.0), SR)
    assert 130.0 < v["f0_mean"] < 170.0, v["f0_mean"]
    assert v["voiced_ratio"] > 0.8
    assert v["hnr"] > 0.0
    assert 0.0 < v["jitter"] < 0.2

    # Same input, same numbers.
    again = prosody.prosody_summary(vowel(3.0, 150.0), SR)
    assert all(v[k] == again[k] for k in prosody.PROSODY_KEYS)


def test_speech_rate_and_pauses():
    x, _ = speech_and_silence()
    rate = prosody.speech_rate(x, SR)
    assert math.isfinite(rate) and 0.0 <= rate <= 12.0
    assert prosody.speech_rate(np.zeros(3 * SR), SR) == 0.0
    assert prosody.speech_rate(np.zeros(10), SR) == 0.0

    # More silence means a higher pause ratio.
    dense = vowel(3.0, 150.0)
    sparse = np.concatenate([vowel(0.4, 150.0), np.zeros(2 * SR), vowel(0.4, 150.0)])
    assert prosody.prosody_summary(sparse, SR)["pause_ratio"] > \
        prosody.prosody_summary(dense, SR)["pause_ratio"]


# --------------------------------------------------------------------------
# vad
# --------------------------------------------------------------------------


def test_vad_finds_speech_in_speech_plus_silence():
    x, spans = speech_and_silence()
    cfg = SETTINGS.vad

    mask = vad.vad_mask(x, SR, cfg)
    frame_len, hop = vad.frame_grid(SR, cfg)
    assert mask.dtype == bool
    assert mask.shape == (framing.frame_count(x.size, frame_len, hop),)
    assert mask.any(), "no speech found at all"
    assert not mask.all(), "everything marked as speech"

    segments = vad.speech_segments(x, SR, cfg)
    assert len(segments) == len(spans), f"expected {len(spans)} segments, got {segments}"
    for (got_a, got_b), (want_a, want_b) in zip(segments, spans):
        assert abs(got_a - want_a) < 0.15, (segments, spans)
        assert abs(got_b - want_b) < 0.15, (segments, spans)
        assert got_b > got_a

    # Frame level agreement with the truth, both ways round.
    times = np.arange(mask.size) * hop / SR + frame_len / (2 * SR)
    truth = np.zeros(mask.size, dtype=bool)
    for a, b in spans:
        truth |= (times >= a) & (times <= b)
    agree = float(np.mean(mask == truth))
    assert agree > 0.90, f"VAD agrees with the truth on only {agree:.2f} of frames"
    assert mask[truth].mean() > 0.90, "missed too much speech"

    # False positives are counted outside an 80 ms guard band around each true
    # boundary, because the config asks for 40 ms of padding on every segment
    # and a frame straddling the edge is genuinely part speech. Segment timing
    # is already checked above, to 150 ms.
    guard = np.zeros(mask.size, dtype=bool)
    for a, b in spans:
        guard |= (times >= a - 0.08) & (times <= b + 0.08)
    assert mask[~guard].mean() < 0.02, "silence away from the edges marked as speech"

    # Silence, in every flavour, is not speech.
    assert not vad.vad_mask(np.zeros(2 * SR), SR, cfg).any()
    assert vad.speech_ratio(1e-6 * np.random.default_rng(0).standard_normal(2 * SR), SR) == 0.0
    assert vad.speech_segments(np.zeros(2 * SR), SR, cfg) == []

    # A call with no pauses is speech throughout, which is the case a naive
    # percentile threshold gets wrong.
    assert vad.speech_ratio(vowel(2.0, 150.0), SR, cfg) > 0.9

    # trim_silence keeps the speech and drops the rest.
    trimmed = vad.trim_silence(x, SR, cfg)
    assert trimmed.size < x.size
    assert abs(trimmed.size / SR - (spans[-1][1] - spans[0][0])) < 0.3
    packed = vad.trim_silence(x, SR, cfg, keep_internal=False)
    assert packed.size <= trimmed.size
    assert vad.trim_silence(np.zeros(SR), SR, cfg).size == SR   # nothing found, nothing cut


# --------------------------------------------------------------------------
# channel
# --------------------------------------------------------------------------


def test_mu_law_round_trip():
    x = vowel(1.0, 150.0) * 0.8
    y = channel.mu_law_roundtrip(x)
    assert y.shape == x.shape and np.all(np.isfinite(y))

    err = y - x
    snr = 10 * math.log10(np.mean(x ** 2) / np.mean(err ** 2))
    # G.711 promises roughly 38 dB across its dynamic range window.
    assert snr > 30.0, f"mu-law round trip SNR is only {snr:.1f} dB"
    assert np.max(np.abs(err)) < 0.05
    assert np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(x ** 2)) < 0.05

    y_a = channel.a_law_roundtrip(x)
    snr_a = 10 * math.log10(np.mean(x ** 2) / np.mean((y_a - x) ** 2))
    assert snr_a > 30.0, f"A-law round trip SNR is only {snr_a:.1f} dB"

    # The codes are exactly 8 bits and they use most of the range.
    codes = channel.linear_to_ulaw(np.rint(x * 32768).astype(np.int32))
    assert codes.dtype == np.uint8 and len(np.unique(codes)) > 50

    # Values fixed by the standard.
    assert int(channel.linear_to_ulaw(np.array([0]))[0]) == 0xFF
    assert int(channel.linear_to_alaw(np.array([0]))[0]) == 0xD5
    assert int(channel.linear_to_ulaw(np.array([32767]))[0]) == 0x80
    assert int(channel.linear_to_ulaw(np.array([-32768]))[0]) == 0x00

    # Encoding is monotonic in the decoded value, and quiet samples are coded
    # more finely than loud ones (that is the whole idea of companding).
    quiet = np.arange(-200, 200, dtype=np.int32)
    loud = np.arange(20000, 20400, dtype=np.int32)
    assert len(np.unique(channel.linear_to_ulaw(quiet))) > \
        len(np.unique(channel.linear_to_ulaw(loud)))

    # Idempotent: a signal already on the mu-law grid survives another pass.
    assert np.max(np.abs(channel.mu_law_roundtrip(y) - y)) < 1e-9


def test_g711_matches_reference_implementation():
    """Cross check against the stdlib audioop C code, when it is available.

    audioop is the reference G.711 implementation shipped with CPython. It was
    removed in 3.13, so this test skips itself rather than failing there.
    """
    try:
        import audioop  # type: ignore
    except Exception:
        return

    rng = np.random.default_rng(0)
    vals = np.concatenate([
        np.arange(-32768, 32768, 37),
        rng.integers(-32768, 32768, 20000),
        np.array([0, 1, -1, 32767, -32768, 132, -132]),
    ]).astype(np.int32)
    pcm = vals.astype("<i2").tobytes()

    ref_u = np.frombuffer(audioop.lin2ulaw(pcm, 2), dtype=np.uint8)
    assert np.array_equal(channel.linear_to_ulaw(vals), ref_u)
    ref_ud = np.frombuffer(audioop.ulaw2lin(ref_u.tobytes(), 2), dtype="<i2").astype(np.int32)
    assert np.array_equal(channel.ulaw_to_linear(ref_u), ref_ud)

    ref_a = np.frombuffer(audioop.lin2alaw(pcm, 2), dtype=np.uint8)
    assert np.array_equal(channel.linear_to_alaw(vals), ref_a)
    ref_ad = np.frombuffer(audioop.alaw2lin(ref_a.tobytes(), 2), dtype="<i2").astype(np.int32)
    assert np.array_equal(channel.alaw_to_linear(ref_a), ref_ad)


def test_add_noise_hits_the_requested_snr():
    x = vowel(2.0, 150.0)
    for kind in ("white", "hum", "babble"):
        for want in (30.0, 20.0, 10.0, 5.0, 0.0, -5.0):
            y = channel.add_noise(x, want, kind, SR, seed=SEED)
            assert y.shape == x.shape and np.all(np.isfinite(y))
            got = channel.measured_snr(x, y)
            assert abs(got - want) < 1.0, f"{kind} at {want} dB measured {got:.2f} dB"

    # Seeded and reproducible, and different seeds really differ.
    a = channel.add_noise(x, 10.0, "babble", SR, seed=1)
    b = channel.add_noise(x, 10.0, "babble", SR, seed=1)
    c = channel.add_noise(x, 10.0, "babble", SR, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)

    # No global random state is touched.
    np.random.seed(0)
    first = channel.add_noise(x, 10.0, "white", SR, seed=1)
    np.random.seed(999)
    assert np.array_equal(first, channel.add_noise(x, 10.0, "white", SR, seed=1))

    # Noise shapes are what they claim: hum has energy at 50 Hz and its
    # harmonics, babble sits inside the speech band, white is flat.
    n = 4 * SR
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectra = {k: np.abs(np.fft.rfft(channel.make_noise(n, k, SR, seed=1) * np.hanning(n)))
               for k in ("white", "hum", "babble")}
    hum = spectra["hum"]
    at_50 = hum[np.argmin(np.abs(freqs - 50.0))]
    off_peak = np.median(hum[(freqs > 20) & (freqs < 400)])
    assert at_50 > 20 * off_peak, "hum has no 50 Hz component"
    white_flat = spectra["white"][(freqs > 200) & (freqs < 3600)]
    assert white_flat.std() / white_flat.mean() < 1.0
    for kind in ("white", "hum", "babble"):
        noise = channel.make_noise(n, kind, SR, seed=1)
        assert abs(np.sqrt(np.mean(noise ** 2)) - 1.0) < 1e-9   # unit power

    # Silence in, silence out. No division by zero.
    assert np.array_equal(channel.add_noise(np.zeros(100), 10.0, "white", SR), np.zeros(100))
    try:
        channel.add_noise(x, 10.0, "cafeteria", SR)
        raise AssertionError("unknown noise kind should raise")
    except ValueError:
        pass


def test_codecs_and_degrade():
    x = vowel(2.0, 150.0)
    for name in channel.CODECS:
        y, sr_out = channel.apply_codec(x, SR, name, seed=1)
        assert np.all(np.isfinite(y)), name
        assert sr_out == SR, name
        assert y.size > 0, name
        assert np.max(np.abs(y)) < 2.0, name

        y2, sr2, info = channel.apply_codec(x, SR, name, return_info=True, seed=1)
        assert np.array_equal(y, y2)
        assert set(info) >= {"codec", "label", "real_codec", "backend"}
        assert isinstance(info["real_codec"], bool)

    # G.711 is always the real thing. GSM and AMR are real only with ffmpeg,
    # and they must say so when they are not.
    for name in ("g711u", "g711a", "narrowband", "packet_loss", "clean"):
        _, _, info = channel.apply_codec(x, SR, name, return_info=True)
        assert info["real_codec"] is True, name
    for name in ("gsm", "amrnb"):
        _, _, info = channel.apply_codec(x, SR, name, return_info=True)
        assert info["real_codec"] is channel.codec_available(name), name
        if not info["real_codec"]:
            assert info["backend"] == "numpy-approx" and "note" in info

    # degrade reports everything needed to reproduce the condition.
    y, info = channel.degrade(x, SR, codec="g711u", snr_db=15.0, noise_kind="babble", seed=3)
    assert np.all(np.isfinite(y))
    for key in ("codec", "snr_db", "noise_kind", "seed", "real_codec", "sr", "label"):
        assert key in info, key
    assert abs(info["snr_measured_db"] - 15.0) < 1.0
    assert np.array_equal(y, channel.degrade(x, SR, codec="g711u", snr_db=15.0,
                                             noise_kind="babble", seed=3)[0])

    # Clean with no noise is a pass through.
    y_clean, info = channel.degrade(x, SR, codec="clean", snr_db=None)
    assert np.max(np.abs(y_clean - x)) < 1e-12
    assert info["noise_kind"] is None

    try:
        channel.apply_codec(x, SR, "opus")
        raise AssertionError("unknown codec should raise")
    except ValueError:
        pass


def test_telephone_band_and_packet_loss():
    x = vowel(2.0, 150.0)
    band = channel.telephone_band(x, SR)
    assert band.shape == x.shape and np.all(np.isfinite(band))

    n = band.size
    freqs = np.fft.rfftfreq(n, 1 / SR)
    before = np.abs(np.fft.rfft(x * np.hanning(n)))
    after = np.abs(np.fft.rfft(band * np.hanning(n)))
    passband = (freqs > 500) & (freqs < 3000)
    stopband = freqs < 150
    assert after[passband].mean() > 0.5 * before[passband].mean()
    assert after[stopband].mean() < 0.05 * before[stopband].mean()

    lost = channel.packet_loss(x, SR, loss_rate=0.2, burst_ms=20.0, seed=1)
    assert lost.shape == x.shape and np.all(np.isfinite(lost))
    assert np.mean(np.abs(lost) < 1e-12) > 0.05, "no packets were actually dropped"
    assert np.array_equal(lost, channel.packet_loss(x, SR, 0.2, 20.0, seed=1))
    assert not np.array_equal(lost, channel.packet_loss(x, SR, 0.2, 20.0, seed=2))
    assert np.array_equal(channel.packet_loss(x, SR, 0.0, 20.0, seed=1), x)


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------


def test_pipeline_survives_a_degraded_call():
    """The whole chain, on a signal that has been through noise and a codec."""
    x, _ = speech_and_silence()
    for codec in ("clean", "g711u", "g711a", "gsm", "narrowband", "packet_loss"):
        y, info = channel.degrade(x, SR, codec=codec, snr_db=15.0, noise_kind="babble", seed=2)
        for name in cepstral.FEATURE_EXTRACTORS:
            feat = cepstral.extract(y, name)
            assert np.all(np.isfinite(feat)), f"{codec}/{name}"
            assert np.all(np.isfinite(deltas.cmvn(deltas.add_deltas(feat)))), f"{codec}/{name}"
        summary = prosody.prosody_summary(y, SR)
        assert all(math.isfinite(v) for v in summary.values()), codec
        assert vad.vad_mask(y, SR).any(), codec


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    total = time.time()
    print(f"running {len(tests)} tests from {Path(__file__).name}\n")
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name:52s} {time.time() - t0:6.2f} s")
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            failures.append((name, exc))
            print(f"  FAIL  {name:52s} {time.time() - t0:6.2f} s")
            print(f"        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed "
          f"in {time.time() - total:.2f} s")
    if failures:
        import traceback
        for name, exc in failures:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.exit(1 if failures else 0)
