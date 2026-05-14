"""
CTC fingerspelling classifier — wraps the ONNX-exported transformer.

Input:  a (T, N_FEATURES) float32 numpy array (one signed phrase, T frames)
        plus an explicit (T, N_LANDMARKS) bool missing mask.
Output: decoded string.

Supports both greedy and beam-search decoding. Beam search is the default
because it typically gets 0.03-0.05 CER reduction over greedy with no
retraining cost.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .ctc_landmarks import normalize_sequence

_ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
_ONNX = _ARTIFACTS / "asl_ctc.onnx"
_META = _ARTIFACTS / "asl_ctc_meta.json"

MAX_FRAMES = 256  # match training-time max_frames

# Beam-search hyperparameters
DEFAULT_BEAM_WIDTH = 10
# Numerical floor for log-probability addition (avoid -inf collapses)
NEG_INF = -1e30


def _greedy_decode(log_probs: np.ndarray, blank: int, idx_to_char: dict[int, str]) -> str:
    """log_probs: (T, V). Greedy collapse repeats, drop blanks."""
    preds = log_probs.argmax(axis=-1)
    out: list[str] = []
    prev = -1
    for v in preds:
        v = int(v)
        if v != prev:
            if v != blank and v in idx_to_char:
                out.append(idx_to_char[v])
            prev = v
    return "".join(out)


def _logsumexp(a: float, b: float) -> float:
    """Numerically-stable log(exp(a) + exp(b))."""
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _beam_search_decode(
    log_probs: np.ndarray,
    blank: int,
    idx_to_char: dict[int, str],
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> str:
    """
    Standard CTC prefix beam search. log_probs shape: (T, V) where the last
    index is the blank.

    Each beam tracks two probabilities for a given prefix:
      - p_b: prob that the prefix ends in a blank (next-token same character
             yields a new emission rather than a repeat collapse).
      - p_nb: prob that the prefix ends in a non-blank (same character would
              collapse via the repeat rule).

    Reference: Hannun et al., "First-Pass Large Vocabulary Continuous Speech
    Recognition using Bi-Directional Recurrent DNNs" (2014), Algorithm 1.
    """
    T, V = log_probs.shape

    # Beams: dict from prefix tuple (chars as ints) -> (log_p_b, log_p_nb)
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(T):
        next_beams: dict[tuple, tuple[float, float]] = {}
        lp_t = log_probs[t]  # (V,)

        # Prune the symbol axis to top-K candidates for efficiency. With V=60
        # this is cheap, but it keeps the inner loop predictable.
        top_k = min(beam_width * 2, V)
        top_idx = np.argpartition(-lp_t, top_k - 1)[:top_k]

        for prefix, (p_b, p_nb) in beams.items():
            for s in top_idx:
                s = int(s)
                p_s = float(lp_t[s])

                if s == blank:
                    # Extending with blank: any beam state can absorb a blank,
                    # and the prefix is unchanged.
                    nb, nnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    new_pb = _logsumexp(nb, _logsumexp(p_b + p_s, p_nb + p_s))
                    next_beams[prefix] = (new_pb, nnb)
                    continue

                last = prefix[-1] if prefix else None
                if s == last:
                    # Repeat character: if previous state was blank we *extend*
                    # the prefix; if previous state was non-blank we *collapse*
                    # (the prefix is unchanged).
                    new_prefix = prefix + (s,)
                    nb, nnb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    new_pnb = _logsumexp(nnb, p_b + p_s)
                    next_beams[new_prefix] = (nb, new_pnb)

                    nb, nnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    new_pnb_same = _logsumexp(nnb, p_nb + p_s)
                    next_beams[prefix] = (nb, new_pnb_same)
                else:
                    # Different character: always extends the prefix.
                    new_prefix = prefix + (s,)
                    nb, nnb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    new_pnb = _logsumexp(nnb, _logsumexp(p_b + p_s, p_nb + p_s))
                    next_beams[new_prefix] = (nb, new_pnb)

        # Prune to top beam_width by total log-prob.
        scored = [
            (prefix, p_b, p_nb, _logsumexp(p_b, p_nb))
            for prefix, (p_b, p_nb) in next_beams.items()
        ]
        scored.sort(key=lambda x: x[3], reverse=True)
        beams = {prefix: (p_b, p_nb) for prefix, p_b, p_nb, _ in scored[:beam_width]}

    # Pick the highest-scoring beam.
    best_prefix = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return "".join(idx_to_char[i] for i in best_prefix if i in idx_to_char)


class CTCClassifier:
    def __init__(self, beam_width: int = DEFAULT_BEAM_WIDTH):
        if not _ONNX.exists():
            raise FileNotFoundError(f"CTC ONNX not found at {_ONNX}.")
        with open(_META) as f:
            meta = json.load(f)
        self.idx_to_char: dict[int, str] = {int(k): v for k, v in meta["idx_to_char"].items()}
        self.blank_idx: int = int(meta["blank_idx"])
        self.session = ort.InferenceSession(str(_ONNX), providers=["CPUExecutionProvider"])
        self.input_x = "landmarks"
        self.input_mask = "pad_mask"
        self.beam_width = beam_width

    def transcribe(
        self,
        features: np.ndarray,
        missing: np.ndarray,
        beam: bool = True,
    ) -> str:
        """features: (T, N_FEATURES) raw (un-normalised, zero-filled missing) frames.
        missing: (T, N_LANDMARKS) bool — True where landmark was absent.
        beam: if True (default), use beam-search decoding; else greedy."""
        if features.shape[0] == 0:
            return ""
        if features.shape[0] > MAX_FRAMES:
            keep = np.linspace(0, features.shape[0] - 1, MAX_FRAMES).astype(int)
            features = features[keep]
            missing = missing[keep]

        x = normalize_sequence(features, missing)
        T = x.shape[0]

        # The dynamo-exported ONNX requires batch >= 2 to keep the batch axis
        # dynamic; pad with an all-zero, fully-masked second item we ignore.
        x_batch = np.stack([x, np.zeros_like(x)], axis=0).astype(np.float32)
        mask = np.zeros((2, T), dtype=bool)
        mask[1, :] = True

        log_probs = self.session.run(
            None,
            {self.input_x: x_batch, self.input_mask: mask},
        )[0]
        # (T, B=2, V) — take first batch element
        lp = log_probs[:, 0, :]
        if beam and self.beam_width > 1:
            return _beam_search_decode(lp, self.blank_idx, self.idx_to_char, self.beam_width)
        return _greedy_decode(lp, self.blank_idx, self.idx_to_char)
