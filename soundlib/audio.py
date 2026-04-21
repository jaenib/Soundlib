"""Audio loading. Uses soundfile for common formats and falls back to ffmpeg
for anything it can't decode (mp3 variants, m4a/aac, opus, flac edge cases, etc.)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".oga", ".opus",
    ".mp3", ".m4a", ".aac", ".aif", ".aiff",
    ".wv", ".ape", ".wma",
}


class AudioLoadError(RuntimeError):
    pass


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    """Load an audio file as mono float32 at the requested sample rate."""
    path = Path(path)
    try:
        import soundfile as sf
        data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = data.mean(axis=1)
        if file_sr != sr:
            audio = _resample(audio, file_sr, sr)
        return audio
    except Exception as sf_err:
        if shutil.which("ffmpeg") is None:
            raise AudioLoadError(
                f"soundfile failed for {path} ({sf_err}) and ffmpeg is not on PATH"
            ) from sf_err
        return _ffmpeg_decode(path, sr)


def _ffmpeg_decode(path: Path, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioLoadError(
            f"ffmpeg failed on {path}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    # Prefer soxr > scipy > linear interpolation, in that order.
    try:
        import soxr
        return soxr.resample(audio, src_sr, dst_sr).astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        pass
    n_out = int(round(len(audio) * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
