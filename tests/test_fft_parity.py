"""JS computation validation — CPU FFT in js/webgpu-fft.ts.

Per CLAUDE.md "JS computation validation": math implemented in JavaScript must be
ported to Python LINE-BY-LINE and validated against NumPy ground truth so the test
catches any JS-side bug. This file ports the CPU-fallback FFT routines from
`js/webgpu-fft.ts` (`fft1d`, `fft2d`, `fftshift`, `computeMagnitude`) and asserts
parity with `numpy.fft`.

The JS `fft2d` zero-pads to next-power-of-two and crops back. For power-of-two
dimensions it must match `np.fft.fft2` exactly. For NON-power-of-two dimensions the
cropped result is NOT a true DFT of the input — see `test_fft2d_nonpow2_*` below,
which documents the actual (mismatching) behavior.
"""

import numpy as np
import pytest

# ── Line-by-line Python ports of js/webgpu-fft.ts ──────────────────────────


def _next_pow2(n):
    """Port of nextPow2 (js/webgpu-fft.ts line 12):
    return Math.pow(2, Math.ceil(Math.log2(n)));"""
    return int(2 ** np.ceil(np.log2(n)))


def _js_fft1d(real, imag, inverse=False):
    """Exact Python port of fft1d (js/webgpu-fft.ts lines 14-43).
    In-place radix-2 iterative Cooley-Tukey on Python lists `real`/`imag`.
    `n` MUST be a power of two (the JS callers guarantee this via padding)."""
    n = len(real)
    if n <= 1:
        return
    # bit-reversal permutation
    j = 0
    for i in range(0, n - 1):
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
        k = n >> 1
        while k <= j:
            j -= k
            k >>= 1
        j += k
    sign = 1 if inverse else -1
    length = 2
    while length <= n:
        half_len = length >> 1
        angle = (sign * 2 * np.pi) / length
        w_real = np.cos(angle)
        w_imag = np.sin(angle)
        i = 0
        while i < n:
            cur_real = 1.0
            cur_imag = 0.0
            for k in range(half_len):
                even_idx = i + k
                odd_idx = i + k + half_len
                t_real = cur_real * real[odd_idx] - cur_imag * imag[odd_idx]
                t_imag = cur_real * imag[odd_idx] + cur_imag * real[odd_idx]
                real[odd_idx] = real[even_idx] - t_real
                imag[odd_idx] = imag[even_idx] - t_imag
                real[even_idx] += t_real
                imag[even_idx] += t_imag
                new_real = cur_real * w_real - cur_imag * w_imag
                cur_imag = cur_real * w_imag + cur_imag * w_real
                cur_real = new_real
            i += length
        length <<= 1
    if inverse:
        for i in range(n):
            real[i] /= n
            imag[i] /= n


def _js_fft2d(real, imag, width, height, inverse=False):
    """Exact Python port of fft2d (js/webgpu-fft.ts lines 45-73).
    `real`/`imag` are flat row-major lists of length width*height. Modifies them
    in place (only the original width*height region when padding is needed)."""
    padded_w = _next_pow2(width)
    padded_h = _next_pow2(height)
    needs_padding = padded_w != width or padded_h != height
    if needs_padding:
        work_real = [0.0] * (padded_w * padded_h)
        work_imag = [0.0] * (padded_w * padded_h)
        for y in range(height):
            for x in range(width):
                work_real[y * padded_w + x] = real[y * width + x]
                work_imag[y * padded_w + x] = imag[y * width + x]
    else:
        work_real = real
        work_imag = imag
    # rows
    for y in range(padded_h):
        offset = y * padded_w
        row_real = [work_real[offset + x] for x in range(padded_w)]
        row_imag = [work_imag[offset + x] for x in range(padded_w)]
        _js_fft1d(row_real, row_imag, inverse)
        for x in range(padded_w):
            work_real[offset + x] = row_real[x]
            work_imag[offset + x] = row_imag[x]
    # columns
    for x in range(padded_w):
        col_real = [work_real[y * padded_w + x] for y in range(padded_h)]
        col_imag = [work_imag[y * padded_w + x] for y in range(padded_h)]
        _js_fft1d(col_real, col_imag, inverse)
        for y in range(padded_h):
            work_real[y * padded_w + x] = col_real[y]
            work_imag[y * padded_w + x] = col_imag[y]
    if needs_padding:
        for y in range(height):
            for x in range(width):
                real[y * width + x] = work_real[y * padded_w + x]
                imag[y * width + x] = work_imag[y * padded_w + x]
    # When no padding is needed, work_real/work_imag ARE real/imag (already updated).


def _js_fftshift(data, width, height):
    """Exact Python port of fftshift (js/webgpu-fft.ts lines 75-82).
    Modifies the flat list `data` in place."""
    half_w = width >> 1
    half_h = height >> 1
    temp = [0.0] * (width * height)
    for y in range(height):
        for x in range(width):
            temp[((y + half_h) % height) * width + ((x + half_w) % width)] = data[y * width + x]
    for i in range(len(data)):
        data[i] = temp[i]


def _js_compute_magnitude(real, imag):
    """Exact Python port of computeMagnitude (js/webgpu-fft.ts lines 424-430)."""
    return [np.sqrt(real[i] * real[i] + imag[i] * imag[i]) for i in range(len(real))]


# ── Helpers to bridge flat JS lists <-> numpy 2D arrays ────────────────────


def _js_fft2d_on_array(arr):
    """Run the JS fft2d port on a 2D numpy array, return (real2d, imag2d)."""
    height, width = arr.shape
    real = arr.astype(np.float64).ravel().tolist()
    imag = [0.0] * (width * height)
    _js_fft2d(real, imag, width, height, inverse=False)
    real2d = np.array(real).reshape(height, width)
    imag2d = np.array(imag).reshape(height, width)
    return real2d, imag2d


# ── Tests: 1D FFT ──────────────────────────────────────────────────────────


def test_fft1d_matches_numpy_pow2():
    """JS fft1d must match np.fft.fft for power-of-two lengths."""
    rng = np.random.default_rng(0)
    for n in [2, 4, 8, 16, 32, 64]:
        signal = rng.standard_normal(n)
        real = signal.tolist()
        imag = [0.0] * n
        _js_fft1d(real, imag, inverse=False)
        js = np.array(real) + 1j * np.array(imag)
        np.testing.assert_allclose(
            js, np.fft.fft(signal), atol=1e-9, rtol=1e-9,
            err_msg=f"JS fft1d != np.fft.fft at N={n}",
        )


def test_fft1d_inverse_roundtrip():
    """JS fft1d forward then inverse recovers the input (the JS inverse divides by n)."""
    rng = np.random.default_rng(1)
    n = 32
    signal = rng.standard_normal(n)
    real = signal.tolist()
    imag = [0.0] * n
    _js_fft1d(real, imag, inverse=False)
    _js_fft1d(real, imag, inverse=True)
    np.testing.assert_allclose(real, signal, atol=1e-9)
    np.testing.assert_allclose(imag, np.zeros(n), atol=1e-9)


# ── Tests: 2D FFT, power-of-two (must match np.fft.fft2 exactly) ───────────


@pytest.mark.parametrize("h,w", [(2, 2), (4, 4), (8, 8), (16, 16), (8, 16), (32, 8)])
def test_fft2d_pow2_matches_numpy(h, w):
    """JS fft2d on power-of-two dims must match np.fft.fft2 exactly (no padding)."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((h, w))
    real2d, imag2d = _js_fft2d_on_array(arr)
    js = real2d + 1j * imag2d
    np.testing.assert_allclose(
        js, np.fft.fft2(arr), atol=1e-8, rtol=1e-8,
        err_msg=f"JS fft2d != np.fft.fft2 at {h}x{w}",
    )


def test_fft2d_pow2_magnitude_matches_numpy():
    """computeMagnitude(JS fft2d) must match |np.fft.fft2| for power-of-two dims."""
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((16, 16))
    real2d, imag2d = _js_fft2d_on_array(arr)
    mag = np.array(
        _js_compute_magnitude(real2d.ravel().tolist(), imag2d.ravel().tolist())
    ).reshape(16, 16)
    np.testing.assert_allclose(mag, np.abs(np.fft.fft2(arr)), atol=1e-8, rtol=1e-8)


# ── Tests: fftshift parity ─────────────────────────────────────────────────


@pytest.mark.parametrize("h,w", [(8, 8), (7, 9), (16, 4), (5, 5), (1, 8)])
def test_fftshift_matches_numpy(h, w):
    """JS fftshift must match np.fft.fftshift for even AND odd dimensions.

    np.fft.fftshift rolls each axis by floor(dim/2). The JS code uses
    `width >> 1` / `height >> 1`, which is also floor(dim/2) — so they agree
    even for odd sizes.
    """
    rng = np.random.default_rng(99)
    arr = rng.standard_normal((h, w))
    data = arr.ravel().tolist()
    _js_fftshift(data, w, h)
    js_shifted = np.array(data).reshape(h, w)
    np.testing.assert_array_equal(js_shifted, np.fft.fftshift(arr))


# ── Tests: 2D FFT, NON-power-of-two (documents ACTUAL behavior) ────────────


@pytest.mark.parametrize("h,w", [(5, 5), (6, 10), (15, 20), (3, 7)])
def test_fft2d_nonpow2_does_not_match_numpy(h, w):
    """DOCUMENTED BEHAVIOR / FLAG: for non-power-of-two dims, JS fft2d zero-pads
    to next-pow2, runs the FFT on the padded array, then CROPS the top-left
    width*height region back out.

    The cropped sub-block of FFT(zero-padded array) is NOT equal to FFT(original
    array): zero-padding changes the frequency grid (it interpolates the spectrum
    onto a finer grid), and cropping a corner of that finer-grid spectrum gives
    neither the original DFT nor a meaningful resampling of it.

    This test asserts the ACTUAL (mismatching) behavior so the parity suite
    stays honest. The DC term (index [0,0]) does still match, because
    FFT[0,0] = sum of all samples and zero-padding adds only zeros.

    >>> PARITY FLAG: callers of fft2d MUST pre-pad non-power-of-two data to a
    >>> power of two themselves (CLAUDE.md "ROI FFT" section already mandates
    >>> `nextPow2` pre-padding for ROI crops). Passing raw non-pow2 dimensions
    >>> straight into fft2d yields a spectrum that does not correspond to the
    >>> input — it is not a usable DFT.
    """
    rng = np.random.default_rng(123)
    arr = rng.standard_normal((h, w))
    real2d, imag2d = _js_fft2d_on_array(arr)
    js = real2d + 1j * imag2d
    expected = np.fft.fft2(arr)

    # DC component still matches (sum of all samples; padding adds only zeros).
    np.testing.assert_allclose(js[0, 0], expected[0, 0], atol=1e-8)

    # But the full spectrum does NOT match — confirm the divergence is real,
    # not numerical noise.
    max_abs_diff = np.max(np.abs(js - expected))
    assert max_abs_diff > 1e-3, (
        f"Expected JS fft2d to DIVERGE from np.fft.fft2 for non-pow2 {h}x{w}, "
        f"but max diff was only {max_abs_diff:.2e}. If this fails, fft2d's "
        f"behavior changed — re-port and re-document."
    )


def test_fft2d_nonpow2_prepadded_matches_numpy():
    """The CORRECT usage (matching CLAUDE.md ROI-FFT guidance): pre-pad the input
    to a power of two BEFORE calling fft2d. Then JS fft2d matches np.fft.fft2 of
    the SAME zero-padded array exactly, because no internal padding is triggered.
    """
    rng = np.random.default_rng(456)
    crop_h, crop_w = 15, 20
    crop = rng.standard_normal((crop_h, crop_w))
    pad_h, pad_w = _next_pow2(crop_h), _next_pow2(crop_w)
    padded = np.zeros((pad_h, pad_w))
    padded[:crop_h, :crop_w] = crop
    real2d, imag2d = _js_fft2d_on_array(padded)
    js = real2d + 1j * imag2d
    np.testing.assert_allclose(
        js, np.fft.fft2(padded), atol=1e-8, rtol=1e-8,
        err_msg="JS fft2d on pre-padded pow2 input must match np.fft.fft2",
    )
