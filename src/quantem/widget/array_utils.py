"""
Array utilities for handling NumPy, CuPy, and PyTorch arrays uniformly.

This module provides utilities to convert arrays from different backends
into NumPy arrays for widget processing.
"""

from typing import Any, Literal, NamedTuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


ArrayBackend = Literal["numpy", "cupy", "torch", "unknown"]


def get_array_backend(data: Any) -> ArrayBackend:
    """
    Detect the array backend of the input data.

    Parameters
    ----------
    data : array-like
        Input array (NumPy, CuPy, PyTorch, or other).

    Returns
    -------
    str
        One of: "numpy", "cupy", "torch", "unknown"
    """
    # Check PyTorch first (has both .numpy and .detach methods)
    if hasattr(data, "detach") and hasattr(data, "numpy"):
        return "torch"
    # Check CuPy (has .get() or __cuda_array_interface__)
    if hasattr(data, "__cuda_array_interface__"):
        return "cupy"
    if hasattr(data, "get") and hasattr(data, "__array__"):
        # CuPy arrays have .get() to transfer to CPU
        type_name = type(data).__module__
        if "cupy" in type_name:
            return "cupy"
    # Check NumPy
    if isinstance(data, np.ndarray):
        return "numpy"
    return "unknown"


def to_numpy(data: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Convert any array-like (NumPy, CuPy, PyTorch) to a NumPy array.

    Parameters
    ----------
    data : array-like
        Input array from any supported backend.
    dtype : np.dtype, optional
        Target dtype for the output array. If None, preserves original dtype.

    Returns
    -------
    np.ndarray
        NumPy array with the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.array_utils import to_numpy
    >>>
    >>> # NumPy passthrough
    >>> arr = np.random.rand(10, 10)
    >>> result = to_numpy(arr)
    >>>
    >>> # CuPy conversion (if available)
    >>> import cupy as cp
    >>> gpu_arr = cp.random.rand(10, 10)
    >>> cpu_arr = to_numpy(gpu_arr)
    >>>
    >>> # PyTorch conversion (if available)
    >>> import torch
    >>> tensor = torch.rand(10, 10)
    >>> arr = to_numpy(tensor)
    """
    backend = get_array_backend(data)

    if backend == "torch":
        # PyTorch tensor: detach from graph, move to CPU, convert to numpy
        result = data.detach().cpu().numpy()

    elif backend == "cupy":
        # CuPy array: use .get() to transfer to CPU
        if hasattr(data, "get"):
            result = data.get()
        else:
            # Fallback for __cuda_array_interface__
            import cupy as cp

            result = cp.asnumpy(data)

    elif backend == "numpy":
        # NumPy array: passthrough (may copy if dtype changes)
        result = data

    else:
        # Unknown backend: try np.asarray as fallback
        result = np.asarray(data)

    # Apply dtype conversion if specified
    if dtype is not None:
        result = np.asarray(result, dtype=dtype)

    return result


def bin2d(data, factor: int = 2, mode: str = "mean", edge_mode: str = "crop") -> np.ndarray:
    """
    Spatial binning for 2D or 3D arrays.

    Uses torch GPU (MPS/CUDA) when available for large arrays (~5× faster on 4K data).

    Parameters
    ----------
    data : array-like
        Input array with shape ``(H, W)`` or ``(N, H, W)``.
    factor : int, default 2
        Bin factor.
    mode : str, default "mean"
        Reduction mode: ``"mean"`` or ``"sum"``.
    edge_mode : str, default "crop"
        How to handle dimensions not divisible by *factor*:
        ``"crop"`` trims extra pixels, ``"pad"`` zero-pads to the next
        multiple (output shape uses ``ceil(dim / factor)``).

    Returns
    -------
    np.ndarray
        Binned array, dtype float32.
    """
    arr = to_numpy(data)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    # Torch GPU fast path: only for arrays between 1M and 500M elements.
    # Larger arrays hit MPS memory transfer bottleneck (>2 GB transfer > CPU compute).
    import torch
    if 1_000_000 < arr.size < 500_000_000 and (torch.backends.mps.is_available() or torch.cuda.is_available()):
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        t = torch.from_numpy(arr).to(dev)
        if t.ndim == 2:
            h, w = t.shape
            oh = h // factor * factor
            ow = w // factor * factor
            t = t[:oh, :ow].reshape(oh // factor, factor, ow // factor, factor)
            t = t.sum(dim=(1, 3)) if mode == "sum" else t.mean(dim=(1, 3))
        elif t.ndim == 3:
            n, h, w = t.shape
            oh = h // factor * factor
            ow = w // factor * factor
            t = t[:, :oh, :ow].reshape(n, oh // factor, factor, ow // factor, factor)
            t = t.sum(dim=(2, 4)) if mode == "sum" else t.mean(dim=(2, 4))
        return t.cpu().numpy().astype(np.float32)

    # CPU fallback (no GPU available or small array)
    reduce = np.ndarray.sum if mode == "sum" else np.ndarray.mean
    if arr.ndim == 2:
        arr = _pad_or_crop_2d(arr, factor, edge_mode)
        h, w = arr.shape
        oh, ow = h // factor, w // factor
        return reduce(arr.reshape(oh, factor, ow, factor), axis=(1, 3)).astype(np.float32)
    # 3D: (N, H, W)
    arr = _pad_or_crop_3d(arr, factor, edge_mode)
    n, h, w = arr.shape
    oh, ow = h // factor, w // factor
    return reduce(arr.reshape(n, oh, factor, ow, factor), axis=(2, 4)).astype(np.float32)


def _pad_or_crop_2d(arr: np.ndarray, factor: int, edge_mode: str) -> np.ndarray:
    h, w = arr.shape
    if edge_mode == "pad":
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
    else:
        oh, ow = h // factor, w // factor
        arr = arr[:oh * factor, :ow * factor]
    return arr


def _pad_or_crop_3d(arr: np.ndarray, factor: int, edge_mode: str) -> np.ndarray:
    _, h, w = arr.shape
    if edge_mode == "pad":
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    else:
        oh, ow = h // factor, w // factor
        arr = arr[:, :oh * factor, :ow * factor]
    return arr


def apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """
    Apply sub-pixel shift using bilinear interpolation.

    Uses ``torch.nn.functional.grid_sample`` on GPU when torch is available,
    falls back to numpy bilinear interpolation otherwise.

    Parameters
    ----------
    img : np.ndarray
        2D image, float32.
    dy : float
        Shift in y (rows).
    dx : float
        Shift in x (columns).

    Returns
    -------
    np.ndarray
        Shifted image, same shape, float32. Out-of-bounds pixels are zero.
    """
    if _HAS_TORCH:
        h, w = img.shape
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        t = torch.as_tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        base_y = torch.linspace(-1, 1, h, device=device)
        base_x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(base_y, base_x, indexing="ij")
        grid = torch.stack([gx - dx * 2.0 / w, gy - dy * 2.0 / h], dim=-1).unsqueeze(0)
        result = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return result.squeeze().cpu().numpy()
    h, w = img.shape
    y_src = np.arange(h, dtype=np.float64) - dy
    x_src = np.arange(w, dtype=np.float64) - dx
    yy, xx = np.meshgrid(y_src, x_src, indexing="ij")
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    fy = (yy - y0).astype(np.float32)
    fx = (xx - x0).astype(np.float32)
    valid = (y0 >= 0) & (y0 + 1 < h) & (x0 >= 0) & (x0 + 1 < w)
    y0c = np.clip(y0, 0, h - 2)
    x0c = np.clip(x0, 0, w - 2)
    result = (img[y0c, x0c] * (1 - fy) * (1 - fx)
              + img[y0c, x0c + 1] * (1 - fy) * fx
              + img[y0c + 1, x0c] * fy * (1 - fx)
              + img[y0c + 1, x0c + 1] * fy * fx)
    result[~valid] = 0.0
    return result.astype(np.float32)


def _resize_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image using bilinear interpolation (pure numpy, no scipy)."""
    h, w = img.shape

    if h == target_h and w == target_w:
        return img

    y_new = np.linspace(0, h - 1, target_h)
    x_new = np.linspace(0, w - 1, target_w)
    x_grid, y_grid = np.meshgrid(x_new, y_new)

    y0 = np.floor(y_grid).astype(int)
    x0 = np.floor(x_grid).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    fy = y_grid - y0
    fx = x_grid - x0

    result = (
        img[y0, x0] * (1 - fy) * (1 - fx) +
        img[y0, x1] * (1 - fy) * fx +
        img[y1, x0] * fy * (1 - fx) +
        img[y1, x1] * fy * fx
    )
    return result.astype(img.dtype)


class DatasetMeta(NamedTuple):
    """Metadata duck-typed from a widget data argument.

    Attributes
    ----------
    array : Any
        The raw array payload, with the container (``IOResult`` /
        quantem ``Dataset``) unwrapped. For plain arrays this is the input
        unchanged. NOT converted to numpy — call ``to_numpy()`` afterwards.
    title : str or None
        Title/name extracted from the container, else ``None``.
    pixel_size : float or None
        Pixel size in Å. ``None`` when no calibration was found. nm values
        are converted to Å (``value * 10``) when ``nm_to_angstrom`` is set.
    units : str or None
        Raw unit string for the chosen ``sampling_axis``, else ``None``.
    labels : list of str or None
        Per-frame labels from an ``IOResult``, else ``None``.
    """

    array: Any
    title: str | None
    pixel_size: float | None
    units: str | None
    labels: list[str] | None


def extract_dataset_meta(
    data: Any,
    *,
    sampling_axis: int = -1,
    angstrom_units: tuple[str, ...] = ("Å", "angstrom", "A"),
    nm_units: tuple[str, ...] = ("nm",),
    nm_to_angstrom: bool = True,
) -> DatasetMeta:
    """Duck-type an ``IOResult`` or quantem ``Dataset`` and extract metadata.

    Handles three input kinds and unwraps them to the underlying array:

    1. ``IOResult`` (``isinstance`` check) — has ``.data``, ``.title``,
       ``.pixel_size``, ``.labels``. ``pixel_size`` is already in Å, used
       verbatim (no unit conversion).
    2. quantem ``Dataset`` — duck-typed via ``hasattr(.array)`` +
       ``hasattr(.name)`` + ``hasattr(.sampling)``. ``pixel_size`` is derived
       from ``sampling[sampling_axis]`` and converted nm→Å when the matching
       entry of ``.units`` is an nm unit.
    3. Plain array (numpy / torch / cupy / anything else) — returned as-is
       with all metadata fields ``None``.

    This intentionally returns BOTH ``IOResult`` pixel_size and ``Dataset``
    pixel_size in the single ``pixel_size`` field; only one path runs because
    a value is never both an ``IOResult`` and a ``Dataset``.

    Parameters
    ----------
    data : Any
        Widget data argument.
    sampling_axis : int, default -1
        Index into ``Dataset.sampling`` / ``Dataset.units`` used for the
        pixel size. Use ``-1`` for 2D viewers (Show2D, Show3DVolume, Edit2D,
        Mark2D, Align2D, ShowComplex2D), ``1`` for Show3D, ``0`` for the
        nav axis of 4D datasets. Ignored for ``IOResult`` and plain arrays.
    angstrom_units : tuple of str, default ("Å", "angstrom", "A")
        Unit strings treated as angstroms (pixel size used verbatim).
    nm_units : tuple of str, default ("nm",)
        Unit strings treated as nanometers. Pass ``("nm", "nanometer")`` to
        match the Show3D / Align2D call sites.
    nm_to_angstrom : bool, default True
        When True and units are nm, multiply the sampling value by 10.
        When False (e.g. a caller that wants raw nm), the value is kept.

    Returns
    -------
    DatasetMeta
        ``(array, title, pixel_size, units, labels)``. ``array`` is the
        unwrapped payload; the caller still applies ``to_numpy()`` and any
        precedence logic (e.g. only override when ``title`` was not set).

    Notes
    -----
    For a ``Dataset`` whose unit is neither an angstrom nor an nm unit
    (e.g. ``"mrad"`` or ``"pixels"``), ``pixel_size`` is ``None`` — matching
    the existing ``elif`` chains that leave pixel size untouched. Callers
    that need mrad handling (Show4D, Show4DSTEM) must handle that axis
    separately and should not rely solely on this helper.
    """
    # IOResult: lazy import to avoid a circular import at module load.
    from quantem.widget.io import IOResult

    if isinstance(data, IOResult):
        return DatasetMeta(
            array=data.data,
            title=data.title or None,
            pixel_size=data.pixel_size,
            units=data.units,
            labels=list(data.labels) if data.labels else None,
        )

    # quantem Dataset duck typing.
    if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
        title = data.name if data.name else None
        pixel_size: float | None = None
        unit_str: str | None = None
        if hasattr(data, "units"):
            units = list(data.units)
            sampling_val = float(data.sampling[sampling_axis])
            unit_str = units[sampling_axis]
            if unit_str in nm_units:
                pixel_size = sampling_val * 10 if nm_to_angstrom else sampling_val
            elif unit_str in angstrom_units:
                pixel_size = sampling_val
        return DatasetMeta(
            array=data.array,
            title=title,
            pixel_size=pixel_size,
            units=unit_str,
            labels=None,
        )

    # Plain array (numpy / torch / cupy / unknown) — no metadata.
    return DatasetMeta(array=data, title=None, pixel_size=None, units=None, labels=None)


def normalize_frame(
    frame: np.ndarray,
    *,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    auto_contrast: bool = False,
    plo: float = 2.0,
    phi: float = 98.0,
) -> np.ndarray:
    """Normalize a 2D frame to a ``uint8`` array for colormap display.

    Ports the per-widget ``_normalize_frame`` logic verbatim. The contrast
    range is chosen by this precedence (matching every existing call site):

    1. **Manual** — both ``vmin`` and ``vmax`` are not ``None``: use them
       directly. When ``log_scale`` is on, the bounds are themselves passed
       through ``log1p(max(v, 0))``.
    2. **Auto-contrast** — ``auto_contrast`` is True: bounds are the
       ``plo`` / ``phi`` percentiles of the (already log-transformed) frame.
    3. **Min/max** — fall back to ``frame.min()`` / ``frame.max()`` of the
       (already log-transformed) frame.

    The frame itself is log-transformed first (``log1p(maximum(frame, 0))``)
    when ``log_scale`` is set, before any percentile/min-max is computed.

    Output: ``clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)`` cast to
    ``uint8``. When ``vmax <= vmin`` the result is an all-zero ``uint8``
    array of the same shape.

    Parameters
    ----------
    frame : np.ndarray
        2D raw float frame.
    log_scale : bool, default False
        Apply ``log1p`` to the frame (and to manual bounds) first.
    vmin, vmax : float or None
        Manual contrast bounds. BOTH must be non-``None`` to take effect;
        if either is ``None`` the manual path is skipped.
    auto_contrast : bool, default False
        Use percentile clipping. Ignored when manual bounds are supplied.
    plo, phi : float, default 2.0 / 98.0
        Low/high percentiles for auto-contrast. Show2D / Mark2D use the
        defaults (2/98); Show3D / Show4D / ShowComplex2D pass their own
        ``percentile_low`` / ``percentile_high`` traits.

    Returns
    -------
    np.ndarray
        ``uint8`` array, same shape as ``frame``.

    Notes
    -----
    This does NOT reproduce Show3D's min/max branch, which uses the
    precomputed ``self._vmin`` / ``self._vmax`` instead of
    ``frame.min()`` / ``frame.max()``. Show3D callers must pass those
    precomputed values explicitly as ``vmin`` / ``vmax`` (see the rewiring
    spec) so the manual path is taken.
    """
    if log_scale:
        frame = np.log1p(np.maximum(frame, 0))
    if vmin is not None and vmax is not None:
        lo = float(vmin)
        hi = float(vmax)
        if log_scale:
            lo = float(np.log1p(max(lo, 0)))
            hi = float(np.log1p(max(hi, 0)))
    elif auto_contrast:
        lo = float(np.percentile(frame, plo))
        hi = float(np.percentile(frame, phi))
    else:
        lo = float(frame.min())
        hi = float(frame.max())
    if hi > lo:
        return np.clip((frame - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return np.zeros(frame.shape, dtype=np.uint8)


def compute_stats(arr: Any) -> dict[str, float]:
    """Compute basic display statistics for an array.

    Parameters
    ----------
    arr : array-like
        Numpy array, torch tensor, or anything with ``.mean()`` / ``.min()``
        / ``.max()`` / ``.std()`` reductions. Torch tensors are reduced on
        their current device and the scalar pulled via ``.item()``.

    Returns
    -------
    dict
        ``{"mean": float, "min": float, "max": float, "std": float}`` — in
        that key order. For the list-style stats traits (Show4D
        ``nav_stats`` / ``sig_stats``), use ``list(compute_stats(x).values())``
        which yields ``[mean, min, max, std]``.
    """
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        return {
            "mean": float(arr.mean().item()),
            "min": float(arr.min().item()),
            "max": float(arr.max().item()),
            "std": float(arr.std().item()),
        }
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }
