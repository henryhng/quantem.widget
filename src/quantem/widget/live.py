"""
live: Real-time image viewer for microscope acquisition.

Designed for incremental streaming — push() sends one frame at a time,
not all previous frames. Supports 2D (latest + filmstrip) and 3D
(stack with playback) modes, togglable at runtime.
"""

import pathlib
import threading
import time as _time_module
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy, _resize_image
from quantem.widget.io import IO, IOResult
from quantem.widget.json_state import resolve_widget_version


class Live(anywidget.AnyWidget):
    """Real-time image viewer for live microscope acquisition.

    Unlike Show2D/Show3D which send all data on every update, Live uses
    incremental push — each new image is sent once, O(1) per frame.

    Parameters
    ----------
    title : str, default "Live"
        Title displayed in the header.
    mode : str, default "2d"
        Display mode: ``"2d"`` (latest image + filmstrip) or
        ``"3d"`` (stack with frame slider).
    cmap : str, default "inferno"
        Colormap name.
    log_scale : bool, default False
        Apply log scale to intensity.
    buffer_size : int, default 50
        Maximum number of frames kept in JS memory. Older frames
        are dropped from the display but still counted.

    Examples
    --------
    >>> from quantem.widget import Live
    >>>
    >>> w = Live(title="SNSF Session")
    >>> w.push(image_array)
    >>> w.push(IO.file("capture.emd"))
    >>>
    >>> # Or watch a folder
    >>> w = Live.watch("./data/", pattern="*.emd")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "live.js"
    _css = pathlib.Path(__file__).parent / "static" / "live.css"

    # =========================================================================
    # Core State
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("Live").tag(sync=True)
    mode = traitlets.Unicode("2d").tag(sync=True)  # "2d" or "3d"
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    buffer_size = traitlets.Int(50).tag(sync=True)
    n_frames = traitlets.Int(0).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Push mechanism — single frame (for live streaming with natural delay)
    # =========================================================================
    _push_bytes = traitlets.Bytes(b"").tag(sync=True)
    _push_height = traitlets.Int(0).tag(sync=True)
    _push_width = traitlets.Int(0).tag(sync=True)
    _push_label = traitlets.Unicode("").tag(sync=True)
    _push_counter = traitlets.Int(0).tag(sync=True)

    # Batch mechanism — multiple frames at once (for bulk loading)
    # =========================================================================
    _batch_bytes = traitlets.Bytes(b"").tag(sync=True)
    _batch_dims = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
    _batch_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    _batch_counter = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # 3D mode — current frame index for playback
    # =========================================================================
    frame_idx = traitlets.Int(0).tag(sync=True)
    playing = traitlets.Bool(False).tag(sync=True)
    fps = traitlets.Float(5.0).tag(sync=True)

    def __init__(
        self,
        data=None,
        *,
        title: str = "Live",
        mode: str = "2d",
        cmap: str = "inferno",
        log_scale: bool = False,
        auto_contrast: bool = False,
        buffer_size: int = 50,
        fps: float = 5.0,
        show_controls: bool = True,
        pixel_size: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.title = title
        self.mode = mode
        self.cmap = cmap
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.buffer_size = buffer_size
        self.fps = fps
        self.pixel_size = pixel_size
        self.show_controls = show_controls

        # Internal state
        self._labels: list[str] = []
        self._watcher = None
        self._pending_batch: list[tuple[np.ndarray, str]] = []

        # Push initial data if provided
        if data is not None:
            if isinstance(data, IOResult):
                if not title or title == "Live":
                    self.title = data.title or "Live"
                if isinstance(data.data, np.ndarray) and data.data.ndim == 3:
                    for i in range(data.data.shape[0]):
                        label = data.labels[i] if i < len(data.labels) else f"Frame {i}"
                        self._queue(data.data[i], label)
                else:
                    self._queue(data.data, data.title or "Image 0")
            elif isinstance(data, (list, tuple)):
                for i, img in enumerate(data):
                    self._queue(to_numpy(img), f"Image {i}")
            elif isinstance(data, np.ndarray):
                if data.ndim == 3:
                    for i in range(data.shape[0]):
                        self._queue(data[i], f"Frame {i}")
                elif data.ndim == 2:
                    self._queue(data, "Image 0")
            self._flush_batch()

    def _queue(self, arr: np.ndarray, label: str) -> None:
        """Queue a frame for batch sending (internal)."""
        arr = to_numpy(arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            return
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._pending_batch.append((arr, label))
        self._labels.append(label)

    def _flush_batch(self) -> None:
        """Send all queued frames to JS in one batch."""
        if not self._pending_batch:
            return
        all_bytes = b"".join(arr.tobytes() for arr, _ in self._pending_batch)
        dims = [[int(arr.shape[0]), int(arr.shape[1])] for arr, _ in self._pending_batch]
        labels = [label for _, label in self._pending_batch]
        count = len(self._pending_batch)
        self._pending_batch = []
        with self.hold_sync():
            self._batch_bytes = all_bytes
            self._batch_dims = dims
            self._batch_labels = labels
            self._batch_counter = self._batch_counter + 1
            self.n_frames = self.n_frames + count

    def push(self, data, *, label: str | None = None) -> Self:
        """Push a new image to the live view.

        Only the new frame is sent to JS — O(1), not O(N).

        Parameters
        ----------
        data : array_like or IOResult
            2D image array, or an IOResult from IO.file().
        label : str, optional
            Label for this frame. Defaults to "Frame N".

        Returns
        -------
        Self
        """
        if isinstance(data, IOResult):
            if label is None:
                label = data.title
            if self.pixel_size == 0.0 and data.pixel_size:
                self.pixel_size = data.pixel_size
            data = data.data

        arr = to_numpy(data)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image, got {arr.ndim}D array with shape {arr.shape}")

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.ascontiguousarray(arr)

        # Auto-bin large frames (>32 MB) for fast trait sync
        frame_mb = arr.nbytes / (1024 * 1024)
        if frame_mb > 32:
            from quantem.widget.array_utils import bin2d
            for bf in [2, 4, 8]:
                if frame_mb / (bf * bf) <= 32:
                    arr = bin2d(arr, factor=bf, mode="mean")
                    break

        if label is None:
            label = f"Frame {self.n_frames}"
        self._labels.append(label)

        with self.hold_sync():
            self._push_height = int(arr.shape[0])
            self._push_width = int(arr.shape[1])
            self._push_label = label
            self._push_bytes = arr.tobytes()
            self._push_counter = self._push_counter + 1
            self.n_frames = self.n_frames + 1

        return self

    @classmethod
    def from_folder(
        cls,
        folder: str | pathlib.Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        max_files: int = 0,
        **kwargs,
    ) -> "Live":
        """Load all matching files from a folder into a Live widget.

        Bulk-loads existing files for quick browsing — each file is pushed
        incrementally (one at a time, not all at once).

        Parameters
        ----------
        folder : str or pathlib.Path
            Directory to scan.
        pattern : str, default "*"
            Glob pattern (e.g. ``"*.emd"``, ``"*.tiff"``).
        recursive : bool, default False
            Search subdirectories.
        max_files : int, default 0
            Maximum files to load. 0 = no limit.
        **kwargs
            Forwarded to Live constructor (title, cmap, etc.).

        Returns
        -------
        Live
        """
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")

        if "title" not in kwargs:
            kwargs["title"] = folder.name

        widget = cls(**kwargs)

        if recursive:
            candidates = folder.rglob(pattern)
        else:
            candidates = folder.glob(pattern)
        files = sorted(
            [p for p in candidates if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS],
            key=lambda p: p.stat().st_mtime,
        )
        if max_files > 0 and len(files) > max_files:
            files = files[:max_files]

        import time as _t
        t0 = _t.perf_counter()
        for path in files:
            try:
                result = IO._read_single_file(path)
            except Exception as exc:
                print(f"Live.from_folder: skipped {path.name}: {exc}")
                continue
            if widget.pixel_size == 0.0 and result.pixel_size:
                widget.pixel_size = result.pixel_size
            arr = result.data
            if arr.ndim == 2:
                widget._queue(arr, result.title or path.stem)
            elif arr.ndim == 3:
                for j in range(arr.shape[0]):
                    label = result.labels[j] if j < len(result.labels) else f"{path.stem}[{j}]"
                    widget._queue(arr[j], label)
        widget._flush_batch()
        elapsed = (_t.perf_counter() - t0) * 1000
        print(f"Live.from_folder: {widget.n_frames} frames from {len(files)} files in {elapsed:.0f} ms")
        return widget

    @classmethod
    def watch(
        cls,
        folder: str | pathlib.Path,
        *,
        pattern: str = "*",
        interval: float = 2.0,
        recursive: bool = False,
        **kwargs,
    ) -> "Live":
        """Create a Live widget that watches a folder for new files.

        Parameters
        ----------
        folder : str or pathlib.Path
            Directory to monitor.
        pattern : str, default "*"
            Glob pattern (e.g. ``"*.emd"``, ``"*.tiff"``).
        interval : float, default 2.0
            Polling interval in seconds.
        recursive : bool, default False
            Search subdirectories.
        **kwargs
            Forwarded to Live constructor (title, cmap, mode, etc.).

        Returns
        -------
        Live
            Widget with built-in folder watching.
        """
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")

        if "title" not in kwargs:
            kwargs["title"] = folder.name

        widget = cls(**kwargs)
        watcher = _LiveWatcher(
            folder=folder,
            widget=widget,
            pattern=pattern,
            interval=interval,
            recursive=recursive,
        )
        widget._watcher = watcher
        watcher.start()
        return widget

    def stop(self) -> Self:
        """Stop folder watching (if active)."""
        if self._watcher is not None:
            self._watcher.stop()
        return self

    @property
    def watching(self) -> bool:
        """Whether folder watching is active."""
        return self._watcher is not None and self._watcher.running

    @property
    def labels(self) -> list[str]:
        """Labels for all pushed frames."""
        return list(self._labels)

    def summary(self):
        lines = [self.title or "Live", "═" * 32]
        lines.append(f"Frames:   {self.n_frames}")
        lines.append(f"Mode:     {self.mode}")
        lines.append(f"Buffer:   {self.buffer_size}")
        scale = "log" if self.log_scale else "linear"
        contrast = "auto" if self.auto_contrast else "manual"
        lines.append(f"Display:  {self.cmap} | {contrast} | {scale}")
        if self._watcher is not None:
            status = "watching" if self.watching else "stopped"
            lines.append(f"Watch:    {self._watcher._folder} ({status})")
        print("\n".join(lines))

    def __repr__(self) -> str:
        mode = self.mode.upper()
        watch = ", watching" if self.watching else ""
        return f"Live({self.n_frames} frames, mode={mode}{watch})"


# =========================================================================
# Internal file watcher for Live.watch()
# =========================================================================

_SUPPORTED_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff",
    ".emd", ".h5", ".hdf5",
    ".dm3", ".dm4",
    ".mrc", ".ser",
    ".npy", ".npz",
}


class _LiveWatcher:
    """Background thread that watches a folder and pushes to a Live widget."""

    def __init__(self, folder, widget, pattern, interval, recursive):
        self._folder = folder
        self._widget = widget
        self._pattern = pattern
        self._interval = interval
        self._recursive = recursive
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._known: set[pathlib.Path] = set()

    def _scan(self) -> list[pathlib.Path]:
        if self._recursive:
            candidates = self._folder.rglob(self._pattern)
        else:
            candidates = self._folder.glob(self._pattern)
        found = [
            p for p in candidates
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
        ]
        found.sort(key=lambda p: p.stat().st_mtime)
        return found

    def _load_and_push(self, files: list[pathlib.Path]) -> None:
        for path in files:
            try:
                result = IO._read_single_file(path)
            except Exception as exc:
                print(f"Live.watch: failed to load {path.name}: {exc}")
                continue
            arr = result.data
            if arr.ndim == 2:
                self._widget.push(arr, label=result.title or path.stem)
            elif arr.ndim == 3:
                for j in range(arr.shape[0]):
                    label = (
                        result.labels[j]
                        if j < len(result.labels)
                        else f"{path.stem}[{j}]"
                    )
                    self._widget.push(arr[j], label=label)
            self._known.add(path)

    def _is_stable(self, path: pathlib.Path, wait: float = 0.5) -> bool:
        """Check if file size is stable (not still being written)."""
        try:
            size1 = path.stat().st_size
            if size1 == 0:
                return False
            _time_module.sleep(wait)
            size2 = path.stat().st_size
            return size1 == size2
        except OSError:
            return False

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            current = self._scan()
            new = [p for p in current if p not in self._known]
            if new:
                # Wait for files to finish writing
                stable = [p for p in new if self._is_stable(p)]
                if stable:
                    self._load_and_push(stable)
                    print(f"Live.watch: +{len(stable)} file(s), {self._widget.n_frames} frames total")
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        initial = self._scan()
        if initial:
            self._load_and_push(initial)
            print(f"Live.watch: loaded {len(initial)} existing file(s), {self._widget.n_frames} frames")
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="live-watch",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
