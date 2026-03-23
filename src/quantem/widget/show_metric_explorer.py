"""
MetricExplorer: Interactive metric explorer for SSB aberration parameter sweeps.

Displays multiple image quality metrics across a parameter sweep, letting scientists
visually connect metric curves to phase images in real-time. Click any point on any
chart to select and inspect the corresponding phase image.
"""

import json
import pathlib
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

_DEFAULT_METRIC_DIRECTIONS = {
    "variance_loss": "min",
    "tv": "min",
    "total_variation": "min",
    "fft_snr": "max",
    "fft_peak_sharpness": "max",
    "snr": "max",
    "contrast": "max",
}


class MetricExplorer(anywidget.AnyWidget):
    """
    Interactive metric explorer for parameter sweep visualization.

    Displays multiple image quality metrics as line charts grouped by a parameter,
    with a large selected phase image and reference images. Hover or click any
    point on any chart to inspect the corresponding phase image.

    Parameters
    ----------
    points : list of dict
        Each dict has keys: ``"image"`` (2D array), ``"label"`` (str),
        ``"metrics"`` (dict of metric name → value), ``"params"`` (dict).
    x_key : str
        Parameter key for the x-axis (must exist in each point's ``params``).
    x_label : str, optional
        Display label for the x-axis.
    group_key : str, optional
        Parameter key for grouping curves into separate colored lines.
    reference_images : list of array_like, optional
        Reference images (always visible, e.g. BF, DPC).
    reference_labels : list of str, optional
        Labels for reference images.
    metric_labels : dict, optional
        Display labels for metrics, e.g. ``{"tv": "Total Variation"}``.
    metric_directions : dict, optional
        Best direction per metric: ``"min"`` or ``"max"``.
    cmap : str, default "inferno"
        Colormap for all images.
    state : dict or str or Path, optional
        Restore display settings from a dict or a JSON file path.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show_metric_explorer.js"
    _css = pathlib.Path(__file__).parent / "static" / "show_metric_explorer.css"

    # =========================================================================
    # Core State
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    points_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)
    width = traitlets.Int(0).tag(sync=True)
    metrics_json = traitlets.Unicode("[]").tag(sync=True)
    params_json = traitlets.Unicode("[]").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # =========================================================================
    # Axis / Grouping
    # =========================================================================
    x_key = traitlets.Unicode("").tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    group_key = traitlets.Unicode("").tag(sync=True)

    # =========================================================================
    # Metric Display
    # =========================================================================
    metric_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    metric_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    metric_directions = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # =========================================================================
    # Calibration
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)

    # =========================================================================
    # Selection
    # =========================================================================
    selected_index = traitlets.Int(0).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)

    # =========================================================================
    # Tool Lock/Hide
    # =========================================================================
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("MetricExplorer", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "MetricExplorer",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "MetricExplorer",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        points,
        x_key: str = "",
        x_label: str = "",
        group_key: str = "",
            metric_labels=None,
        metric_directions=None,
        cmap: str = "inferno",
        metric_filter=None,
        pixel_size: float = 0.0,
        pixel_unit: str = "px",
        selected_index: int = 0,
        disabled_tools=None,
        disable_display: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hidden_tools=None,
        hide_display: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Validate points
        if not points or len(points) == 0:
            raise ValueError("points must be a non-empty list of dicts.")

        # Apply metric filter: {"metric_name": (min, max)} or {"metric_name": max_value}
        if metric_filter:
            filtered = []
            for p in points:
                keep = True
                for metric_name, threshold in metric_filter.items():
                    val = p.get("metrics", {}).get(metric_name)
                    if val is None:
                        continue
                    if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                        lo, hi = threshold
                        if lo is not None and val < lo:
                            keep = False
                        if hi is not None and val > hi:
                            keep = False
                    elif isinstance(threshold, (int, float)):
                        # Single number = max threshold
                        if val > threshold:
                            keep = False
                if keep:
                    filtered.append(p)
            points = filtered

        if not points or len(points) == 0:
            raise ValueError("points must be a non-empty list of dicts (check metric_filter).")

        self._points = list(points)
        n = len(points)

        # Extract and stack images
        images = []
        for i, p in enumerate(points):
            img = to_numpy(p["image"]).astype(np.float32)
            if img.ndim != 2:
                raise ValueError(f"Point {i} image must be 2D, got {img.ndim}D.")
            images.append(img)

        h, w = images[0].shape
        for i, img in enumerate(images):
            if img.shape != (h, w):
                raise ValueError(
                    f"All images must have the same shape. Point 0 is {(h, w)}, "
                    f"point {i} is {img.shape}."
                )

        self._data = np.stack(images)  # (N, H, W)
        self.n_points = n
        self.height = h
        self.width = w

        # Extract metrics
        m_names = list(points[0]["metrics"].keys())
        metrics_list = [p["metrics"] for p in points]
        params_list = [p.get("params", {}) for p in points]

        self.metrics_json = json.dumps(metrics_list)
        self.params_json = json.dumps(params_list)
        self.labels = [str(p.get("label", f"Point {i}")) for i, p in enumerate(points)]
        self.metric_names = m_names

        # Metric display labels
        if metric_labels and isinstance(metric_labels, dict):
            self.metric_labels = [metric_labels.get(mn, mn) for mn in m_names]
        else:
            self.metric_labels = list(m_names)

        # Metric directions
        if metric_directions and isinstance(metric_directions, dict):
            self.metric_directions = [
                metric_directions.get(mn, _DEFAULT_METRIC_DIRECTIONS.get(mn, "min"))
                for mn in m_names
            ]
        else:
            self.metric_directions = [
                _DEFAULT_METRIC_DIRECTIONS.get(mn, "min") for mn in m_names
            ]

        # Axis / grouping
        self.x_key = x_key
        self.x_label = x_label
        self.group_key = group_key
        self.cmap = cmap
        self.pixel_size = pixel_size
        self.pixel_unit = pixel_unit
        self.selected_index = selected_index

        # Tool visibility
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_export=hide_export,
            hide_all=hide_all,
        )

        # Send data
        self.points_bytes = self._data.tobytes()

        # Restore state
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # =========================================================================
    # Properties for downstream use
    # =========================================================================
    @property
    def selected_point(self) -> dict:
        """Full point dict for the currently selected index."""
        return self._points[self.selected_index]

    @property
    def selected_params(self) -> dict:
        """Parameter dict — feed directly into iterative ptycho."""
        return self._points[self.selected_index].get("params", {})

    @property
    def selected_image(self) -> np.ndarray:
        """Phase image for the selected point."""
        return self._data[self.selected_index]

    # =========================================================================
    # State Persistence
    # =========================================================================
    def state_dict(self):
        return {
            "selected_index": self.selected_index,
            "cmap": self.cmap,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        save_state_file(path, "MetricExplorer", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = ["MetricExplorer", "=" * 32]
        lines.append(f"Points:   {self.n_points}")
        lines.append(f"Image:    {self.height} x {self.width}")
        lines.append(f"Metrics:  {', '.join(self.metric_names)}")
        if self.x_key:
            lines.append(f"X axis:   {self.x_label or self.x_key}")
        if self.group_key:
            lines.append(f"Group by: {self.group_key}")
        sel = self._points[self.selected_index]
        lines.append(f"Selected: [{self.selected_index}] {sel.get('label', '')}")
        for k, v in sel.get("metrics", {}).items():
            lines.append(f"  {k}: {v:.4g}")
        lines.append(f"Display:  {self.cmap}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        return (
            f"MetricExplorer({self.n_points} points, "
            f"{self.height}x{self.width}, "
            f"{len(self.metric_names)} metrics)"
        )


bind_tool_runtime_api(MetricExplorer, "MetricExplorer")
