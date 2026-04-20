/**
 * ShowDiffraction — Interactive d-spacing measurement for 4D-STEM.
 *
 * Dual-panel: diffraction pattern (left) + virtual image (right).
 * Click on DP to add spot markers with d-spacing calculation.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawColorbar } from "../scalebar";
import { formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, sliderRange } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";
import "./showdiffraction.css";

// ============================================================================
// Style constants
// ============================================================================

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

// ============================================================================
// Helper: Histogram (inline, same pattern as other widgets)
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
}

function Histogram({ data, vminPct, vmaxPct, onRangeChange, width = 110, height = 50, theme = "dark" }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => data ? computeHistogramFromBytes(data) : null, [data]);
  const draggingRef = React.useRef<"left" | "right" | null>(null);
  const isDark = theme === "dark";

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bins) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = isDark ? "#1a1a2e" : "#f0f0f0";
    ctx.fillRect(0, 0, width, height);
    const maxBin = Math.max(...Array.from(bins));
    if (maxBin > 0) {
      ctx.fillStyle = isDark ? "#555" : "#999";
      for (let i = 0; i < bins.length; i++) {
        const x = (i / bins.length) * width;
        const bw = width / bins.length;
        const bh = (bins[i] / maxBin) * height;
        ctx.fillRect(x, height - bh, bw, bh);
      }
    }
    const lx = (vminPct / 100) * width;
    const rx = (vmaxPct / 100) * width;
    ctx.fillStyle = isDark ? "rgba(0,0,0,0.5)" : "rgba(0,0,0,0.2)";
    ctx.fillRect(0, 0, lx, height);
    ctx.fillRect(rx, 0, width - rx, height);
    ctx.fillStyle = isDark ? "#4fc3f7" : "#1976d2";
    ctx.fillRect(lx - 1, 0, 3, height);
    ctx.fillRect(rx - 1, 0, 3, height);
  }, [bins, vminPct, vmaxPct, width, height, isDark]);

  const handleMouse = (e: React.MouseEvent, isDown: boolean) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / width) * 100));
    if (isDown) {
      const dl = Math.abs(pct - vminPct);
      const dr = Math.abs(pct - vmaxPct);
      draggingRef.current = dl < dr ? "left" : "right";
    }
    if (draggingRef.current === "left") onRangeChange(Math.min(pct, vmaxPct - 1), vmaxPct);
    else if (draggingRef.current === "right") onRangeChange(vminPct, Math.max(pct, vminPct + 1));
  };

  return (
    <canvas
      ref={canvasRef} width={width} height={height}
      style={{ cursor: "ew-resize", display: "block" }}
      onMouseDown={(e) => handleMouse(e, true)}
      onMouseMove={(e) => { if (draggingRef.current) handleMouse(e, false); }}
      onMouseUp={() => { draggingRef.current = null; }}
      onMouseLeave={() => { draggingRef.current = null; }}
    />
  );
}

// ============================================================================
// Helper: format stat value
// ============================================================================
function formatStat(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// ============================================================================
// Spot type
// ============================================================================

interface SpotDict {
  id: number;
  row: number;
  col: number;
  d_spacing: number | null;
  g_magnitude: number | null;
  r_pixels: number;
  intensity: number;
}

// ============================================================================
// Main component
// ============================================================================

function ShowDiffraction() {
  const { themeInfo, colors: themeColors } = useTheme();

  const themedSelect = {
    "& .MuiSelect-select": { py: 0.25, px: 1, fontSize: 10, color: themeColors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
    bgcolor: themeColors.controlBg,
    minWidth: 80,
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  // ── Model state ─────────────────────────────────────────────────────
  const [title] = useModelState<string>("title");
  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [shapeRows] = useModelState<number>("shape_rows");
  const [shapeCols] = useModelState<number>("shape_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");
  const [centerRow] = useModelState<number>("center_row");
  const [centerCol] = useModelState<number>("center_col");
  const [bfRadius] = useModelState<number>("bf_radius");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [spots] = useModelState<SpotDict[]>("spots");
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius] = useModelState<number>("snap_radius");
  const [, setSpotAddRequest] = useModelState<number[]>("_spot_add_request");
  const [, setSpotUndoRequest] = useModelState<boolean>("_spot_undo_request");
  const [, setSpotClearRequest] = useModelState<boolean>("_spot_clear_request");
  const [dpColormap, setDpColormap] = useModelState<string>("dp_colormap");
  const [dpScaleMode, setDpScaleMode] = useModelState<string>("dp_scale_mode");
  const [dpVminPct, setDpVminPct] = useModelState<number>("dp_vmin_pct");
  const [dpVmaxPct, setDpVmaxPct] = useModelState<number>("dp_vmax_pct");
  const [viColormap] = useModelState<string>("vi_colormap");
  const [viVminPct] = useModelState<number>("vi_vmin_pct");
  const [viVmaxPct] = useModelState<number>("vi_vmax_pct");
  const [dpStats] = useModelState<number[]>("dp_stats");
  const [viStats] = useModelState<number[]>("vi_stats");
  // dp_global_min/max available via useModelState if needed for histogram
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("ShowDiffraction", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideStats = toolVisibility.isHidden("stats");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideDisplay = toolVisibility.isHidden("display");
  const hideExport = toolVisibility.isHidden("export");
  const hideSpots = toolVisibility.isHidden("spots");
  const lockSpots = toolVisibility.isLocked("spots");

  // ── Local UI state ──────────────────────────────────────────────────
  const CANVAS_SIZE = 384;
  const canvasSize = CANVAS_SIZE;
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [dpHistData, setDpHistData] = React.useState<Float32Array | null>(null);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const viCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const viUiRef = React.useRef<HTMLCanvasElement>(null);
  const viOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [dpVersion, setDpVersion] = React.useState(0);
  const [viVersion, setViVersion] = React.useState(0);
  const dpVminRef = React.useRef(0);
  const dpVmaxRef = React.useRef(1);

  // ── DP rendering (expensive: colormap) ──────────────────────────────
  React.useEffect(() => {
    if (!frameBytes || !frameBytes.byteLength) return;
    const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    let scaled: Float32Array;
    if (dpScaleMode === "log") {
      scaled = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) scaled[i] = Math.log1p(Math.max(0, raw[i]));
    } else {
      scaled = raw;
    }
    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, dpVminPct, dpVmaxPct);
    dpVminRef.current = vmin;
    dpVmaxRef.current = vmax;
    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
    let offscreen = dpOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); dpOffscreenRef.current = offscreen; }
    offscreen.width = detCols;
    offscreen.height = detRows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(detCols, detRows);
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
    setDpHistData(scaled);
    setDpVersion(v => v + 1);
  }, [frameBytes, dpColormap, dpScaleMode, dpVminPct, dpVmaxPct, detRows, detCols]);

  // ── DP draw (cheap: zoom/pan) ───────────────────────────────────────
  React.useLayoutEffect(() => {
    const canvas = dpCanvasRef.current;
    const offscreen = dpOffscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * dpZoom) / 2 + dpPanX;
    const offY = (canvasSize - canvasSize * dpZoom) / 2 + dpPanY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * dpZoom, canvasSize * dpZoom);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols]);

  // ── DP UI overlay (spots, center, scale bar) ────────────────────────
  React.useLayoutEffect(() => {
    const canvas = dpUiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    const scX = (cssW / detCols) * dpZoom;
    const scY = (cssW / detRows) * dpZoom;
    const offX = (cssW - cssW * dpZoom) / 2 + dpPanX;
    const offY = (cssW - cssW * dpZoom) / 2 + dpPanY;

    // Center crosshair
    const cx = offX + centerCol * scX;
    const cy = offY + centerRow * scY;
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10);
    ctx.stroke();
    // BF disk circle
    const br = bfRadius * scX;
    ctx.beginPath();
    ctx.arc(cx, cy, br, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Spot markers
    const spotColor = themeInfo.theme === "dark" ? "#00ff88" : "#1a7a1a";
    if (spots && spots.length > 0) {
      for (const spot of spots) {
        const sx = offX + spot.col * scX;
        const sy = offY + spot.row * scY;
        ctx.strokeStyle = spotColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
        ctx.stroke();
        // Number label
        ctx.fillStyle = spotColor;
        ctx.font = "bold 10px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${spot.id}`, sx + 8, sy - 2);
      }
    }

    // Colorbar
    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
    drawColorbar(ctx, cssW, cssW, lut, dpVminRef.current, dpVmaxRef.current, dpScaleMode === "log");

    // Zoom indicator
    if (dpZoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${dpZoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols, centerRow, centerCol, bfRadius, spots, dpColormap, dpScaleMode, themeInfo.theme]);

  // ── VI rendering (expensive: colormap) ──────────────────────────────
  React.useEffect(() => {
    if (!virtualImageBytes || !virtualImageBytes.byteLength) return;
    const raw = new Float32Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, virtualImageBytes.byteLength / 4);
    const { min: dataMin, max: dataMax } = findDataRange(raw);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, viVminPct, viVmaxPct);
    const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
    let offscreen = viOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); viOffscreenRef.current = offscreen; }
    offscreen.width = shapeCols;
    offscreen.height = shapeRows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(shapeCols, shapeRows);
    applyColormap(raw, imgData.data, lut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
    setViVersion(v => v + 1);
  }, [virtualImageBytes, viColormap, viVminPct, viVmaxPct, shapeRows, shapeCols]);

  // ── VI draw (cheap: zoom/pan) ───────────────────────────────────────
  React.useLayoutEffect(() => {
    const canvas = viCanvasRef.current;
    const offscreen = viOffscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * viZoom) / 2 + viPanX;
    const offY = (canvasSize - canvasSize * viZoom) / 2 + viPanY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * viZoom, canvasSize * viZoom);
  }, [viVersion, viZoom, viPanX, viPanY, canvasSize]);

  // ── VI UI overlay (position crosshair, scale bar) ───────────────────
  React.useLayoutEffect(() => {
    const canvas = viUiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    const scX = (cssW / shapeCols) * viZoom;
    const scY = (cssW / shapeRows) * viZoom;
    const offX = (cssW - cssW * viZoom) / 2 + viPanX;
    const offY = (cssW - cssW * viZoom) / 2 + viPanY;

    // Position crosshair
    const px = offX + (posCol + 0.5) * scX;
    const py = offY + (posRow + 0.5) * scY;
    ctx.strokeStyle = "rgba(255,100,100,0.8)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px - 8, py); ctx.lineTo(px + 8, py);
    ctx.moveTo(px, py - 8); ctx.lineTo(px, py + 8);
    ctx.stroke();

    // Scale bar
    if (pixelSize > 0 && canvas) {
      drawScaleBarHiDPI(canvas, DPR, viZoom, pixelSize, "Å", shapeCols);
    }

    // Zoom indicator
    if (viZoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${viZoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [viVersion, viZoom, viPanX, viPanY, canvasSize, shapeRows, shapeCols, posRow, posCol, pixelSize]);

  // ── DP mouse handlers ───────────────────────────────────────────────
  const dpIsDragging = React.useRef(false);
  const dpDragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const dpToImage = (e: React.MouseEvent) => {
    const canvas = dpCanvasRef.current;
    if (!canvas) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const offX = (canvasSize - canvasSize * dpZoom) / 2 + dpPanX;
    const offY = (canvasSize - canvasSize * dpZoom) / 2 + dpPanY;
    const col = (mx - offX) / (canvasSize * dpZoom) * detCols;
    const row = (my - offY) / (canvasSize * dpZoom) * detRows;
    return { row, col };
  };

  const handleDpMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 2 || e.shiftKey) {
      dpIsDragging.current = true;
      dpDragStart.current = { x: e.clientX, y: e.clientY, panX: dpPanX, panY: dpPanY };
      return;
    }
    // Left click: add spot
    if (!lockSpots) {
      const { row, col } = dpToImage(e);
      if (row >= 0 && row < detRows && col >= 0 && col < detCols) {
        setSpotAddRequest([row, col]);
      }
    }
  };

  const handleDpMouseMove = (e: React.MouseEvent) => {
    if (dpIsDragging.current) {
      setDpPanX(dpDragStart.current.panX + (e.clientX - dpDragStart.current.x));
      setDpPanY(dpDragStart.current.panY + (e.clientY - dpDragStart.current.y));
      return;
    }
    // Cursor readout
    if (!frameBytes || !frameBytes.byteLength) return;
    const { row, col } = dpToImage(e);
    const ri = Math.round(row), ci = Math.round(col);
    if (ri >= 0 && ri < detRows && ci >= 0 && ci < detCols) {
      const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
      setCursorInfo({ row: ri, col: ci, value: raw[ri * detCols + ci] });
    } else {
      setCursorInfo(null);
    }
  };

  const handleDpMouseUp = () => { dpIsDragging.current = false; };
  const handleDpMouseLeave = () => { dpIsDragging.current = false; setCursorInfo(null); };

  const handleDpWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setDpZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };

  const resetDpView = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  // ── VI mouse handlers ───────────────────────────────────────────────
  const viIsDragging = React.useRef(false);
  const viDragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const viToImage = (e: React.MouseEvent) => {
    const canvas = viCanvasRef.current;
    if (!canvas) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const offX = (canvasSize - canvasSize * viZoom) / 2 + viPanX;
    const offY = (canvasSize - canvasSize * viZoom) / 2 + viPanY;
    const col = (mx - offX) / (canvasSize * viZoom) * shapeCols;
    const row = (my - offY) / (canvasSize * viZoom) * shapeRows;
    return { row, col };
  };

  const handleViMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 2 || e.shiftKey) {
      viIsDragging.current = true;
      viDragStart.current = { x: e.clientX, y: e.clientY, panX: viPanX, panY: viPanY };
      return;
    }
    const { row, col } = viToImage(e);
    const r = Math.round(row), c = Math.round(col);
    if (r >= 0 && r < shapeRows && c >= 0 && c < shapeCols) {
      setPosRow(r);
      setPosCol(c);
    }
  };

  const handleViMouseMove = (e: React.MouseEvent) => {
    if (viIsDragging.current) {
      setViPanX(viDragStart.current.panX + (e.clientX - viDragStart.current.x));
      setViPanY(viDragStart.current.panY + (e.clientY - viDragStart.current.y));
    }
  };

  const handleViMouseUp = () => { viIsDragging.current = false; };
  const handleViWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setViZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };
  const resetViView = () => { setViZoom(1); setViPanX(0); setViPanY(0); };

  // ── Wheel scroll prevention ─────────────────────────────────────────
  const dpContainerRef = React.useRef<HTMLDivElement>(null);
  const viContainerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const prevent = (e: WheelEvent) => e.preventDefault();
    const dp = dpContainerRef.current;
    const vi = viContainerRef.current;
    if (dp) dp.addEventListener("wheel", prevent, { passive: false });
    if (vi) vi.addEventListener("wheel", prevent, { passive: false });
    return () => {
      if (dp) dp.removeEventListener("wheel", prevent);
      if (vi) vi.removeEventListener("wheel", prevent);
    };
  }, []);

  // ── Export DP ───────────────────────────────────────────────────────
  const handleCopyDP = () => {
    const offscreen = dpOffscreenRef.current;
    if (!offscreen) return;
    offscreen.toBlob((blob) => {
      if (blob) {
        try { navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]); }
        catch { downloadBlob(blob, "diffraction.png"); }
      }
    });
  };

  // ── Keyboard ────────────────────────────────────────────────────────
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") { resetDpView(); resetViView(); }
    if (e.key === "z" || e.key === "Z") { if (!lockSpots) setSpotUndoRequest(true); }
  };

  // ── JSX ─────────────────────────────────────────────────────────────
  const canvasBox = {
    position: "relative" as const,
    border: `1px solid ${themeColors.border}`,
    overflow: "hidden",
    width: canvasSize,
    height: canvasSize,
    bgcolor: "#000",
  };

  return (
    <Box
      sx={{ p: `${SPACING.LG}px`, bgcolor: themeColors.bg, color: themeColors.text, outline: "none" }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Diffraction"}</Typography>
        <Stack direction="row" spacing={`${SPACING.XS}px`}>
          {!hideExport && (
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={handleCopyDP}>
              COPY
            </Button>
          )}
        </Stack>
      </Stack>

      {/* Main panels */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* DP Panel */}
        <Box>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            DP at ({posRow}, {posCol})
            {cursorInfo && <span style={{ marginLeft: 8, color: themeColors.accent }}>
              ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
            </span>}
          </Typography>
          <Box ref={dpContainerRef} sx={canvasBox}>
            <canvas ref={dpCanvasRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, imageRendering: "pixelated" }} />
            <canvas ref={dpUiRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            <canvas
              style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, cursor: "crosshair", opacity: 0 }}
              width={canvasSize} height={canvasSize}
              onMouseDown={handleDpMouseDown}
              onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp}
              onMouseLeave={handleDpMouseLeave}
              onWheel={handleDpWheel}
              onDoubleClick={resetDpView}
            />
          </Box>
          {/* DP Stats */}
          {!hideStats && showStats && dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2 }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[0])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[1])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[2])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[3])}</Box>
              </Typography>
            </Box>
          )}
        </Box>

        {/* VI Panel */}
        <Box>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            Virtual Image (BF)
          </Typography>
          <Box ref={viContainerRef} sx={canvasBox}>
            <canvas ref={viCanvasRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, imageRendering: "pixelated" }} />
            <canvas ref={viUiRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            <canvas
              style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, cursor: "crosshair", opacity: 0 }}
              width={canvasSize} height={canvasSize}
              onMouseDown={handleViMouseDown}
              onMouseMove={handleViMouseMove}
              onMouseUp={handleViMouseUp}
              onMouseLeave={() => { viIsDragging.current = false; }}
              onWheel={handleViWheel}
              onDoubleClick={resetViView}
            />
          </Box>
          {/* VI Stats */}
          {!hideStats && showStats && viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2 }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[0])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[1])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[2])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[3])}</Box>
              </Typography>
            </Box>
          )}
        </Box>
      </Stack>

      {/* Spots Table */}
      {!hideSpots && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize * 2 + SPACING.LG }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.label, color: themeColors.text }}>
              Spots ({spots ? spots.length : 0})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockSpots || !spots || spots.length === 0}
                onClick={() => setSpotUndoRequest(true)}
              >
                UNDO
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockSpots || !spots || spots.length === 0}
                onClick={() => setSpotClearRequest(true)}
              >
                CLEAR
              </Button>
            </Stack>
          </Stack>
          {spots && spots.length > 0 && (
            <Box sx={{ maxHeight: 200, overflow: "auto", border: `1px solid ${themeColors.border}` }}>
              <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left" }}>
                    <th style={{ padding: "2px 6px" }}>#</th>
                    <th style={{ padding: "2px 6px" }}>(row, col)</th>
                    <th style={{ padding: "2px 6px" }}>d (Å)</th>
                    <th style={{ padding: "2px 6px" }}>|g| (1/Å)</th>
                    <th style={{ padding: "2px 6px" }}>I</th>
                  </tr>
                </thead>
                <tbody>
                  {spots.map((spot: SpotDict) => (
                    <tr key={spot.id} style={{ borderBottom: `1px solid ${themeColors.border}22` }}>
                      <td style={{ padding: "2px 6px", color: themeColors.accent }}>{spot.id}</td>
                      <td style={{ padding: "2px 6px" }}>({spot.row.toFixed(1)}, {spot.col.toFixed(1)})</td>
                      <td style={{ padding: "2px 6px" }}>{spot.d_spacing != null ? spot.d_spacing.toFixed(3) : "—"}</td>
                      <td style={{ padding: "2px 6px" }}>{spot.g_magnitude != null ? spot.g_magnitude.toFixed(4) : `${spot.r_pixels.toFixed(1)} px`}</td>
                      <td style={{ padding: "2px 6px" }}>{formatNumber(spot.intensity)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          )}
        </Box>
      )}

      {/* Controls */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize * 2 + SPACING.LG }}>
          <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap" }}>
            {/* Snap control */}
            {!hideSpots && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Snap:</Typography>
                <Switch
                  size="small" checked={snapEnabled}
                  onChange={(_, v) => { if (!lockSpots) setSnapEnabled(v); }}
                  sx={switchStyles.small}
                  disabled={lockSpots}
                />
                {snapEnabled && (
                  <>
                    <Typography sx={typography.label}>r:</Typography>
                    <Typography sx={typography.value}>{snapRadius}</Typography>
                  </>
                )}
              </Box>
            )}

            {/* DP Colormap */}
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Colormap:</Typography>
                <Select
                  size="small" value={dpColormap}
                  onChange={(e) => setDpColormap(e.target.value)}
                  sx={themedSelect}
                  MenuProps={themedMenuProps}
                >
                  {COLORMAP_NAMES.map(n => <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>)}
                </Select>
              </Box>
            )}

            {/* Scale mode */}
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Scale:</Typography>
                <Select
                  size="small" value={dpScaleMode}
                  onChange={(e) => setDpScaleMode(e.target.value)}
                  sx={{ ...themedSelect, minWidth: 60 }}
                  MenuProps={themedMenuProps}
                >
                  <MenuItem value="linear" sx={{ fontSize: 10 }}>Linear</MenuItem>
                  <MenuItem value="log" sx={{ fontSize: 10 }}>Log</MenuItem>
                </Select>
              </Box>
            )}

            {/* Histogram */}
            {!hideHistogram && (
              <Box sx={controlRow}>
                <Histogram
                  data={dpHistData}
                  vminPct={dpVminPct}
                  vmaxPct={dpVmaxPct}
                  onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }}
                  theme={themeInfo.theme}
                />
              </Box>
            )}
          </Stack>

          {/* Center info */}
          <Box sx={{ ...controlRow, mt: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              Center: ({centerRow.toFixed(1)}, {centerCol.toFixed(1)})  BF r={bfRadius.toFixed(1)}
              {kCalibrated && <span style={{ marginLeft: 8 }}>k={kPixelSize.toFixed(4)} 1/Å/px</span>}
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(ShowDiffraction);
