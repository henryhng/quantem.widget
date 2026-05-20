/**
 * TransferFunctionEditor — interactive opacity + color transfer function designer.
 *
 * Single-canvas editor inspired by Tomviz / 3D Slicer / VTK:
 *   - histogram as filled gray polygon (with optional log y-axis)
 *   - opacity curve drawn as a polyline through handles (x, 1 - opacity)
 *   - handles drawn as circles, filled with their per-handle color
 *   - LUT preview drawn as a thin gradient strip below the canvas
 *
 * Interactions
 *   - drag handle to move (x stays monotonic; endpoints x-locked)
 *   - double-click empty area to add a handle (color sampled from cmap ramp)
 *   - right-click (or alt-click) handle to remove (cannot remove endpoints)
 *   - click handle color swatch in control row to open a color picker
 *   - "Reset" button restores the default 2-handle ramp
 *   - "Stretch" dropdown: linear / log / power / asinh
 *   - "Histogram log" toggle: scales histogram bar heights logarithmically
 *
 * The widget owns no rendering for the output volume — it only emits
 * ``tf_lut_bytes`` (256 × 4 uint8 RGBA). A future PR will wire Show3DVolume
 * to consume this.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import { useTheme } from "../theme";
import { COLORMAPS, COLORMAP_NAMES } from "../colormaps";
import { extractBytes, extractFloat32 } from "../format";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const DPR = window.devicePixelRatio || 1;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const DEFAULT_CANVAS_W = 360;
const DEFAULT_CANVAS_H = 200;
const LUT_STRIP_H = 14;
const HANDLE_RADIUS = 6;

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Handle {
  x: number;          // [0, 1]
  opacity: number;    // [0, 1]
  color: [number, number, number]; // RGB uint8
}

interface RawHandle {
  x: number;
  opacity: number;
  color: number[];
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function sampleCmap(cmap: string, x: number): [number, number, number] {
  const lut = COLORMAPS[cmap] || COLORMAPS["viridis"];
  const idx = Math.max(0, Math.min(255, Math.floor(clamp01(x) * 255)));
  const off = idx * 3;
  return [lut[off], lut[off + 1], lut[off + 2]];
}

function colorToHex(rgb: [number, number, number]): string {
  const hex = (n: number) => clampByte(n).toString(16).padStart(2, "0");
  return `#${hex(rgb[0])}${hex(rgb[1])}${hex(rgb[2])}`;
}

function hexToRgb(hex: string): [number, number, number] {
  const m = hex.match(/^#?([0-9a-f]{6})$/i);
  if (!m) return [255, 255, 255];
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

function normalizeHandles(raw: RawHandle[] | null | undefined): Handle[] {
  if (!raw) return [];
  const out: Handle[] = raw.map((h) => {
    const color = Array.isArray(h.color) ? h.color : [255, 255, 255];
    return {
      x: clamp01(Number(h.x) || 0),
      opacity: clamp01(Number(h.opacity) || 0),
      color: [
        clampByte(Number(color[0]) || 0),
        clampByte(Number(color[1]) || 0),
        clampByte(Number(color[2]) || 0),
      ],
    };
  });
  out.sort((a, b) => a.x - b.x);
  return out;
}

function defaultHandles(cmap: string): Handle[] {
  return [
    { x: 0, opacity: 0, color: sampleCmap(cmap, 0) },
    { x: 1, opacity: 1, color: sampleCmap(cmap, 1) },
  ];
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

interface DrawArgs {
  handles: Handle[];
  histogram: Float32Array | null;
  logHistogram: boolean;
  selectedIdx: number;
  themeColors: { bg: string; bgAlt: string; border: string; textMuted: string; accent: string };
  cssW: number;
  cssH: number;
}

function drawEditor(ctx: CanvasRenderingContext2D, args: DrawArgs): void {
  const { handles, histogram, logHistogram, selectedIdx, themeColors, cssW, cssH } = args;

  // Background
  ctx.fillStyle = themeColors.bgAlt;
  ctx.fillRect(0, 0, cssW, cssH);

  // Histogram
  if (histogram && histogram.length > 0) {
    const n = histogram.length;
    let maxBar = 0;
    for (let i = 0; i < n; i++) {
      const v = logHistogram ? Math.log1p(histogram[i] * 1000) : histogram[i];
      if (v > maxBar) maxBar = v;
    }
    if (maxBar > 0) {
      ctx.fillStyle = themeColors.border;
      ctx.beginPath();
      ctx.moveTo(0, cssH);
      for (let i = 0; i < n; i++) {
        const v = logHistogram ? Math.log1p(histogram[i] * 1000) : histogram[i];
        const x = (i / (n - 1)) * cssW;
        const y = cssH - (v / maxBar) * cssH;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(cssW, cssH);
      ctx.closePath();
      ctx.fill();
    }
  }

  // Opacity curve (polyline through handles)
  if (handles.length >= 1) {
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < handles.length; i++) {
      const x = handles[i].x * cssW;
      const y = (1 - handles[i].opacity) * cssH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Subtle fill under the opacity curve so users see what is "above the floor"
    ctx.fillStyle = themeColors.accent + "22";
    ctx.beginPath();
    ctx.moveTo(handles[0].x * cssW, cssH);
    for (let i = 0; i < handles.length; i++) {
      ctx.lineTo(handles[i].x * cssW, (1 - handles[i].opacity) * cssH);
    }
    ctx.lineTo(handles[handles.length - 1].x * cssW, cssH);
    ctx.closePath();
    ctx.fill();
  }

  // Handles
  for (let i = 0; i < handles.length; i++) {
    const h = handles[i];
    const x = h.x * cssW;
    const y = (1 - h.opacity) * cssH;
    const isSel = i === selectedIdx;
    ctx.fillStyle = `rgb(${h.color[0]}, ${h.color[1]}, ${h.color[2]})`;
    ctx.strokeStyle = isSel ? themeColors.accent : "#000";
    ctx.lineWidth = isSel ? 2.5 : 1;
    ctx.beginPath();
    ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Frame
  ctx.strokeStyle = themeColors.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, cssW - 1, cssH - 1);
}

function drawLutStrip(
  ctx: CanvasRenderingContext2D,
  lut: Uint8Array | null,
  cssW: number,
  cssH: number,
  border: string,
): void {
  // Checkerboard so alpha is visible
  const tile = 6;
  for (let yy = 0; yy < cssH; yy += tile) {
    for (let xx = 0; xx < cssW; xx += tile) {
      const dark = ((xx / tile) + (yy / tile)) % 2 === 0;
      ctx.fillStyle = dark ? "#333" : "#666";
      ctx.fillRect(xx, yy, tile, tile);
    }
  }
  if (lut && lut.length >= 4) {
    const n = lut.length / 4;
    const stripData = ctx.createImageData(Math.round(cssW * DPR), Math.round(cssH * DPR));
    const sw = stripData.width;
    const sh = stripData.height;
    for (let xx = 0; xx < sw; xx++) {
      const idx = Math.min(n - 1, Math.floor((xx / sw) * n));
      const off = idx * 4;
      const r = lut[off];
      const g = lut[off + 1];
      const b = lut[off + 2];
      const a = lut[off + 3];
      for (let yy = 0; yy < sh; yy++) {
        const p = (yy * sw + xx) * 4;
        stripData.data[p] = r;
        stripData.data[p + 1] = g;
        stripData.data[p + 2] = b;
        stripData.data[p + 3] = a;
      }
    }
    // Draw onto an offscreen canvas at device pixels, then blit at CSS size.
    const off = document.createElement("canvas");
    off.width = sw;
    off.height = sh;
    const offCtx = off.getContext("2d");
    if (offCtx) {
      offCtx.putImageData(stripData, 0, 0);
      ctx.drawImage(off, 0, 0, cssW, cssH);
    }
  }
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, cssW - 1, cssH - 1);
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

function hitTestHandle(
  handles: Handle[],
  px: number,
  py: number,
  cssW: number,
  cssH: number,
): number {
  let best = -1;
  let bestD2 = (HANDLE_RADIUS + 4) * (HANDLE_RADIUS + 4);
  for (let i = 0; i < handles.length; i++) {
    const hx = handles[i].x * cssW;
    const hy = (1 - handles[i].opacity) * cssH;
    const dx = hx - px;
    const dy = hy - py;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) {
      bestD2 = d2;
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Main Widget
// ---------------------------------------------------------------------------

function TransferFunctionEditorWidget() {
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = tc;
  const isDark = themeInfo.theme === "dark";

  // Model state
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [tfHandlesRaw, setTfHandlesRaw] = useModelState<RawHandle[]>("tf_handles");
  const [tfLutBytesRaw] = useModelState<DataView | null>("tf_lut_bytes");
  const [dataMin] = useModelState<number>("data_min");
  const [dataMax] = useModelState<number>("data_max");
  const [histogramBytesRaw] = useModelState<DataView | null>("histogram_bytes");
  const [stretchPreset, setStretchPreset] = useModelState<string>("stretch_preset");
  const [logHistogram, setLogHistogram] = useModelState<boolean>("log_histogram");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");

  const handles = React.useMemo(() => normalizeHandles(tfHandlesRaw), [tfHandlesRaw]);

  const histogram = React.useMemo(
    () => extractFloat32(histogramBytesRaw as DataView),
    [histogramBytesRaw],
  );

  const lut = React.useMemo(() => {
    if (!tfLutBytesRaw) return null;
    return extractBytes(tfLutBytesRaw as DataView);
  }, [tfLutBytesRaw]);

  // Selection
  const [selectedIdx, setSelectedIdx] = React.useState<number>(-1);

  // Canvas + sizing
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const lutStripRef = React.useRef<HTMLCanvasElement>(null);
  // Fixed canvas size for now — drag-resize is a future enhancement.
  const canvasW = DEFAULT_CANVAS_W;
  const canvasH = DEFAULT_CANVAS_H;

  const commitHandles = React.useCallback(
    (next: Handle[]) => {
      // Sort + clamp on commit so Python validator stays consistent with JS.
      const cleaned = normalizeHandles(next);
      setTfHandlesRaw(cleaned.map((h) => ({ x: h.x, opacity: h.opacity, color: [...h.color] })));
    },
    [setTfHandlesRaw],
  );

  // ---- Drawing ----
  React.useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    c.width = canvasW * DPR;
    c.height = canvasH * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    drawEditor(ctx, {
      handles,
      histogram,
      logHistogram,
      selectedIdx,
      themeColors,
      cssW: canvasW,
      cssH: canvasH,
    });
  }, [handles, histogram, logHistogram, selectedIdx, themeColors, canvasW, canvasH]);

  React.useEffect(() => {
    const c = lutStripRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    c.width = canvasW * DPR;
    c.height = LUT_STRIP_H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    drawLutStrip(ctx, lut, canvasW, LUT_STRIP_H, themeColors.border);
  }, [lut, canvasW, themeColors]);

  // ---- Mouse interactions on the main canvas ----
  const dragRef = React.useRef<{ idx: number; pointerId: number } | null>(null);

  const getMousePos = React.useCallback((e: React.MouseEvent | MouseEvent) => {
    const c = canvasRef.current;
    if (!c) return { x: 0, y: 0 };
    const rect = c.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      const { x, y } = getMousePos(e);
      const hit = hitTestHandle(handles, x, y, canvasW, canvasH);

      // Right-click or alt-click → remove (cannot remove endpoints)
      if ((e.button === 2 || e.altKey) && hit >= 0) {
        if (hit === 0 || hit === handles.length - 1) return;
        e.preventDefault();
        const next = handles.slice();
        next.splice(hit, 1);
        commitHandles(next);
        setSelectedIdx(-1);
        return;
      }

      if (e.button === 0 && hit >= 0) {
        setSelectedIdx(hit);
        dragRef.current = { idx: hit, pointerId: e.nativeEvent ? 1 : 0 };
      } else if (e.button === 0 && hit < 0) {
        setSelectedIdx(-1);
      }
    },
    [handles, canvasW, canvasH, commitHandles, getMousePos],
  );

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      if (!dragRef.current) return;
      const { x, y } = getMousePos(e);
      const i = dragRef.current.idx;
      if (i < 0 || i >= handles.length) return;
      const isEndpoint = i === 0 || i === handles.length - 1;
      const next = handles.slice();
      let newX = clamp01(x / canvasW);
      const newY = clamp01(1 - y / canvasH);
      if (isEndpoint) {
        // Endpoints are pinned in x; only opacity can move.
        newX = next[i].x;
      } else {
        // Keep monotonic: clamp into neighbors' range with a small epsilon.
        const lo = next[i - 1].x + 1e-4;
        const hi = next[i + 1].x - 1e-4;
        newX = Math.max(lo, Math.min(hi, newX));
      }
      next[i] = { ...next[i], x: newX, opacity: newY };
      commitHandles(next);
    },
    [handles, canvasW, canvasH, commitHandles, getMousePos],
  );

  const handleMouseUp = React.useCallback(() => {
    dragRef.current = null;
  }, []);

  const handleDoubleClick = React.useCallback(
    (e: React.MouseEvent) => {
      const { x, y } = getMousePos(e);
      const hit = hitTestHandle(handles, x, y, canvasW, canvasH);
      if (hit >= 0) return; // Don't add on top of an existing handle
      const newX = clamp01(x / canvasW);
      const newOpacity = clamp01(1 - y / canvasH);
      const color = sampleCmap(cmap, newX);
      const next = handles.slice();
      next.push({ x: newX, opacity: newOpacity, color });
      commitHandles(next);
      // Select the new handle (after sort, find by x)
      const sorted = normalizeHandles(next);
      const newIdx = sorted.findIndex((h) => Math.abs(h.x - newX) < 1e-6);
      setSelectedIdx(newIdx);
    },
    [handles, canvasW, canvasH, cmap, commitHandles, getMousePos],
  );

  const handleContextMenu = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
  }, []);

  // ---- Editing actions ----
  const resetHandles = React.useCallback(() => {
    commitHandles(defaultHandles(cmap));
    setSelectedIdx(-1);
  }, [cmap, commitHandles]);

  const removeSelected = React.useCallback(() => {
    if (selectedIdx < 0) return;
    if (selectedIdx === 0 || selectedIdx === handles.length - 1) return;
    const next = handles.slice();
    next.splice(selectedIdx, 1);
    commitHandles(next);
    setSelectedIdx(-1);
  }, [handles, selectedIdx, commitHandles]);

  const changeSelectedColor = React.useCallback(
    (hex: string) => {
      if (selectedIdx < 0 || selectedIdx >= handles.length) return;
      const next = handles.slice();
      next[selectedIdx] = { ...next[selectedIdx], color: hexToRgb(hex) };
      commitHandles(next);
    },
    [handles, selectedIdx, commitHandles],
  );

  // ---- Render ----
  const canvasContainer = {
    position: "relative" as const,
    bgcolor: themeColors.bgAlt,
    border: `1px solid ${themeColors.border}`,
    width: canvasW,
    height: canvasH,
  };

  const selectedHandle =
    selectedIdx >= 0 && selectedIdx < handles.length ? handles[selectedIdx] : null;
  const canRemoveSelected =
    selectedHandle !== null && selectedIdx !== 0 && selectedIdx !== handles.length - 1;

  const themedSelect = {
    fontSize: 10,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };
  const themedMenuProps = {
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
      },
    },
    sx: { zIndex: 9999 },
  };

  return (
    <Box
      className="transfer-function-editor-root"
      sx={{
        p: 2,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        width: "fit-content",
        outline: "none",
      }}
    >
      {/* Title */}
      <Typography
        variant="caption"
        sx={{
          ...typography.label,
          color: themeColors.accent,
          mb: `${SPACING.XS}px`,
          display: "block",
          height: 16,
          lineHeight: "16px",
          overflow: "hidden",
        }}
      >
        {title || "Transfer Function Editor"}
      </Typography>

      {/* Main editor canvas + LUT strip */}
      <Box sx={canvasContainer}>
        <canvas
          ref={canvasRef}
          style={{
            width: canvasW,
            height: canvasH,
            display: "block",
            cursor: "crosshair",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDoubleClick}
          onContextMenu={handleContextMenu}
        />
      </Box>
      <Box
        sx={{
          mt: `${SPACING.XS}px`,
          bgcolor: themeColors.bgAlt,
          border: `1px solid ${themeColors.border}`,
          width: canvasW,
          height: LUT_STRIP_H,
        }}
      >
        <canvas
          ref={lutStripRef}
          style={{ width: canvasW, height: LUT_STRIP_H, display: "block" }}
        />
      </Box>

      {/* Domain ticks */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          width: canvasW,
          mt: `${SPACING.XS}px`,
        }}
      >
        <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
          {Number(dataMin).toExponential(2)}
        </Typography>
        <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
          {Number(dataMax).toExponential(2)}
        </Typography>
      </Box>

      {/* Stats bar */}
      {showStats && (
        <Box
          sx={{
            mt: `${SPACING.SM}px`,
            px: 1,
            py: 0.5,
            bgcolor: themeColors.bgAlt,
            display: "flex",
            gap: 2,
            whiteSpace: "nowrap",
          }}
        >
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
            Handles{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {handles.length}
            </Box>
          </Typography>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
            Stretch{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {stretchPreset}
            </Box>
          </Typography>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
            Cmap{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {cmap}
            </Box>
          </Typography>
        </Box>
      )}

      {/* Controls */}
      {showControls && (
        <Box
          sx={{
            mt: `${SPACING.SM}px`,
            display: "flex",
            flexDirection: "column",
            gap: `${SPACING.XS}px`,
          }}
        >
          {/* Row 1: Reset + Stretch + Hist log + Cmap */}
          <Box
            sx={{
              ...controlRow,
              border: `1px solid ${themeColors.border}`,
              bgcolor: themeColors.controlBg,
            }}
          >
            <Button size="small" sx={compactButton} onClick={resetHandles}>
              Reset
            </Button>
            <Typography sx={{ ...typography.label, fontSize: 10 }}>Stretch:</Typography>
            <Select
              size="small"
              value={stretchPreset}
              onChange={(e) => setStretchPreset(String(e.target.value))}
              sx={{ ...themedSelect, minWidth: 70 }}
              MenuProps={themedMenuProps}
            >
              <MenuItem value="linear">Linear</MenuItem>
              <MenuItem value="log">Log</MenuItem>
              <MenuItem value="power">Power</MenuItem>
              <MenuItem value="asinh">Asinh</MenuItem>
            </Select>
            <Typography sx={{ ...typography.label, fontSize: 10 }}>Hist log:</Typography>
            <Switch
              checked={logHistogram}
              onChange={(e) => setLogHistogram(e.target.checked)}
              size="small"
              sx={switchStyles.small}
            />
            <Typography sx={{ ...typography.label, fontSize: 10 }}>Cmap:</Typography>
            <Select
              size="small"
              value={cmap}
              onChange={(e) => setCmap(String(e.target.value))}
              sx={{ ...themedSelect, minWidth: 80 }}
              MenuProps={themedMenuProps}
            >
              {COLORMAP_NAMES.map((name) => (
                <MenuItem key={name} value={name}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </Box>

          {/* Row 2: Selected handle editor */}
          <Box
            sx={{
              ...controlRow,
              border: `1px solid ${themeColors.border}`,
              bgcolor: themeColors.controlBg,
              opacity: selectedHandle ? 1 : 0.5,
            }}
          >
            <Typography sx={{ ...typography.label, fontSize: 10 }}>Selected:</Typography>
            {selectedHandle ? (
              <>
                <Typography sx={{ ...typography.value, minWidth: 80 }}>
                  x={selectedHandle.x.toFixed(3)} α={selectedHandle.opacity.toFixed(2)}
                </Typography>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Box
                  component="input"
                  type="color"
                  value={colorToHex(selectedHandle.color)}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    changeSelectedColor(e.target.value)
                  }
                  sx={{
                    width: 28,
                    height: 20,
                    p: 0,
                    border: `1px solid ${themeColors.border}`,
                    bgcolor: "transparent",
                    cursor: "pointer",
                  }}
                />
                <Button
                  size="small"
                  sx={compactButton}
                  disabled={!canRemoveSelected}
                  onClick={removeSelected}
                >
                  Remove
                </Button>
              </>
            ) : (
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                (none — click a handle to edit; double-click to add; alt/right-click to remove)
              </Typography>
            )}
          </Box>
        </Box>
      )}

      {/* Help footer */}
      <Typography
        sx={{
          ...typography.labelSmall,
          color: themeColors.textMuted,
          mt: `${SPACING.XS}px`,
          opacity: isDark ? 0.8 : 0.7,
        }}
      >
        Drag handles · Double-click empty space to add · Alt/right-click to remove
      </Typography>
    </Box>
  );
}

export const render = createRender(TransferFunctionEditorWidget);
