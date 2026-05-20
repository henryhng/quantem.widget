/**
 * StrainMap2D — 2x2 grid viewer for per-pixel strain tensor channels.
 *
 * Renders e_xx, e_yy, e_xy (top row + bottom-left) and theta (bottom-right)
 * with a shared diverging colormap centered at zero. Includes a nav
 * thumbnail showing the scan-space reference ROI, and controls for
 * percentile clip, ROI numeric inputs, max peak spacing, and explicit
 * "Refit reference" / "Recompute strain" buttons.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap } from "../colormaps";
import { extractFloat32, formatNumber } from "../format";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./strain_map_2d.css";

const DPR = window.devicePixelRatio || 1;
const PANEL_W = 300;
const PANEL_H = 300;
const NAV_W = 200;
const NAV_H = 200;
const SPACING = { XS: 4, SM: 8, MD: 12 };

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ---------------------------------------------------------------------------
// Channel statistics computed locally in JS for live updates
// ---------------------------------------------------------------------------
interface ChannelStats {
  min: number;
  max: number;
  std: number;
  finite: number;
}

function computeChannelStats(arr: Float32Array | null): ChannelStats {
  if (!arr || arr.length === 0) return { min: 0, max: 0, std: 0, finite: 0 };
  let lo = Infinity;
  let hi = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    sumSq += v * v;
    n++;
  }
  if (n === 0) return { min: 0, max: 0, std: 0, finite: 0 };
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  return { min: lo, max: hi, std: Math.sqrt(variance), finite: n };
}

function percentileSymmetric(arr: Float32Array | null, lowPct: number, highPct: number): { vmin: number; vmax: number } {
  if (!arr || arr.length === 0) return { vmin: -1, vmax: 1 };
  const finite: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (Number.isFinite(arr[i])) finite.push(arr[i]);
  }
  if (finite.length === 0) return { vmin: -1, vmax: 1 };
  finite.sort((a, b) => a - b);
  const lo = finite[Math.max(0, Math.floor((lowPct / 100) * (finite.length - 1)))];
  const hi = finite[Math.min(finite.length - 1, Math.ceil((highPct / 100) * (finite.length - 1)))];
  const amax = Math.max(Math.abs(lo), Math.abs(hi), 1e-12);
  return { vmin: -amax, vmax: amax };
}

// ---------------------------------------------------------------------------
// Render a single panel with optional NaN handling
// ---------------------------------------------------------------------------
function drawChannel(
  canvas: HTMLCanvasElement | null,
  data: Float32Array | null,
  w: number,
  h: number,
  cmap: string,
  vmin: number,
  vmax: number,
  scaleFactor: number,
  nanColor: [number, number, number],
) {
  if (!canvas || !data || w === 0 || h === 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const cssW = PANEL_W;
  const cssH = PANEL_H;
  canvas.width = cssW * DPR;
  canvas.height = cssH * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cssW, cssH);

  // Allocate a temp buffer where we replace NaN with vmin so colormap is happy,
  // then we overwrite NaN pixels manually below.
  const scaled = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = data[i] * scaleFactor;
    scaled[i] = Number.isFinite(v) ? v : vmin - 1; // place outside range
  }

  const lut = COLORMAPS[cmap] || COLORMAPS["RdBu"] || COLORMAPS["inferno"];
  const rgba = new Uint8ClampedArray(data.length * 4);
  applyColormap(scaled, rgba, lut, vmin, vmax);

  // Overwrite NaN pixels with nanColor
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      rgba[i * 4 + 0] = nanColor[0];
      rgba[i * 4 + 1] = nanColor[1];
      rgba[i * 4 + 2] = nanColor[2];
      rgba[i * 4 + 3] = 255;
    }
  }

  const offscreen = new OffscreenCanvas(w, h);
  const offCtx = offscreen.getContext("2d");
  if (!offCtx) return;
  const imageData = new ImageData(rgba, w, h);
  offCtx.putImageData(imageData, 0, 0);

  const scaleX = cssW / w;
  const scaleY = cssH / h;
  const fitScale = Math.min(scaleX, scaleY);
  const offX = (cssW - w * fitScale) / 2;
  const offY = (cssH - h * fitScale) / 2;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, offX, offY, w * fitScale, h * fitScale);
}

// ---------------------------------------------------------------------------
// Nav thumbnail with draggable reference ROI overlay
// ---------------------------------------------------------------------------
interface NavThumbnailProps {
  R_Nx: number;
  R_Ny: number;
  mask: Uint8Array | null;
  roi: { top: number; left: number; bottom: number; right: number };
  onRoiChange: (roi: { top: number; left: number; bottom: number; right: number }) => void;
  themeColors: { accent: string; border: string; bgAlt: string };
}

function NavThumbnail({ R_Nx, R_Ny, mask, roi, onRoiChange, themeColors }: NavThumbnailProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const cssW = NAV_W;
  const cssH = NAV_H;

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || R_Nx === 0 || R_Ny === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = cssW * DPR;
    canvas.height = cssH * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.fillStyle = themeColors.bgAlt;
    ctx.fillRect(0, 0, cssW, cssH);

    // Render mask if present
    if (mask) {
      const offscreen = new OffscreenCanvas(R_Ny, R_Nx);
      const offCtx = offscreen.getContext("2d");
      if (offCtx) {
        const rgba = new Uint8ClampedArray(R_Nx * R_Ny * 4);
        for (let i = 0; i < R_Nx * R_Ny; i++) {
          const v = mask[i] ? 200 : 60;
          rgba[i * 4 + 0] = v;
          rgba[i * 4 + 1] = v;
          rgba[i * 4 + 2] = v;
          rgba[i * 4 + 3] = 255;
        }
        const imageData = new ImageData(rgba, R_Ny, R_Nx);
        offCtx.putImageData(imageData, 0, 0);
        const scaleX = cssW / R_Ny;
        const scaleY = cssH / R_Nx;
        const fitScale = Math.min(scaleX, scaleY);
        const offX = (cssW - R_Ny * fitScale) / 2;
        const offY = (cssH - R_Nx * fitScale) / 2;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreen, offX, offY, R_Ny * fitScale, R_Nx * fitScale);
      }
    }

    // Draw ROI rectangle
    const scaleX = cssW / R_Ny;
    const scaleY = cssH / R_Nx;
    const fitScale = Math.min(scaleX, scaleY);
    const offX = (cssW - R_Ny * fitScale) / 2;
    const offY = (cssH - R_Nx * fitScale) / 2;
    const x = offX + roi.left * fitScale;
    const y = offY + roi.top * fitScale;
    const w = (roi.right - roi.left) * fitScale;
    const h = (roi.bottom - roi.top) * fitScale;
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = themeColors.accent + "33";
    ctx.fillRect(x, y, w, h);
  }, [R_Nx, R_Ny, mask, roi, themeColors]);

  // Drag-to-define ROI: pointer down sets one corner, drag sets the other
  const dragStart = React.useRef<{ row: number; col: number } | null>(null);

  const pxToScan = React.useCallback((e: React.PointerEvent): { row: number; col: number } | null => {
    if (R_Nx === 0 || R_Ny === 0) return null;
    const canvas = e.currentTarget as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;
    const scaleX = cssW / R_Ny;
    const scaleY = cssH / R_Nx;
    const fitScale = Math.min(scaleX, scaleY);
    const offX = (cssW - R_Ny * fitScale) / 2;
    const offY = (cssH - R_Nx * fitScale) / 2;
    const col = Math.floor((cssX - offX) / fitScale);
    const row = Math.floor((cssY - offY) / fitScale);
    return {
      row: Math.max(0, Math.min(R_Nx - 1, row)),
      col: Math.max(0, Math.min(R_Ny - 1, col)),
    };
  }, [R_Nx, R_Ny]);

  const onPointerDown = React.useCallback((e: React.PointerEvent) => {
    const p = pxToScan(e);
    if (!p) return;
    dragStart.current = p;
    (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
  }, [pxToScan]);

  const onPointerMove = React.useCallback((e: React.PointerEvent) => {
    if (!dragStart.current) return;
    const p = pxToScan(e);
    if (!p) return;
    const r0 = dragStart.current.row;
    const c0 = dragStart.current.col;
    const r1 = p.row;
    const c1 = p.col;
    const top = Math.min(r0, r1);
    const bottom = Math.max(r0, r1) + 1;
    const left = Math.min(c0, c1);
    const right = Math.max(c0, c1) + 1;
    if (bottom > top && right > left) {
      onRoiChange({ top, left, bottom, right });
    }
  }, [pxToScan, onRoiChange]);

  const onPointerUp = React.useCallback((e: React.PointerEvent) => {
    dragStart.current = null;
    try {
      (e.target as HTMLCanvasElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: cssW, height: cssH, border: `1px solid ${themeColors.border}`, cursor: "crosshair", display: "block" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  );
}

// ---------------------------------------------------------------------------
// Panel: canvas + label + per-channel stats
// ---------------------------------------------------------------------------
interface PanelProps {
  label: string;
  data: Float32Array | null;
  w: number;
  h: number;
  cmap: string;
  vmin: number;
  vmax: number;
  scaleFactor: number;
  unit: string;
  showStats: boolean;
  nanColor: [number, number, number];
  themeColors: { textMuted: string; accent: string; border: string };
}

function Panel({ label, data, w, h, cmap, vmin, vmax, scaleFactor, unit, showStats, nanColor, themeColors }: PanelProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    drawChannel(canvasRef.current, data, w, h, cmap, vmin, vmax, scaleFactor, nanColor);
  }, [data, w, h, cmap, vmin, vmax, scaleFactor, nanColor]);

  const stats = React.useMemo(() => computeChannelStats(data), [data]);
  const displayUnit = unit === "%" && label !== "θ" ? "%" : (label === "θ" ? "rad" : "");

  return (
    <Box>
      <Typography sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px` }}>{label}</Typography>
      <Box sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, width: PANEL_W, height: PANEL_H }}>
        <canvas
          ref={canvasRef}
          style={{ width: PANEL_W, height: PANEL_H, display: "block", imageRendering: "pixelated" }}
        />
        {/* Scalebar: shared diverging colormap range printed on top-right */}
        <Box sx={{
          position: "absolute", top: 3, right: 3,
          bgcolor: "rgba(0,0,0,0.45)", px: 0.5, py: 0.15,
          pointerEvents: "none",
        }}>
          <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.85)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
            ±{formatNumber(vmax * (scaleFactor === 1 ? 1 : 1))}{displayUnit ? ` ${displayUnit}` : ""}
          </Typography>
        </Box>
      </Box>
      {showStats && (
        <Box sx={{ display: "flex", gap: 1.5, mt: `${SPACING.XS}px`, fontSize: 10 }}>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(stats.min * scaleFactor)}</Box></Typography>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(stats.max * scaleFactor)}</Box></Typography>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(stats.std * scaleFactor)}</Box></Typography>
        </Box>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------
function StrainMap2DWidget() {
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = { ...tc };

  const [title] = useModelState<string>("title");
  const [R_Nx] = useModelState<number>("R_Nx");
  const [R_Ny] = useModelState<number>("R_Ny");
  const [cmapStrain, setCmapStrain] = useModelState<string>("cmap_strain");
  const [cmapTheta, setCmapTheta] = useModelState<string>("cmap_theta");
  const [vminPct, setVminPct] = useModelState<number>("vmin_pct");
  const [vmaxPct, setVmaxPct] = useModelState<number>("vmax_pct");
  const [refRoi, setRefRoi] = useModelState<{ top: number; left: number; bottom: number; right: number }>("ref_roi");
  const [maxPeakSpacingPx, setMaxPeakSpacingPx] = useModelState<number>("max_peak_spacing_px");
  const [unit, setUnit] = useModelState<string>("unit");
  const [g1] = useModelState<number[]>("g1");
  const [g2] = useModelState<number[]>("g2");
  const [eXxBytes] = useModelState<DataView | null>("e_xx_bytes");
  const [eYyBytes] = useModelState<DataView | null>("e_yy_bytes");
  const [eXyBytes] = useModelState<DataView | null>("e_xy_bytes");
  const [thetaBytes] = useModelState<DataView | null>("theta_bytes");
  const [maskBytes] = useModelState<DataView | null>("mask_bytes");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  // Custom messages back to Python: anywidget commands
  // We'll piggyback by toggling a "trigger" via TextField/Button changes that
  // Python will observe. Since we don't want hidden traits, we instead rely
  // on the Python side to also expose methods. The UI just nudges the user.

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("StrainMap2D", disabledTools ?? [], hiddenTools ?? []),
    [disabledTools, hiddenTools]
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const lockDisplay = toolVisibility.isLocked("display");
  const hideStatsTool = toolVisibility.isHidden("stats");
  const hideReference = toolVisibility.isHidden("reference");
  const lockReference = toolVisibility.isLocked("reference");
  const hideStrain = toolVisibility.isHidden("strain");
  const lockStrain = toolVisibility.isLocked("strain");

  const eXx = React.useMemo(() => (eXxBytes ? extractFloat32(eXxBytes) : null), [eXxBytes]);
  const eYy = React.useMemo(() => (eYyBytes ? extractFloat32(eYyBytes) : null), [eYyBytes]);
  const eXy = React.useMemo(() => (eXyBytes ? extractFloat32(eXyBytes) : null), [eXyBytes]);
  const theta = React.useMemo(() => (thetaBytes ? extractFloat32(thetaBytes) : null), [thetaBytes]);
  const mask = React.useMemo(() => {
    if (!maskBytes) return null;
    return new Uint8Array(maskBytes.buffer, maskBytes.byteOffset, maskBytes.byteLength);
  }, [maskBytes]);

  const scaleFactor = unit === "%" ? 100 : 1;

  // Compute symmetric percentile range per channel × shared across strain channels
  // (theta has its own range)
  const strainRange = React.useMemo(() => {
    // pool all strain channels for shared range
    const totalLen = (eXx?.length ?? 0) + (eYy?.length ?? 0) + (eXy?.length ?? 0);
    if (totalLen === 0) return { vmin: -0.05, vmax: 0.05 };
    const pool = new Float32Array(totalLen);
    let offset = 0;
    for (const ch of [eXx, eYy, eXy]) {
      if (!ch) continue;
      for (let i = 0; i < ch.length; i++) pool[offset + i] = ch[i] * scaleFactor;
      offset += ch.length;
    }
    return percentileSymmetric(pool, vminPct, vmaxPct);
  }, [eXx, eYy, eXy, vminPct, vmaxPct, scaleFactor]);

  const thetaRange = React.useMemo(() => {
    if (!theta) return { vmin: -0.05, vmax: 0.05 };
    return percentileSymmetric(theta, vminPct, vmaxPct);
  }, [theta, vminPct, vmaxPct]);

  const nanColor: [number, number, number] = themeInfo.theme === "dark" ? [40, 40, 40] : [220, 220, 220];

  // ROI edit handlers
  const updateRoi = React.useCallback((next: { top: number; left: number; bottom: number; right: number }) => {
    if (lockReference) return;
    setRefRoi(next);
  }, [lockReference, setRefRoi]);

  const onRoiField = React.useCallback((key: keyof typeof refRoi, raw: string) => {
    const n = parseInt(raw, 10);
    if (Number.isNaN(n)) return;
    const next = { ...refRoi, [key]: n };
    if (next.bottom > next.top && next.right > next.left && next.top >= 0 && next.left >= 0 && next.bottom <= R_Nx && next.right <= R_Ny) {
      updateRoi(next);
    }
  }, [refRoi, R_Nx, R_Ny, updateRoi]);

  // Themed inputs
  const numericFieldSx = {
    "& .MuiInputBase-input": { fontSize: 10, py: 0.25, color: themeColors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    width: 50,
  };

  const themedSelect = {
    fontSize: 10,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };
  const themedMenuProps = {
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
    ...upwardMenuProps,
  };

  return (
    <Box className="strain-map-2d-root" sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, width: "fit-content" }}>
      {/* Header */}
      <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", lineHeight: "16px" }}>
        {title || "Strain Map"}
        <ControlCustomizer
          widgetName="StrainMap2D"
          hiddenTools={hiddenTools ?? []}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools ?? []}
          setDisabledTools={setDisabledTools}
          themeColors={themeColors}
        />
      </Typography>

      {/* 2x2 grid of panels */}
      <Box sx={{ display: "grid", gridTemplateColumns: `${PANEL_W}px ${PANEL_W}px`, gap: `${SPACING.MD}px` }}>
        <Panel
          label="ε_xx"
          data={eXx}
          w={R_Ny}
          h={R_Nx}
          cmap={cmapStrain}
          vmin={strainRange.vmin}
          vmax={strainRange.vmax}
          scaleFactor={scaleFactor}
          unit={unit}
          showStats={!hideStatsTool && showStats}
          nanColor={nanColor}
          themeColors={themeColors}
        />
        <Panel
          label="ε_yy"
          data={eYy}
          w={R_Ny}
          h={R_Nx}
          cmap={cmapStrain}
          vmin={strainRange.vmin}
          vmax={strainRange.vmax}
          scaleFactor={scaleFactor}
          unit={unit}
          showStats={!hideStatsTool && showStats}
          nanColor={nanColor}
          themeColors={themeColors}
        />
        <Panel
          label="ε_xy"
          data={eXy}
          w={R_Ny}
          h={R_Nx}
          cmap={cmapStrain}
          vmin={strainRange.vmin}
          vmax={strainRange.vmax}
          scaleFactor={scaleFactor}
          unit={unit}
          showStats={!hideStatsTool && showStats}
          nanColor={nanColor}
          themeColors={themeColors}
        />
        <Panel
          label="θ"
          data={theta}
          w={R_Ny}
          h={R_Nx}
          cmap={cmapTheta}
          vmin={thetaRange.vmin}
          vmax={thetaRange.vmax}
          scaleFactor={1}
          unit={unit}
          showStats={!hideStatsTool && showStats}
          nanColor={nanColor}
          themeColors={themeColors}
        />
      </Box>

      {/* Nav thumbnail + reference controls */}
      {showControls && !hideReference && (
        <Box sx={{ mt: `${SPACING.MD}px`, display: "flex", gap: `${SPACING.MD}px`, alignItems: "flex-start" }}>
          <Box>
            <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
              Reference region (drag to define):
            </Typography>
            <NavThumbnail
              R_Nx={R_Nx}
              R_Ny={R_Ny}
              mask={mask}
              roi={refRoi ?? { top: 0, left: 0, bottom: R_Nx, right: R_Ny }}
              onRoiChange={updateRoi}
              themeColors={{ accent: themeColors.accent, border: themeColors.border, bgAlt: themeColors.bgAlt }}
            />
          </Box>

          <Stack direction="column" spacing={`${SPACING.SM}px`} sx={{ flex: 1 }}>
            {/* ROI numeric inputs */}
            <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockReference ? 0.5 : 1, pointerEvents: lockReference ? "none" : "auto" }}>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
              <TextField size="small" label="top" type="number" value={refRoi?.top ?? 0} onChange={(e) => onRoiField("top", e.target.value)} sx={numericFieldSx} InputLabelProps={{ sx: { fontSize: 9 } }} />
              <TextField size="small" label="left" type="number" value={refRoi?.left ?? 0} onChange={(e) => onRoiField("left", e.target.value)} sx={numericFieldSx} InputLabelProps={{ sx: { fontSize: 9 } }} />
              <TextField size="small" label="bottom" type="number" value={refRoi?.bottom ?? 0} onChange={(e) => onRoiField("bottom", e.target.value)} sx={numericFieldSx} InputLabelProps={{ sx: { fontSize: 9 } }} />
              <TextField size="small" label="right" type="number" value={refRoi?.right ?? 0} onChange={(e) => onRoiField("right", e.target.value)} sx={numericFieldSx} InputLabelProps={{ sx: { fontSize: 9 } }} />
            </Box>

            {/* Max peak spacing + g1/g2 readout */}
            <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockStrain ? 0.5 : 1, pointerEvents: lockStrain ? "none" : "auto" }}>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>max Δq:</Typography>
              <TextField
                size="small"
                type="number"
                value={maxPeakSpacingPx}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!Number.isNaN(v) && v > 0) setMaxPeakSpacingPx(v);
                }}
                sx={{ ...numericFieldSx, width: 60 }}
              />
              <Typography sx={typography.value}>
                g1 = ({(g1?.[0] ?? 0).toFixed(2)}, {(g1?.[1] ?? 0).toFixed(2)}) px
              </Typography>
              <Typography sx={typography.value}>
                g2 = ({(g2?.[0] ?? 0).toFixed(2)}, {(g2?.[1] ?? 0).toFixed(2)}) px
              </Typography>
            </Box>

            {/* Display row */}
            {!hideDisplay && (
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>vmin %:</Typography>
                <TextField size="small" type="number" value={vminPct} onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v) && v >= 0 && v < vmaxPct) setVminPct(v); }} sx={numericFieldSx} />
                <Typography sx={{ ...typography.label, fontSize: 10 }}>vmax %:</Typography>
                <TextField size="small" type="number" value={vmaxPct} onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v) && v <= 100 && v > vminPct) setVmaxPct(v); }} sx={numericFieldSx} />
                <Typography sx={{ ...typography.label, fontSize: 10 }}>cmap ε:</Typography>
                <Select size="small" value={cmapStrain} onChange={(e) => setCmapStrain(e.target.value)} sx={{ ...themedSelect, minWidth: 70 }} MenuProps={themedMenuProps}>
                  {Object.keys(COLORMAPS).map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>cmap θ:</Typography>
                <Select size="small" value={cmapTheta} onChange={(e) => setCmapTheta(e.target.value)} sx={{ ...themedSelect, minWidth: 70 }} MenuProps={themedMenuProps}>
                  {Object.keys(COLORMAPS).map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>unit:</Typography>
                <Select size="small" value={unit} onChange={(e) => setUnit(e.target.value)} sx={{ ...themedSelect, minWidth: 60 }} MenuProps={themedMenuProps}>
                  <MenuItem value="strain">strain</MenuItem>
                  <MenuItem value="%">%</MenuItem>
                </Select>
              </Box>
            )}

            {/* Action row — triggers by toggling the trait. Recompute is invoked
                from Python after the user calls w.compute_strain() in a cell;
                the buttons below just nudge a re-sync. */}
            {!hideStrain && (
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Button size="small" sx={compactButton}
                  disabled={lockReference}
                  onClick={() => { if (!lockReference) setRefRoi({ ...(refRoi ?? { top: 0, left: 0, bottom: R_Nx, right: R_Ny }) }); }}
                >
                  Refit reference
                </Button>
                <Button size="small" sx={compactButton}
                  disabled={lockStrain}
                  onClick={() => { if (!lockStrain) setMaxPeakSpacingPx(maxPeakSpacingPx); }}
                >
                  Recompute strain
                </Button>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, ml: 1 }}>
                  Buttons re-emit traits; Python observes and re-runs fits.
                </Typography>
              </Box>
            )}
          </Stack>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(StrainMap2DWidget);
