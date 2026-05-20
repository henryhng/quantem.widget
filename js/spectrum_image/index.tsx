/**
 * SpectrumImage — 2-panel hyperspectral viewer.
 *
 *  Left panel: spatial map (sum/max/argmax/mean over integration window).
 *              Click sets nav_index; a small "+" marker shows the current cell.
 *  Right panel: spectrum at nav_index with an integration window
 *               (two draggable handles + shaded band). When bg_subtract is on,
 *               a dashed background curve and a second pair of handles for
 *               the background fit window are drawn.
 *
 *  Single-file widget per project convention.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import { useTheme } from "../theme";
import { COLORMAPS, renderToOffscreen } from "../colormaps";
import { extractFloat32 } from "../format";
import { computeToolVisibility } from "../tool-parity";

// ─────────────────────────────────────────────────────────────────────────────
// Style constants (per-widget, matching Show3D/Show4DSTEM conventions).
// ─────────────────────────────────────────────────────────────────────────────
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };

const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
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
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const container = {
  root: {
    p: 2,
    bgcolor: "transparent",
    color: "inherit",
    fontFamily: "monospace",
    overflow: "visible",
  },
  imageBox: {
    bgcolor: "#000",
    border: "1px solid #444",
    overflow: "hidden",
    position: "relative" as const,
  },
};

const MAP_SIZE = 360;
const SPEC_WIDTH = 480;
const SPEC_HEIGHT = 360;
const HANDLE_HIT_PX = 8;

const COLORMAP_NAMES = [
  "viridis",
  "plasma",
  "inferno",
  "magma",
  "cividis",
  "gray",
  "hot",
  "cool",
] as const;
const MAP_MODES = ["sum", "max", "argmax", "mean"] as const;

function fmt(value: number): string {
  if (!isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs < 0.001 || abs >= 10000) return value.toExponential(2);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

interface ThemeColors {
  background: string;
  text: string;
  textDim: string;
  border: string;
  accent: string;
  accentBg: string;
  accentGray: string;
}

function getColors(isDark: boolean): ThemeColors {
  return {
    background: isDark ? "#0a0a0a" : "#fafafa",
    text: isDark ? "#ddd" : "#333",
    textDim: isDark ? "#888" : "#666",
    border: isDark ? "#444" : "#bbb",
    accent: "#4fc3f7",
    accentBg: "rgba(79, 195, 247, 0.18)",
    accentGray: isDark ? "#999" : "#777",
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main widget
// ─────────────────────────────────────────────────────────────────────────────
function SpectrumImageView() {
  const { themeInfo } = useTheme();
  const isDark = themeInfo.theme === "dark";
  const colors = React.useMemo(() => getColors(isDark), [isDark]);

  // Sync state from Python
  const [mapBytes] = useModelState<DataView>("map_bytes");
  const [spectrumBytes] = useModelState<DataView>("spectrum_bytes");
  const [bgCurveBytes] = useModelState<DataView>("bg_curve_bytes");
  const [energyAxisBytes] = useModelState<DataView>("energy_axis_bytes");
  const [ny] = useModelState<number>("ny");
  const [nx] = useModelState<number>("nx");
  const [nEnergy] = useModelState<number>("n_energy");

  const [title] = useModelState<string>("title");
  const [energyUnit] = useModelState<string>("energy_unit");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [navIndex, setNavIndex] = useModelState<number[]>("nav_index");
  const [windowEMin, setWindowEMin] = useModelState<number>("window_e_min");
  const [windowEMax, setWindowEMax] = useModelState<number>("window_e_max");
  const [mapMode, setMapMode] = useModelState<string>("map_mode");

  const [bgSubtract, setBgSubtract] = useModelState<boolean>("bg_subtract");
  const [bgEMin, setBgEMin] = useModelState<number>("bg_e_min");
  const [bgEMax, setBgEMax] = useModelState<number>("bg_e_max");
  const [bgParams] = useModelState<number[]>("bg_params");

  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [mapMean] = useModelState<number>("map_stats_mean");
  const [mapMin] = useModelState<number>("map_stats_min");
  const [mapMax] = useModelState<number>("map_stats_max");
  const [mapStd] = useModelState<number>("map_stats_std");
  const [autoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");
  const [logScale] = useModelState<boolean>("log_scale");

  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("SpectrumImage", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hideStats = toolVisibility.isHidden("stats");
  const hideBackground = toolVisibility.isHidden("background");
  const hideWindow = toolVisibility.isHidden("window");
  const lockBackground = toolVisibility.isLocked("background");
  const lockWindow = toolVisibility.isLocked("window");
  const lockDisplay = toolVisibility.isLocked("display");

  // ── Decode arrays ─────────────────────────────────────────────────────────
  const mapData = React.useMemo<Float32Array | null>(() => {
    if (!mapBytes || mapBytes.byteLength < 4) return null;
    return extractFloat32(mapBytes);
  }, [mapBytes]);

  const spectrumData = React.useMemo<Float32Array | null>(() => {
    if (!spectrumBytes || spectrumBytes.byteLength < 4) return null;
    return extractFloat32(spectrumBytes);
  }, [spectrumBytes]);

  const bgCurveData = React.useMemo<Float32Array | null>(() => {
    if (!bgCurveBytes || bgCurveBytes.byteLength < 4) return null;
    return extractFloat32(bgCurveBytes);
  }, [bgCurveBytes]);

  const energyAxis = React.useMemo<Float32Array | null>(() => {
    if (!energyAxisBytes || energyAxisBytes.byteLength < 4) return null;
    return extractFloat32(energyAxisBytes);
  }, [energyAxisBytes]);

  // ── Map canvas rendering ─────────────────────────────────────────────────
  const mapCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const mapOverlayRef = React.useRef<HTMLCanvasElement | null>(null);

  React.useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || !mapData || ny < 1 || nx < 1) return;
    if (mapData.length !== ny * nx) return;

    // Compute vmin/vmax
    let dataForRange: Float32Array = mapData;
    if (logScale) {
      const tmp = new Float32Array(mapData.length);
      for (let i = 0; i < mapData.length; i++) {
        tmp[i] = Math.log1p(Math.max(mapData[i], 0));
      }
      dataForRange = tmp;
    }
    let vmin = Infinity;
    let vmax = -Infinity;
    for (let i = 0; i < dataForRange.length; i++) {
      const v = dataForRange[i];
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
    if (autoContrast && dataForRange.length > 1) {
      // Percentile clip
      const sorted = Float32Array.from(dataForRange).sort();
      const lo = Math.floor((percentileLow / 100) * (sorted.length - 1));
      const hi = Math.floor((percentileHigh / 100) * (sorted.length - 1));
      vmin = sorted[lo];
      vmax = sorted[hi];
    }
    if (!(vmax > vmin)) {
      vmax = vmin + 1e-9;
    }

    const lut = COLORMAPS[cmap] || COLORMAPS["viridis"];
    const offscreen = renderToOffscreen(dataForRange, nx, ny, lut, vmin, vmax);
    if (!offscreen) return;
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0);
  }, [mapData, ny, nx, cmap, autoContrast, percentileLow, percentileHigh, logScale]);

  // ── Overlay (cursor "+" marker) ───────────────────────────────────────────
  React.useEffect(() => {
    const canvas = mapOverlayRef.current;
    if (!canvas) return;
    canvas.width = MAP_SIZE;
    canvas.height = MAP_SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, MAP_SIZE, MAP_SIZE);
    if (ny < 1 || nx < 1) return;
    const r = navIndex && navIndex.length === 2 ? navIndex[0] : 0;
    const c = navIndex && navIndex.length === 2 ? navIndex[1] : 0;
    const px = ((c + 0.5) / nx) * MAP_SIZE;
    const py = ((r + 0.5) / ny) * MAP_SIZE;
    ctx.strokeStyle = "rgba(255, 80, 80, 0.95)";
    ctx.lineWidth = 1.5;
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 2;
    const size = 8;
    ctx.beginPath();
    ctx.moveTo(px - size, py);
    ctx.lineTo(px + size, py);
    ctx.moveTo(px, py - size);
    ctx.lineTo(px, py + size);
    ctx.stroke();
  }, [navIndex, ny, nx]);

  // ── Map click → nav_index ─────────────────────────────────────────────────
  const onMapClick = React.useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (ny < 1 || nx < 1) return;
      const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const col = Math.max(0, Math.min(nx - 1, Math.floor((px / rect.width) * nx)));
      const row = Math.max(0, Math.min(ny - 1, Math.floor((py / rect.height) * ny)));
      setNavIndex([row, col]);
    },
    [ny, nx, setNavIndex],
  );

  // ── Spectrum panel ───────────────────────────────────────────────────────
  // We render the spectrum on a single 2D canvas, then overlay HTML divs as
  // drag handles positioned by left/top in CSS pixels.
  const specCanvasRef = React.useRef<HTMLCanvasElement | null>(null);

  const eMin = energyAxis && energyAxis.length > 0 ? energyAxis[0] : 0;
  const eMax = energyAxis && energyAxis.length > 0 ? energyAxis[energyAxis.length - 1] : 1;
  const eSpan = Math.max(eMax - eMin, 1e-9);

  // Plot region inside the spectrum canvas (leave room for axes)
  const PAD_L = 48;
  const PAD_R = 12;
  const PAD_T = 12;
  const PAD_B = 28;
  const plotW = SPEC_WIDTH - PAD_L - PAD_R;
  const plotH = SPEC_HEIGHT - PAD_T - PAD_B;

  const eToPx = React.useCallback(
    (e: number) => PAD_L + ((e - eMin) / eSpan) * plotW,
    [eMin, eSpan, plotW],
  );
  const pxToE = React.useCallback(
    (px: number) => eMin + ((px - PAD_L) / plotW) * eSpan,
    [eMin, eSpan, plotW],
  );

  // Spectrum y-range
  const { yMin, yMax } = React.useMemo(() => {
    if (!spectrumData || spectrumData.length === 0) {
      return { yMin: 0, yMax: 1 };
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < spectrumData.length; i++) {
      const v = spectrumData[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (bgSubtract && bgCurveData && bgCurveData.length === spectrumData.length) {
      for (let i = 0; i < bgCurveData.length; i++) {
        const v = bgCurveData[i];
        if (isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (!(hi > lo)) hi = lo + 1e-9;
    const pad = (hi - lo) * 0.05;
    return { yMin: lo - pad, yMax: hi + pad };
  }, [spectrumData, bgCurveData, bgSubtract]);

  const yToPx = React.useCallback(
    (y: number) => PAD_T + plotH - ((y - yMin) / Math.max(yMax - yMin, 1e-9)) * plotH,
    [yMin, yMax, plotH],
  );

  React.useEffect(() => {
    const canvas = specCanvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = SPEC_WIDTH * dpr;
    canvas.height = SPEC_HEIGHT * dpr;
    canvas.style.width = `${SPEC_WIDTH}px`;
    canvas.style.height = `${SPEC_HEIGHT}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, SPEC_WIDTH, SPEC_HEIGHT);

    // Background
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, SPEC_WIDTH, SPEC_HEIGHT);

    // Plot frame
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

    // Shaded integration window
    if (energyAxis && energyAxis.length > 1) {
      const w0 = Math.min(windowEMin, windowEMax);
      const w1 = Math.max(windowEMin, windowEMax);
      const x0 = Math.max(PAD_L, Math.min(PAD_L + plotW, eToPx(w0)));
      const x1 = Math.max(PAD_L, Math.min(PAD_L + plotW, eToPx(w1)));
      ctx.fillStyle = colors.accentBg;
      ctx.fillRect(x0, PAD_T, x1 - x0, plotH);

      if (bgSubtract) {
        const b0 = Math.min(bgEMin, bgEMax);
        const b1 = Math.max(bgEMin, bgEMax);
        const bx0 = Math.max(PAD_L, Math.min(PAD_L + plotW, eToPx(b0)));
        const bx1 = Math.max(PAD_L, Math.min(PAD_L + plotW, eToPx(b1)));
        ctx.fillStyle = isDark
          ? "rgba(180,180,180,0.12)"
          : "rgba(100,100,100,0.12)";
        ctx.fillRect(bx0, PAD_T, bx1 - bx0, plotH);
      }
    }

    // Y axis ticks (3)
    ctx.fillStyle = colors.textDim;
    ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let k = 0; k <= 2; k++) {
      const y = yMin + (k / 2) * (yMax - yMin);
      const py = yToPx(y);
      ctx.fillText(fmt(y), PAD_L - 4, py);
      ctx.beginPath();
      ctx.moveTo(PAD_L - 2, py);
      ctx.lineTo(PAD_L, py);
      ctx.stroke();
    }
    // X axis ticks (5)
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    if (energyAxis && energyAxis.length > 1) {
      for (let k = 0; k <= 4; k++) {
        const e = eMin + (k / 4) * eSpan;
        const px = eToPx(e);
        ctx.fillText(fmt(e), px, PAD_T + plotH + 4);
        ctx.beginPath();
        ctx.moveTo(px, PAD_T + plotH);
        ctx.lineTo(px, PAD_T + plotH + 2);
        ctx.stroke();
      }
    }
    // X axis label
    ctx.fillStyle = colors.text;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText(
      `Energy (${energyUnit || ""})`,
      PAD_L + plotW / 2,
      PAD_T + plotH + 14,
    );

    // Spectrum trace
    if (spectrumData && energyAxis && spectrumData.length === energyAxis.length) {
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = 0; i < spectrumData.length; i++) {
        const px = eToPx(energyAxis[i]);
        const py = yToPx(spectrumData[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Background curve
    if (
      bgSubtract &&
      bgCurveData &&
      energyAxis &&
      bgCurveData.length === energyAxis.length
    ) {
      ctx.strokeStyle = colors.accentGray;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < bgCurveData.length; i++) {
        const v = bgCurveData[i];
        if (!isFinite(v)) {
          started = false;
          continue;
        }
        const px = eToPx(energyAxis[i]);
        const py = yToPx(v);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [
    spectrumData,
    bgCurveData,
    energyAxis,
    eMin,
    eSpan,
    yMin,
    yMax,
    eToPx,
    yToPx,
    windowEMin,
    windowEMax,
    bgEMin,
    bgEMax,
    bgSubtract,
    energyUnit,
    colors,
    isDark,
    plotW,
    plotH,
  ]);

  // ── Drag handles for window / bg window ───────────────────────────────────
  type HandleKind = "win-min" | "win-max" | "bg-min" | "bg-max";
  const dragRef = React.useRef<HandleKind | null>(null);

  const onSpecMouseDown = (kind: HandleKind) => (e: React.MouseEvent) => {
    if (kind.startsWith("win") && lockWindow) return;
    if (kind.startsWith("bg") && lockBackground) return;
    e.preventDefault();
    dragRef.current = kind;
    window.addEventListener("mousemove", onWindowMove);
    window.addEventListener("mouseup", onWindowUp);
  };

  const onWindowMove = React.useCallback(
    (e: MouseEvent) => {
      const kind = dragRef.current;
      if (!kind) return;
      const canvas = specCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const eVal = Math.max(eMin, Math.min(eMax, pxToE(px)));
      if (kind === "win-min") {
        setWindowEMin(Math.min(eVal, windowEMax));
      } else if (kind === "win-max") {
        setWindowEMax(Math.max(eVal, windowEMin));
      } else if (kind === "bg-min") {
        setBgEMin(Math.min(eVal, bgEMax));
      } else if (kind === "bg-max") {
        setBgEMax(Math.max(eVal, bgEMin));
      }
    },
    [eMin, eMax, pxToE, setWindowEMin, setWindowEMax, setBgEMin, setBgEMax, windowEMax, windowEMin, bgEMax, bgEMin],
  );

  const onWindowUp = React.useCallback(() => {
    dragRef.current = null;
    window.removeEventListener("mousemove", onWindowMove);
    window.removeEventListener("mouseup", onWindowUp);
  }, [onWindowMove]);

  React.useEffect(() => {
    return () => {
      window.removeEventListener("mousemove", onWindowMove);
      window.removeEventListener("mouseup", onWindowUp);
    };
  }, [onWindowMove, onWindowUp]);

  // Compute handle positions (in CSS pixels)
  const handlePos = (e: number) => Math.max(PAD_L, Math.min(PAD_L + plotW, eToPx(e)));

  const renderHandle = (
    kind: HandleKind,
    eVal: number,
    color: string,
    accentBgVal: string,
  ) => {
    const x = handlePos(eVal);
    return (
      <Box
        key={kind}
        onMouseDown={onSpecMouseDown(kind)}
        sx={{
          position: "absolute",
          left: x - HANDLE_HIT_PX / 2,
          top: PAD_T,
          width: HANDLE_HIT_PX,
          height: plotH,
          cursor: "ew-resize",
          zIndex: 5,
          "&:hover > div": {
            background: accentBgVal,
          },
        }}
        title={`${kind}: ${fmt(eVal)} ${energyUnit}`}
      >
        <Box
          sx={{
            position: "absolute",
            left: HANDLE_HIT_PX / 2 - 1,
            top: 0,
            width: 2,
            height: plotH,
            background: color,
            opacity: 0.85,
          }}
        />
      </Box>
    );
  };

  // ── Numeric inputs ────────────────────────────────────────────────────────
  const [winMinText, setWinMinText] = React.useState("");
  const [winMaxText, setWinMaxText] = React.useState("");
  const [bgMinText, setBgMinText] = React.useState("");
  const [bgMaxText, setBgMaxText] = React.useState("");
  React.useEffect(() => setWinMinText(fmt(windowEMin)), [windowEMin]);
  React.useEffect(() => setWinMaxText(fmt(windowEMax)), [windowEMax]);
  React.useEffect(() => setBgMinText(fmt(bgEMin)), [bgEMin]);
  React.useEffect(() => setBgMaxText(fmt(bgEMax)), [bgEMax]);

  const commit = (txt: string, fallback: number): number => {
    const v = parseFloat(txt);
    if (!isFinite(v)) return fallback;
    return Math.max(eMin, Math.min(eMax, v));
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <Box sx={container.root}>
      {title && (
        <Typography sx={{ ...typography.title, mb: 1, color: colors.text }}>
          {title}
        </Typography>
      )}

      <Stack direction="row" spacing={2} alignItems="flex-start">
        {/* ── Left: spatial map ── */}
        <Box>
          <Box
            onClick={onMapClick}
            sx={{
              ...container.imageBox,
              width: MAP_SIZE,
              height: MAP_SIZE,
              cursor: "crosshair",
              position: "relative",
            }}
          >
            <canvas
              ref={mapCanvasRef}
              style={{
                width: "100%",
                height: "100%",
                imageRendering: "pixelated",
                display: "block",
              }}
            />
            <canvas
              ref={mapOverlayRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
              }}
            />
          </Box>
          {showStats && !hideStats && (
            <Box sx={{ ...controlRow, color: colors.textDim }}>
              <Typography sx={typography.value}>
                mean={fmt(mapMean)} min={fmt(mapMin)} max={fmt(mapMax)} std={fmt(mapStd)}
              </Typography>
            </Box>
          )}
          <Box sx={{ ...controlRow }}>
            <Typography sx={{ ...typography.label, color: colors.text }}>
              cursor:
            </Typography>
            <Typography sx={{ ...typography.value, color: colors.textDim }}>
              row={navIndex?.[0] ?? 0}, col={navIndex?.[1] ?? 0}
            </Typography>
          </Box>
        </Box>

        {/* ── Right: spectrum panel ── */}
        <Box sx={{ position: "relative" }}>
          <Box
            sx={{
              ...container.imageBox,
              width: SPEC_WIDTH,
              height: SPEC_HEIGHT,
              bgcolor: colors.background,
              position: "relative",
            }}
          >
            <canvas
              ref={specCanvasRef}
              style={{ display: "block", width: SPEC_WIDTH, height: SPEC_HEIGHT }}
            />
            {/* Integration window handles */}
            {renderHandle("win-min", windowEMin, colors.accent, colors.accentBg)}
            {renderHandle("win-max", windowEMax, colors.accent, colors.accentBg)}
            {bgSubtract && [
              renderHandle("bg-min", bgEMin, colors.accentGray, "rgba(150,150,150,0.18)"),
              renderHandle("bg-max", bgEMax, colors.accentGray, "rgba(150,150,150,0.18)"),
            ]}
          </Box>
          {showStats && !hideStats && bgSubtract && (
            <Box sx={{ ...controlRow, color: colors.textDim }}>
              <Typography sx={typography.value}>
                bg fit: A={fmt(bgParams?.[0] ?? 0)} r={fmt(bgParams?.[1] ?? 0)}
              </Typography>
            </Box>
          )}
        </Box>
      </Stack>

      {/* ── Controls ── */}
      {showControls && (
        <Stack
          direction="row"
          spacing={2}
          alignItems="center"
          flexWrap="wrap"
          sx={{ mt: 1, gap: 1 }}
        >
          {!hideDisplay && (
            <Box sx={controlRow}>
              <Typography sx={{ ...typography.label, color: colors.text }}>
                cmap:
              </Typography>
              <Select
                size="small"
                value={cmap}
                disabled={lockDisplay}
                onChange={(e) => setCmap(String(e.target.value))}
                sx={{ fontSize: 11, height: 26 }}
              >
                {COLORMAP_NAMES.map((c) => (
                  <MenuItem key={c} value={c} sx={{ fontSize: 11 }}>
                    {c}
                  </MenuItem>
                ))}
              </Select>
            </Box>
          )}

          {!hideWindow && (
            <Box sx={controlRow}>
              <Typography sx={{ ...typography.label, color: colors.text }}>
                mode:
              </Typography>
              <Select
                size="small"
                value={mapMode}
                disabled={lockWindow}
                onChange={(e) => setMapMode(String(e.target.value))}
                sx={{ fontSize: 11, height: 26 }}
              >
                {MAP_MODES.map((m) => (
                  <MenuItem key={m} value={m} sx={{ fontSize: 11 }}>
                    {m}
                  </MenuItem>
                ))}
              </Select>
            </Box>
          )}

          {!hideWindow && (
            <Box sx={controlRow}>
              <Typography sx={{ ...typography.label, color: colors.text }}>
                window ({energyUnit}):
              </Typography>
              <TextField
                size="small"
                value={winMinText}
                disabled={lockWindow}
                onChange={(e) => setWinMinText(e.target.value)}
                onBlur={() => setWindowEMin(commit(winMinText, windowEMin))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    setWindowEMin(commit(winMinText, windowEMin));
                  }
                }}
                inputProps={{
                  style: { fontSize: 11, padding: "2px 4px", width: 64 },
                }}
              />
              <Typography sx={{ ...typography.label, color: colors.textDim }}>
                –
              </Typography>
              <TextField
                size="small"
                value={winMaxText}
                disabled={lockWindow}
                onChange={(e) => setWinMaxText(e.target.value)}
                onBlur={() => setWindowEMax(commit(winMaxText, windowEMax))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    setWindowEMax(commit(winMaxText, windowEMax));
                  }
                }}
                inputProps={{
                  style: { fontSize: 11, padding: "2px 4px", width: 64 },
                }}
              />
            </Box>
          )}

          {!hideBackground && (
            <Box sx={controlRow}>
              <Typography sx={{ ...typography.label, color: colors.text }}>
                bg subtract:
              </Typography>
              <Switch
                size="small"
                checked={bgSubtract}
                disabled={lockBackground}
                onChange={(e) => setBgSubtract(e.target.checked)}
                sx={switchStyles.small}
              />
              <Typography sx={{ ...typography.label, color: colors.text }}>
                bg ({energyUnit}):
              </Typography>
              <TextField
                size="small"
                value={bgMinText}
                disabled={lockBackground || !bgSubtract}
                onChange={(e) => setBgMinText(e.target.value)}
                onBlur={() => setBgEMin(commit(bgMinText, bgEMin))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    setBgEMin(commit(bgMinText, bgEMin));
                  }
                }}
                inputProps={{
                  style: { fontSize: 11, padding: "2px 4px", width: 64 },
                }}
              />
              <Typography sx={{ ...typography.label, color: colors.textDim }}>
                –
              </Typography>
              <TextField
                size="small"
                value={bgMaxText}
                disabled={lockBackground || !bgSubtract}
                onChange={(e) => setBgMaxText(e.target.value)}
                onBlur={() => setBgEMax(commit(bgMaxText, bgEMax))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    setBgEMax(commit(bgMaxText, bgEMax));
                  }
                }}
                inputProps={{
                  style: { fontSize: 11, padding: "2px 4px", width: 64 },
                }}
              />
            </Box>
          )}
        </Stack>
      )}

      {/* Energy range label */}
      <Typography sx={{ ...typography.value, color: colors.textDim, mt: 0.5 }}>
        {ny}×{nx} pixels, {nEnergy} energy bins · range [{fmt(eMin)}, {fmt(eMax)}] {energyUnit}
      </Typography>
    </Box>
  );
}

export default { render: createRender(SpectrumImageView) };
