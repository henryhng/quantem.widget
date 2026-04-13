/**
 * Live — Real-time incremental image viewer.
 *
 * Layout: filmstrip sidebar (2-col thumbnail grid) on left + main inspect panel on right.
 * Push model: frames arrive via _push_bytes / _batch_bytes, newest frame auto-selected.
 * GPU colormap engine (zero-copy ImageBitmap path) used for main canvas when available.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI } from "../scalebar";
import {
  COLORMAPS,
  COLORMAP_NAMES,
  renderToOffscreen,
  renderToOffscreenReuse,
  GPUColormapEngine,
  getGPUColormapEngine,
} from "../colormaps";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import {
  findDataRange,
  applyLogScale,
  percentileClip,
  sliderRange,
  computeStats,
} from "../stats";
import "./live.css";

// ============================================================================
// Design constants — exact copies from Show2D/Show3D
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const MIN_CANVAS = 256;
const THUMB_SIZE = 64;
const THUMB_COLS = 2;

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ============================================================================
// InfoTooltip — exact copy from Show2D
// ============================================================================
function InfoTooltip({
  text,
  theme = "dark",
}: {
  text: React.ReactNode;
  theme?: "light" | "dark";
}) {
  const isDark = theme === "dark";
  const content =
    typeof text === "string" ? (
      <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    ) : (
      text
    );
  return (
    <Tooltip
      title={content}
      arrow
      placement="bottom"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: isDark ? "#333" : "#fff",
            color: isDark ? "#ddd" : "#333",
            border: `1px solid ${isDark ? "#555" : "#ccc"}`,
            maxWidth: 280,
            p: 1,
          },
        },
        arrow: {
          sx: {
            color: isDark ? "#333" : "#fff",
            "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` },
          },
        },
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 12,
          color: isDark ? "#888" : "#666",
          cursor: "help",
          ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
        }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

// ============================================================================
// Histogram — exact copy from Show2D (GPU precomputed bins supported)
// ============================================================================
interface HistogramProps {
  data: Float32Array | null;
  precomputedBins?: number[] | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

function Histogram({
  data,
  precomputedBins,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 110,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const cpuBins = React.useMemo(
    () => (precomputedBins ? null : computeHistogramFromBytes(data)),
    [data, precomputedBins],
  );
  const bins = precomputedBins || cpuBins || new Array(256).fill(0);
  const isDark = theme === "dark";
  const colors = isDark
    ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle =
        i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(
        i * barWidth + 0.5,
        height - barHeight,
        Math.max(1, barWidth - 1),
        barHeight,
      );
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(
            Math.min(newMin, newMax - 1),
            Math.max(newMax, newMin + 1),
          );
        }}
        min={0}
        max={100}
        size="small"
        valueLabelDisplay="auto"
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
        }}
        sx={{
          width,
          py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography
          sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}
        >
          {(() => {
            const v = dataMin + (vminPct / 100) * (dataMax - dataMin);
            return v >= 1000 ? v.toExponential(1) : v.toFixed(1);
          })()}
        </Typography>
        <Typography
          sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}
        >
          {(() => {
            const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin);
            return v >= 1000 ? v.toExponential(1) : v.toFixed(1);
          })()}
        </Typography>
      </Box>
    </Box>
  );
}

// ============================================================================
// Types
// ============================================================================
interface FrameEntry {
  data: Float32Array;
  height: number;
  width: number;
  label: string;
  thumbCanvas: HTMLCanvasElement | null;
}

// ============================================================================
// Main Component
// ============================================================================
const render = createRender(() => {
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = {
    ...tc,
    accentGreen: themeInfo.theme === "dark" ? "#66bb6a" : "#2e7d32",
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
    ...upwardMenuProps,
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
      },
    },
  };

  // ─── Model state ─────────────────────────────────────────────────
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [bufferSize] = useModelState<number>("buffer_size");
  const [nFrames] = useModelState<number>("n_frames");
  const [showControls] = useModelState<boolean>("show_controls");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [showStats] = useModelState<boolean>("show_stats");

  // Push / Batch traits
  const [pushBytes] = useModelState<DataView>("_push_bytes");
  const [pushHeight] = useModelState<number>("_push_height");
  const [pushWidth] = useModelState<number>("_push_width");
  const [pushLabel] = useModelState<string>("_push_label");
  const [pushCounter] = useModelState<number>("_push_counter");
  const [batchBytes] = useModelState<DataView>("_batch_bytes");
  const [batchDims] = useModelState<number[][]>("_batch_dims");
  const [batchLabels] = useModelState<string[]>("_batch_labels");
  const [batchCounter] = useModelState<number>("_batch_counter");

  // ─── Local state ─────────────────────────────────────────────────
  const framesRef = React.useRef<FrameEntry[]>([]);
  const [frameCount, setFrameCount] = React.useState(0);
  const [selectedIdx, setSelectedIdx] = React.useState(0);
  const [canvasSize, setCanvasSize] = React.useState(500);

  // Histogram / contrast
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);
  const vminPctRef = React.useRef(0);
  const vmaxPctRef = React.useRef(100);
  const sliderRafRef = React.useRef(0);
  const [histData, setHistData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{
    min: number;
    max: number;
  }>({ min: 0, max: 1 });

  // Stats
  const [statsMean, setStatsMean] = React.useState(0);
  const [statsMin, setStatsMin] = React.useState(0);
  const [statsMax, setStatsMax] = React.useState(0);
  const [statsStd, setStatsStd] = React.useState(0);

  // Cursor
  const [cursorInfo, setCursorInfo] = React.useState<{
    row: number;
    col: number;
    value: number;
  } | null>(null);

  // Zoom / pan
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const zoomRef = React.useRef(1);
  const panXRef = React.useRef(0);
  const panYRef = React.useRef(0);
  zoomRef.current = zoom;
  panXRef.current = panX;
  panYRef.current = panY;
  const panStartRef = React.useRef<{
    x: number;
    y: number;
    pX: number;
    pY: number;
  } | null>(null);
  const dragRef = React.useRef({ wasDrag: false });

  // Canvas resize
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const resizeStartRef = React.useRef<{
    x: number;
    y: number;
    size: number;
  } | null>(null);

  // Refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const thumbListRef = React.useRef<HTMLDivElement>(null);
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);

  // Ref mirrors for async callbacks
  const logScaleRef = React.useRef(logScale);
  logScaleRef.current = logScale;
  const cmapRef = React.useRef(cmap);
  cmapRef.current = cmap;
  const autoContrastRef = React.useRef(autoContrast);
  autoContrastRef.current = autoContrast;
  const selectedIdxRef = React.useRef(selectedIdx);
  selectedIdxRef.current = selectedIdx;
  const dataRangeRef = React.useRef<{ min: number; max: number }>({
    min: 0,
    max: 1,
  });

  // ─── GPU colormap engine ─────────────────────────────────────────
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);

  React.useEffect(() => {
    getGPUColormapEngine().then((engine) => {
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        console.log("[Live] WebGPU colormap engine initialized");
      } else {
        console.log("[Live] WebGPU unavailable — using CPU fallback");
      }
    });
  }, []);

  // ─── Watching indicator: true when new frame arrived in last 3s ──
  const [isWatching, setIsWatching] = React.useState(false);
  const watchTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const markWatching = React.useCallback(() => {
    setIsWatching(true);
    if (watchTimerRef.current) clearTimeout(watchTimerRef.current);
    watchTimerRef.current = setTimeout(() => setIsWatching(false), 3000);
  }, []);

  // ─── Build thumbnail canvas for a frame ─────────────────────────
  const buildThumb = React.useCallback(
    (data: Float32Array, w: number, h: number): HTMLCanvasElement => {
      const lut = COLORMAPS[cmapRef.current] || COLORMAPS.inferno;
      const range = findDataRange(data);
      const src = logScaleRef.current ? applyLogScale(data) : data;
      const logRange = logScaleRef.current
        ? {
            min: Math.log1p(Math.max(range.min, 0)),
            max: Math.log1p(Math.max(range.max, 0)),
          }
        : range;
      return (
        renderToOffscreen(src, w, h, lut, logRange.min, logRange.max) ||
        (() => {
          const c = document.createElement("canvas");
          c.width = w;
          c.height = h;
          return c;
        })()
      );
    },
    [],
  );

  // ─── Accumulate single-push frames ──────────────────────────────
  const prevPushCounterRef = React.useRef(0);
  React.useEffect(() => {
    if (pushCounter === prevPushCounterRef.current) return;
    prevPushCounterRef.current = pushCounter;
    if (!pushBytes || pushHeight === 0 || pushWidth === 0) return;
    const floats = extractFloat32(pushBytes);
    if (!floats || floats.length !== pushHeight * pushWidth) return;
    const data = new Float32Array(floats);
    const thumbCanvas = buildThumb(data, pushWidth, pushHeight);
    const entry: FrameEntry = {
      data,
      height: pushHeight,
      width: pushWidth,
      label: pushLabel || `Frame ${framesRef.current.length + 1}`,
      thumbCanvas,
    };
    // Newest frame at index 0 (top of list)
    framesRef.current = [entry, ...framesRef.current];
    if (framesRef.current.length > bufferSize) {
      framesRef.current = framesRef.current.slice(0, bufferSize);
    }
    setFrameCount(framesRef.current.length);
    setSelectedIdx(0);
    if (thumbListRef.current) thumbListRef.current.scrollTop = 0;
    markWatching();
  }, [pushCounter]);

  // ─── Accumulate batch frames ─────────────────────────────────────
  const prevBatchCounterRef = React.useRef(0);
  React.useEffect(() => {
    if (batchCounter === prevBatchCounterRef.current) return;
    prevBatchCounterRef.current = batchCounter;
    if (!batchBytes || !batchDims || batchDims.length === 0) return;
    const allFloats = extractFloat32(batchBytes);
    if (!allFloats || allFloats.length === 0) return;
    const newEntries: FrameEntry[] = [];
    let offset = 0;
    for (let i = 0; i < batchDims.length; i++) {
      const [h, w] = batchDims[i];
      const count = h * w;
      if (offset + count > allFloats.length) break;
      const slice = allFloats.subarray(offset, offset + count);
      const data = new Float32Array(slice);
      const thumbCanvas = buildThumb(data, w, h);
      const label =
        batchLabels && i < batchLabels.length ? batchLabels[i] : `Frame ${i + 1}`;
      newEntries.push({ data, height: h, width: w, label, thumbCanvas });
      offset += count;
    }
    // Prepend newest-first (batch arrives oldest→newest, reverse so [0]=newest)
    newEntries.reverse();
    framesRef.current = [...newEntries, ...framesRef.current];
    if (framesRef.current.length > bufferSize) {
      framesRef.current = framesRef.current.slice(0, bufferSize);
    }
    setFrameCount(framesRef.current.length);
    setSelectedIdx(0);
    if (thumbListRef.current) thumbListRef.current.scrollTop = 0;
    markWatching();
  }, [batchCounter]);

  // ─── Displayed frame ────────────────────────────────────────────
  const displayIdx = Math.min(selectedIdx, Math.max(0, framesRef.current.length - 1));
  const displayFrame =
    frameCount > 0 && framesRef.current.length > displayIdx
      ? framesRef.current[displayIdx]
      : null;

  // Layout: compute canvas display dimensions preserving aspect ratio
  const displayW = displayFrame ? displayFrame.width : canvasSize;
  const displayH = displayFrame ? displayFrame.height : canvasSize;
  const displayScale =
    canvasSize / Math.max(displayW, displayH, 1);
  const canvasW = Math.round(displayW * displayScale);
  const canvasH = Math.round(displayH * displayScale);

  // ─── Compute stats + histogram on frame/display change ──────────
  React.useEffect(() => {
    if (!displayFrame) return;
    const raw = displayFrame.data;
    const rawRange = findDataRange(raw);
    const processed = logScale ? applyLogScale(raw) : raw;
    const logRange = logScale
      ? {
          min: Math.log1p(Math.max(rawRange.min, 0)),
          max: Math.log1p(Math.max(rawRange.max, 0)),
        }
      : rawRange;
    setImageDataRange(logRange);
    dataRangeRef.current = logRange;
    setHistData(processed);

    // Stats
    const s = computeStats(raw);
    setStatsMean(s.mean);
    setStatsMin(s.min);
    setStatsMax(s.max);
    setStatsStd(s.std);
  }, [displayIdx, frameCount, logScale]);

  // ─── Initialize / resize offscreen canvas ───────────────────────
  React.useEffect(() => {
    if (!displayFrame) return;
    const { width, height } = displayFrame;
    if (
      !mainOffscreenRef.current ||
      mainOffscreenRef.current.width !== width ||
      mainOffscreenRef.current.height !== height
    ) {
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      mainOffscreenRef.current = c;
      mainImgDataRef.current = c.getContext("2d")!.createImageData(width, height);
    }
  }, [displayIdx, frameCount]);

  // ─── GPU upload + render offscreen ───────────────────────────────
  const renderOffscreen = React.useCallback(() => {
    if (!displayFrame) return;
    const { data, width, height } = displayFrame;
    const lut = COLORMAPS[cmapRef.current] || COLORMAPS.inferno;
    const rawRange = findDataRange(data);
    const logRange = {
      min: Math.log1p(Math.max(rawRange.min, 0)),
      max: Math.log1p(Math.max(rawRange.max, 0)),
    };
    const range = logScaleRef.current ? logRange : rawRange;

    let vmin: number, vmax: number;
    if (autoContrastRef.current) {
      const processed = logScaleRef.current ? applyLogScale(data) : data;
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else if (vminPctRef.current > 0 || vmaxPctRef.current < 100) {
      ({ vmin, vmax } = sliderRange(
        range.min,
        range.max,
        vminPctRef.current,
        vmaxPctRef.current,
      ));
    } else {
      vmin = range.min;
      vmax = range.max;
    }

    // Ensure offscreen
    if (
      !mainOffscreenRef.current ||
      mainOffscreenRef.current.width !== width ||
      mainOffscreenRef.current.height !== height
    ) {
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      mainOffscreenRef.current = c;
      mainImgDataRef.current = c.getContext("2d")!.createImageData(width, height);
    }

    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      engine.uploadData(0, data, width, height);
      engine.uploadLUT(cmapRef.current, lut);
      const bitmaps = engine.renderSlotsToImageBitmap(
        [0],
        [{ vmin, vmax }],
        logScaleRef.current,
      );
      if (bitmaps && bitmaps[0]) {
        mainOffscreenRef.current.getContext("2d")!.drawImage(bitmaps[0], 0, 0);
        setOffscreenVersion((v) => v + 1);
        return;
      }
      // Fallback within GPU path
    }
    // CPU path
    const processed = logScaleRef.current ? applyLogScale(data) : data;
    if (mainOffscreenRef.current && mainImgDataRef.current) {
      renderToOffscreenReuse(
        processed,
        lut,
        vmin,
        vmax,
        mainOffscreenRef.current,
        mainImgDataRef.current,
      );
    }
    setOffscreenVersion((v) => v + 1);
  }, [displayIdx, frameCount]);

  // Re-render offscreen when display/contrast params change
  React.useEffect(() => {
    renderOffscreen();
  }, [displayIdx, frameCount, cmap, logScale, autoContrast, vminPct, vmaxPct]);

  // ─── Fast slider path (bypass React effect batching) ─────────────
  const handleRangeChange = React.useCallback(
    (newMin: number, newMax: number) => {
      vminPctRef.current = newMin;
      vmaxPctRef.current = newMax;
      setVminPct(newMin);
      setVmaxPct(newMax);
      // Fire on next rAF — coalesces rapid drag events
      cancelAnimationFrame(sliderRafRef.current);
      sliderRafRef.current = requestAnimationFrame(() => {
        renderOffscreen();
      });
    },
    [renderOffscreen],
  );

  // ─── Draw effect: zoom/pan → cheap drawImage from cached offscreen ─
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const offscreen = mainOffscreenRef.current;
    if (!canvas || !offscreen || !displayFrame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (zoom !== 1 || panX !== 0 || panY !== 0) {
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(
        offscreen,
        0,
        0,
        displayFrame.width,
        displayFrame.height,
        0,
        0,
        canvasW,
        canvasH,
      );
      ctx.restore();
    } else {
      ctx.drawImage(
        offscreen,
        0,
        0,
        displayFrame.width,
        displayFrame.height,
        0,
        0,
        canvasW,
        canvasH,
      );
    }
  }, [offscreenVersion, canvasW, canvasH, zoom, panX, panY, displayFrame]);

  // ─── Scale bar overlay (HiDPI UI canvas) ────────────────────────
  React.useEffect(() => {
    const uiCanvas = overlayRef.current;
    if (!uiCanvas || !displayFrame) return;
    const ctx = uiCanvas.getContext("2d");
    if (!ctx) return;
    uiCanvas.width = Math.round(canvasW * DPR);
    uiCanvas.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
    if (pixelSize > 0) {
      const unit = pixelSize > 0 ? ("Å" as const) : ("px" as const);
      drawScaleBarHiDPI(uiCanvas, DPR, zoom, pixelSize, unit, displayFrame.width);
    }
  }, [canvasW, canvasH, zoom, pixelSize, displayFrame]);

  // ─── Prevent page scroll on canvas ──────────────────────────────
  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []);

  // ─── Zoom (scroll wheel, cursor-centered) ───────────────────────
  const handleWheel = React.useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const rect = canvasContainerRef.current!.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const nz = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));
      setPanX((prev) => mx - (mx - prev) * (nz / zoom));
      setPanY((prev) => my - (my - prev) * (nz / zoom));
      setZoom(nz);
    },
    [zoom],
  );

  // ─── Pan (click + drag) ──────────────────────────────────────────
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      dragRef.current = { wasDrag: false };
      panStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        pX: panXRef.current,
        pY: panYRef.current,
      };
      const onMove = (ev: MouseEvent) => {
        if (!panStartRef.current) return;
        const dx = ev.clientX - panStartRef.current.x;
        const dy = ev.clientY - panStartRef.current.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragRef.current.wasDrag = true;
        setPanX(panStartRef.current.pX + dx);
        setPanY(panStartRef.current.pY + dy);
      };
      const onUp = () => {
        panStartRef.current = null;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [],
  );

  // ─── Cursor readout ──────────────────────────────────────────────
  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      if (!displayFrame) {
        setCursorInfo(null);
        return;
      }
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (canvasW / rect.width);
      const my = (e.clientY - rect.top) * (canvasH / rect.height);
      // Invert zoom/pan transform (center-based)
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imgX = ((mx - cx - panX) / zoom + cx) / displayScale;
      const imgY = ((my - cy - panY) / zoom + cy) / displayScale;
      const imgCol = Math.floor(imgX);
      const imgRow = Math.floor(imgY);
      if (
        imgCol >= 0 &&
        imgCol < displayFrame.width &&
        imgRow >= 0 &&
        imgRow < displayFrame.height
      ) {
        const value = displayFrame.data[imgRow * displayFrame.width + imgCol];
        setCursorInfo({ row: imgRow, col: imgCol, value });
      } else {
        setCursorInfo(null);
      }
    },
    [displayFrame, canvasW, canvasH, zoom, panX, panY, displayScale],
  );

  // ─── Reset zoom ──────────────────────────────────────────────────
  const resetZoom = React.useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // ─── Copy to clipboard ───────────────────────────────────────────
  const handleCopy = React.useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/png"),
      );
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      canvas.toBlob(
        (b) => {
          if (b)
            downloadBlob(
              b,
              `live_${displayFrame?.label ?? "frame"}.png`,
            );
        },
        "image/png",
      );
    }
  }, [displayFrame]);

  // ─── Canvas resize handle ────────────────────────────────────────
  const handleResizeMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsResizingCanvas(true);
      resizeStartRef.current = { x: e.clientX, y: e.clientY, size: canvasSize };
    },
    [canvasSize],
  );

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeStartRef.current ? resizeStartRef.current.size : canvasSize;

    const onMove = (e: MouseEvent) => {
      if (!resizeStartRef.current) return;
      const delta = Math.max(
        e.clientX - resizeStartRef.current.x,
        e.clientY - resizeStartRef.current.y,
      );
      latestSize = Math.max(MIN_CANVAS, resizeStartRef.current.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasSize(latestSize);
        });
      }
    };
    const onUp = () => {
      cancelAnimationFrame(rafId);
      setCanvasSize(latestSize);
      setIsResizingCanvas(false);
      resizeStartRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [isResizingCanvas]);

  // ─── Keyboard shortcuts ──────────────────────────────────────────
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT"
      )
        return;
      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          e.stopPropagation();
          setSelectedIdx((prev) =>
            Math.min(prev + 1, framesRef.current.length - 1),
          );
          break;
        case "ArrowRight":
          e.preventDefault();
          e.stopPropagation();
          setSelectedIdx((prev) => Math.max(prev - 1, 0));
          break;
        case "r":
        case "R":
          e.preventDefault();
          resetZoom();
          break;
        default:
          break;
      }
    },
    [resetZoom],
  );

  // ─── Derived display values ──────────────────────────────────────
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const displayLabel = displayFrame?.label ?? "—";
  const displayDims = displayFrame
    ? `${displayFrame.height}×${displayFrame.width}`
    : "—";
  const calibratedUnit =
    pixelSize > 0
      ? Math.max(displayH, displayW) * pixelSize >= 10
        ? "nm"
        : "Å"
      : "";
  const calibratedFactor =
    calibratedUnit === "nm" ? pixelSize / 10 : pixelSize;

  // Thumb panel width: based on 2 thumbnails + gap + padding
  const thumbPanelWidth = THUMB_COLS * THUMB_SIZE + (THUMB_COLS - 1) * SPACING.XS + 2 * SPACING.SM + 2;

  // ─── Render ──────────────────────────────────────────────────────
  return (
    <Box
      ref={rootRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={() => rootRef.current?.focus()}
      sx={{
        p: 2,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        width: "fit-content",
        outline: "none",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      {/* ── Title row ─────────────────────────────────────────────── */}
      <Stack
        direction="row"
        alignItems="center"
        spacing={`${SPACING.SM}px`}
        sx={{ mb: `${SPACING.XS}px`, height: 16 }}
      >
        <Typography
          variant="caption"
          sx={{
            ...typography.label,
            color: themeColors.accent,
            lineHeight: "16px",
            overflow: "hidden",
          }}
        >
          {title || "Live"}
        </Typography>
        <InfoTooltip
          text={
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>
                Live Controls
              </Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                Frames are pushed in real-time from Python. Newest frame is
                always shown first (index 0).
              </Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>
                Keyboard
              </Typography>
              <Box
                component="table"
                sx={{
                  borderCollapse: "collapse",
                  "& td": {
                    py: 0.25,
                    fontSize: 11,
                    lineHeight: 1.3,
                    verticalAlign: "top",
                  },
                  "& td:first-of-type": {
                    pr: 1.5,
                    opacity: 0.7,
                    fontFamily: "monospace",
                    fontSize: 10,
                    whiteSpace: "nowrap",
                  },
                }}
              >
                <tbody>
                  {(
                    [
                      ["← / →", "Older / newer frame"],
                      ["R", "Reset zoom"],
                      ["Scroll", "Zoom in/out"],
                      ["Dbl-click", "Reset view"],
                    ] as [string, string][]
                  ).map(([k, d], i) => (
                    <tr key={i}>
                      <td>{k}</td>
                      <td>{d}</td>
                    </tr>
                  ))}
                </tbody>
              </Box>
            </Box>
          }
          theme={themeInfo.theme}
        />
        <Box sx={{ flex: 1 }} />
        {/* Frame count */}
        <Typography
          sx={{
            ...typography.labelSmall,
            color: themeColors.textMuted,
            fontFamily: "monospace",
          }}
        >
          {frameCount} frame{frameCount !== 1 ? "s" : ""}
          {nFrames > frameCount ? ` / ${nFrames}` : ""}
        </Typography>
        {/* LIVE indicator */}
        {isWatching && (
          <Typography
            sx={{
              fontSize: 10,
              fontFamily: "monospace",
              color: themeColors.accentGreen,
              fontWeight: "bold",
              "@keyframes livePulse": {
                "0%,100%": { opacity: 1 },
                "50%": { opacity: 0.4 },
              },
              animation: "livePulse 1.2s ease-in-out infinite",
            }}
          >
            ● LIVE
          </Typography>
        )}
      </Stack>

      {/* ── Body: filmstrip + inspect panel ───────────────────────── */}
      <Stack direction="row" spacing={`${SPACING.LG}px`} alignItems="flex-start">

        {/* ══ LEFT: Filmstrip ══════════════════════════════════════ */}
        <Box sx={{ width: thumbPanelWidth, flexShrink: 0 }}>
          <Box
            ref={thumbListRef}
            sx={{
              border: `1px solid ${themeColors.border}`,
              bgcolor: themeColors.controlBg,
              p: `${SPACING.SM}px`,
              maxHeight: canvasH + 150,
              overflowY: "auto",
              overflowX: "hidden",
            }}
          >
            <Typography
              sx={{
                ...typography.labelSmall,
                color: themeColors.textMuted,
                mb: `${SPACING.XS}px`,
                display: "block",
              }}
            >
              Frames
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: `repeat(${THUMB_COLS}, ${THUMB_SIZE}px)`,
                gap: `${SPACING.XS}px`,
              }}
            >
              {framesRef.current.slice(0, frameCount).map((frame, i) => (
                <Box
                  key={`thumb-${frameCount}-${i}`}
                  onClick={() => setSelectedIdx(i)}
                  sx={{
                    border: `2px solid ${i === selectedIdx ? themeColors.accent : "transparent"}`,
                    cursor: "pointer",
                    opacity: i === selectedIdx ? 1 : 0.65,
                    "&:hover": { opacity: 1 },
                    position: "relative",
                  }}
                >
                  {frame.thumbCanvas ? (
                    <canvas
                      ref={(el) => {
                        if (el && frame.thumbCanvas) {
                          el.width = THUMB_SIZE;
                          el.height = THUMB_SIZE;
                          const ctx = el.getContext("2d");
                          if (ctx) {
                            ctx.imageSmoothingEnabled = true;
                            ctx.drawImage(
                              frame.thumbCanvas,
                              0,
                              0,
                              el.width,
                              el.height,
                            );
                          }
                        }
                      }}
                      style={{
                        width: THUMB_SIZE,
                        height: THUMB_SIZE,
                        display: "block",
                        imageRendering: "auto",
                      }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: THUMB_SIZE,
                        height: THUMB_SIZE,
                        bgcolor: themeColors.border,
                      }}
                    />
                  )}
                  <Typography
                    sx={{
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      right: 0,
                      fontSize: 8,
                      textAlign: "center",
                      bgcolor: "rgba(0,0,0,0.55)",
                      color: "#fff",
                      lineHeight: 1.3,
                      px: 0.25,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {frame.label}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </Box>

        {/* ══ RIGHT: Inspect Panel ═════════════════════════════════ */}
        <Box
          sx={{
            width: canvasW,
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            gap: 0,
          }}
        >
          {/* Controls row: Reset + Copy + Export */}
          <Stack
            direction="row"
            alignItems="center"
            spacing={`${SPACING.SM}px`}
            sx={{ mb: `${SPACING.XS}px`, height: 28 }}
          >
            <Box sx={{ flex: 1 }} />
            <Button
              size="small"
              sx={compactButton}
              disabled={!needsReset}
              onClick={resetZoom}
            >
              Reset
            </Button>
            <Button size="small" sx={compactButton} onClick={handleCopy}>
              Copy
            </Button>
          </Stack>

          {/* Canvas */}
          <Box
            ref={canvasContainerRef}
            sx={{
              position: "relative",
              bgcolor: "#000",
              border: `1px solid ${themeColors.border}`,
              width: canvasW,
              height: canvasH,
              cursor: "grab",
              overflow: "hidden",
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setCursorInfo(null)}
            onWheel={handleWheel}
            onDoubleClick={resetZoom}
          >
            {frameCount === 0 ? (
              <Box
                sx={{
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Typography
                  sx={{ ...typography.label, color: themeColors.textMuted }}
                >
                  Waiting for data…
                </Typography>
              </Box>
            ) : (
              <canvas
                ref={canvasRef}
                width={canvasW}
                height={canvasH}
                style={{
                  width: canvasW,
                  height: canvasH,
                  display: "block",
                  imageRendering: "pixelated",
                }}
              />
            )}

            {/* HiDPI overlay: scale bar */}
            <canvas
              ref={overlayRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: canvasW,
                height: canvasH,
                pointerEvents: "none",
              }}
            />

            {/* Cursor readout (top-right) */}
            {cursorInfo && (
              <Box
                sx={{
                  position: "absolute",
                  top: 3,
                  right: 3,
                  bgcolor: "rgba(0,0,0,0.35)",
                  px: 0.5,
                  py: 0.15,
                  pointerEvents: "none",
                  minWidth: 100,
                  textAlign: "right",
                }}
              >
                <Typography
                  sx={{
                    fontSize: 9,
                    fontFamily: "monospace",
                    color: "rgba(255,255,255,0.7)",
                    whiteSpace: "nowrap",
                    lineHeight: 1.2,
                  }}
                >
                  ({cursorInfo.row}, {cursorInfo.col})
                  {pixelSize > 0
                    ? ` = (${(cursorInfo.row * calibratedFactor).toFixed(1)}, ${(cursorInfo.col * calibratedFactor).toFixed(1)} ${calibratedUnit})`
                    : ""}{" "}
                  {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}

            {/* Resize handle (bottom-right triangle) */}
            <Box
              onMouseDown={handleResizeMouseDown}
              sx={{
                position: "absolute",
                bottom: 0,
                right: 0,
                width: 16,
                height: 16,
                cursor: "nwse-resize",
                opacity: 0.6,
                pointerEvents: "auto",
                background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
                "&:hover": { opacity: 1 },
              }}
            />
          </Box>

          {/* Stats bar — same format as Show2D */}
          {showStats && frameCount > 0 && (
            <Box
              sx={{
                mt: `${SPACING.XS}px`,
                px: 1,
                py: 0.5,
                bgcolor: themeColors.bgAlt,
                display: "flex",
                gap: 2,
                alignItems: "center",
                boxSizing: "border-box",
                overflow: "hidden",
                whiteSpace: "nowrap",
              }}
            >
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Mean{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMean)}
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Min{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMin)}
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Max{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMax)}
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Std{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsStd)}
                </Box>
              </Typography>
              {zoom !== 1 && (
                <>
                  <Box sx={{ flex: 1 }} />
                  <Typography
                    sx={{
                      ...typography.value,
                      color: themeColors.accent,
                      fontWeight: "bold",
                    }}
                  >
                    {zoom.toFixed(1)}×
                  </Typography>
                </>
              )}
            </Box>
          )}

          {/* Controls + Histogram */}
          {showControls && (
            <Box
              sx={{
                mt: `${SPACING.SM}px`,
                display: "flex",
                flexDirection: "column",
                gap: `${SPACING.XS}px`,
                boxSizing: "border-box",
              }}
            >
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                {/* Left: two control rows */}
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: `${SPACING.XS}px`,
                    flex: 1,
                    justifyContent: "center",
                  }}
                >
                  {/* Row 1: Scale + Color + Colorbar */}
                  <Box
                    sx={{
                      ...controlRow,
                      border: `1px solid ${themeColors.border}`,
                      bgcolor: themeColors.controlBg,
                    }}
                  >
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      Scale:
                    </Typography>
                    <Select
                      value={logScale ? "log" : "linear"}
                      onChange={(e) => setLogScale(e.target.value === "log")}
                      size="small"
                      sx={{ ...themedSelect, minWidth: 45 }}
                      MenuProps={themedMenuProps}
                    >
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      Color:
                    </Typography>
                    <Select
                      size="small"
                      value={cmap}
                      onChange={(e) => setCmap(e.target.value)}
                      MenuProps={themedMenuProps}
                      sx={{ ...themedSelect, minWidth: 60 }}
                    >
                      {COLORMAP_NAMES.map((name) => (
                        <MenuItem key={name} value={name}>
                          {name.charAt(0).toUpperCase() + name.slice(1)}
                        </MenuItem>
                      ))}
                    </Select>
                  </Box>

                  {/* Row 2: Auto + zoom indicator */}
                  <Box
                    sx={{
                      ...controlRow,
                      border: `1px solid ${themeColors.border}`,
                      bgcolor: themeColors.controlBg,
                    }}
                  >
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      Auto:
                    </Typography>
                    <Switch
                      checked={autoContrast}
                      onChange={() => {
                        const next = !autoContrast;
                        autoContrastRef.current = next;
                        setAutoContrast(next);
                      }}
                      size="small"
                      sx={switchStyles.small}
                    />
                    {/* Label: current frame info */}
                    {displayFrame && (
                      <Typography
                        sx={{
                          ...typography.value,
                          color: themeColors.textMuted,
                          ml: 0.5,
                          maxWidth: 140,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {displayLabel}
                      </Typography>
                    )}
                    {displayFrame && (
                      <Typography
                        sx={{
                          ...typography.value,
                          color: themeColors.textMuted,
                          ml: 0.5,
                        }}
                      >
                        {displayDims}
                      </Typography>
                    )}
                  </Box>
                </Box>

                {/* Right: Histogram */}
                {histData && (
                  <Box
                    sx={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-end",
                      justifyContent: "center",
                    }}
                  >
                    <Histogram
                      data={histData}
                      vminPct={vminPct}
                      vmaxPct={vmaxPct}
                      onRangeChange={handleRangeChange}
                      width={110}
                      height={58}
                      theme={themeInfo.theme === "dark" ? "dark" : "light"}
                      dataMin={imageDataRange.min}
                      dataMax={imageDataRange.max}
                    />
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </Box>
      </Stack>
    </Box>
  );
});

export default { render };
