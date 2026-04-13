import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import "./live.css";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse, GPUColormapEngine, getGPUColormapEngine } from "../colormaps";
import { extractFloat32, formatNumber } from "../format";
import { findDataRange, applyLogScale, percentileClip, sliderRange } from "../stats";
import { useTheme } from "../theme";
import { computeHistogramFromBytes } from "../histogram";
import { drawScaleBarHiDPI } from "../scalebar";

// ─── Design Constants (matching quantem.live SSB layout) ─────────────
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const DPR = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
const THUMB_SIZE = 64;
const THUMB_COLS = 2;
const THUMB_PANEL_WIDTH_RATIO = 0.45;
const MIN_CANVAS = 256;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { overflow: "hidden", position: "relative" as const },
};

function formatStat(v: number): string {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 0.001 || abs >= 10000) return v.toExponential(2);
  if (abs < 1) return v.toFixed(4);
  return v.toFixed(1);
}

// ─── Types ────────────────────────────────────────────────────────────
interface FrameEntry {
  data: Float32Array;
  height: number;
  width: number;
  label: string;
  thumbCanvas: HTMLCanvasElement | null;
}

// ─── Histogram ────────────────────────────────────────────────────────
interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  themeColors: Record<string, string>;
  dataMin?: number;
  dataMax?: number;
}
function Histogram({ data, vminPct, vmaxPct, onRangeChange, width = 110, height = 58, themeColors, dataMin = 0, dataMax = 1 }: HistogramProps) {
  const bins = React.useMemo(() => data ? computeHistogramFromBytes(data) : null, [data]);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    if (!bins || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const n = bins.length;
    ctx.clearRect(0, 0, width, height);
    const lo = Math.floor((vminPct / 100) * n);
    const hi = Math.floor((vmaxPct / 100) * n);
    const barW = Math.max(1, width / n);
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = i >= lo && i <= hi ? (themeColors.accent || "#90caf9") : (themeColors.border || "#333");
      ctx.fillRect(i * barW, height - bins[i] * height * 0.9, barW, bins[i] * height * 0.9);
    }
  }, [bins, vminPct, vmaxPct, width, height, themeColors]);
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      <canvas ref={canvasRef} width={width} height={height} style={{ width, height }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [a, b] = v as number[]; onRangeChange(a, b); }}
        valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        min={0} max={100} step={0.5} size="small"
        sx={{ width, "& .MuiSlider-thumb": { width: 8, height: 8 } }}
      />
    </Box>
  );
}

// ─── Main Component ──────────────────────────────────────────────────
const render = createRender(() => {
  const { themeInfo, colors: themeColors } = useTheme();

  // Model state
  const [title] = useModelState<string>("title");
  const [mode, setMode] = useModelState<string>("mode");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [bufferSize] = useModelState<number>("buffer_size");
  const [nFrames] = useModelState<number>("n_frames");
  const [showControls] = useModelState<boolean>("show_controls");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [frameIdx, setFrameIdx] = useModelState<number>("frame_idx");
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [fps] = useModelState<number>("fps");

  // Push / Batch
  const [pushBytes] = useModelState<DataView>("_push_bytes");
  const [pushHeight] = useModelState<number>("_push_height");
  const [pushWidth] = useModelState<number>("_push_width");
  const [pushLabel] = useModelState<string>("_push_label");
  const [pushCounter] = useModelState<number>("_push_counter");
  const [batchBytes] = useModelState<DataView>("_batch_bytes");
  const [batchDims] = useModelState<number[][]>("_batch_dims");
  const [batchLabels] = useModelState<string[]>("_batch_labels");
  const [batchCounter] = useModelState<number>("_batch_counter");

  // Local state
  const framesRef = React.useRef<FrameEntry[]>([]);
  const [frameCount, setFrameCount] = React.useState(0);
  const [selectedIdx, setSelectedIdx] = React.useState(0);
  const [canvasSize, setCanvasSize] = React.useState(400);
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);
  const [histData, setHistData] = React.useState<Float32Array | null>(null);
  const [dataRange, setDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Zoom/pan
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const panStartRef = React.useRef<{ x: number; y: number; pX: number; pY: number } | null>(null);
  const dragRef = React.useRef({ wasDrag: false });

  // Refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const resizeStartRef = React.useRef<{ x: number; y: number; s: number } | null>(null);
  const playingRef = React.useRef(false);
  const animFrameRef = React.useRef(0);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const thumbListRef = React.useRef<HTMLDivElement>(null);
  const uiCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // GPU colormap engine (zero-copy rendering)
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);
  React.useEffect(() => {
    getGPUColormapEngine().then(engine => {
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        console.log("[Live] WebGPU colormap engine initialized");
      }
    });
  }, []);

  // Computed
  const thumbPanelWidth = Math.floor(canvasSize * THUMB_PANEL_WIDTH_RATIO);

  // ─── Accumulate single-push frames ────────────────────────────────
  const prevCounterRef = React.useRef(0);
  React.useEffect(() => {
    if (pushCounter === prevCounterRef.current) return;
    prevCounterRef.current = pushCounter;
    if (!pushBytes || pushHeight === 0 || pushWidth === 0) return;
    const floats = extractFloat32(pushBytes);
    if (floats.length !== pushHeight * pushWidth) return;
    const data = new Float32Array(floats);
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const range = findDataRange(data);
    const thumbCanvas = renderToOffscreen(data, pushWidth, pushHeight, lut, range.min, range.max);
    const entry: FrameEntry = { data, height: pushHeight, width: pushWidth, label: pushLabel || `Frame ${framesRef.current.length}`, thumbCanvas };
    framesRef.current = [entry, ...framesRef.current];
    if (framesRef.current.length > bufferSize) framesRef.current = framesRef.current.slice(0, bufferSize);
    setFrameCount(framesRef.current.length);
    setSelectedIdx(0);
    if (thumbListRef.current) thumbListRef.current.scrollTop = 0;
  }, [pushCounter]);

  // ─── Accumulate batch frames ──────────────────────────────────────
  const prevBatchRef = React.useRef(0);
  React.useEffect(() => {
    if (batchCounter === prevBatchRef.current) return;
    prevBatchRef.current = batchCounter;
    if (!batchBytes || !batchDims || batchDims.length === 0) return;
    const allFloats = extractFloat32(batchBytes);
    if (!allFloats || allFloats.length === 0) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const newEntries: FrameEntry[] = [];
    let offset = 0;
    for (let i = 0; i < batchDims.length; i++) {
      const [h, w] = batchDims[i];
      const count = h * w;
      if (offset + count > allFloats.length) break;
      const data = new Float32Array(allFloats.buffer, allFloats.byteOffset + offset * 4, count);
      const dataCopy = new Float32Array(data);
      const range = findDataRange(dataCopy);
      const thumbCanvas = renderToOffscreen(dataCopy, w, h, lut, range.min, range.max);
      const label = (batchLabels && i < batchLabels.length) ? batchLabels[i] : `Frame ${i}`;
      newEntries.push({ data: dataCopy, height: h, width: w, label, thumbCanvas });
      offset += count;
    }
    newEntries.reverse();
    framesRef.current = [...newEntries, ...framesRef.current];
    if (framesRef.current.length > bufferSize) framesRef.current = framesRef.current.slice(0, bufferSize);
    setFrameCount(framesRef.current.length);
    setSelectedIdx(0);
    if (thumbListRef.current) thumbListRef.current.scrollTop = 0;
  }, [batchCounter]);

  // ─── Display index ────────────────────────────────────────────────
  const getDisplayIdx = React.useCallback(() => {
    if (framesRef.current.length === 0) return -1;
    if (mode === "3d") {
      const idx = Math.min(Math.max(0, frameIdx), framesRef.current.length - 1);
      return framesRef.current.length - 1 - idx;
    }
    return Math.min(selectedIdx, framesRef.current.length - 1);
  }, [mode, frameIdx, selectedIdx, frameCount]);

  // ─── Histogram ────────────────────────────────────────────────────
  React.useEffect(() => {
    const idx = getDisplayIdx();
    if (idx < 0) return;
    const frame = framesRef.current[idx];
    if (!frame) return;
    const processed = logScale ? applyLogScale(frame.data) : frame.data;
    setHistData(processed);
    setDataRange(findDataRange(processed));
  }, [getDisplayIdx, logScale, frameCount]);

  // ─── Render main canvas (GPU zero-copy when available) ────────────
  React.useEffect(() => {
    const idx = getDisplayIdx();
    if (idx < 0) return;
    const frame = framesRef.current[idx];
    if (!frame) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { data, height, width } = frame;

    // Compute vmin/vmax
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    let vmin: number, vmax: number;
    const range = findDataRange(data);
    const logRange = logScale
      ? { min: Math.log1p(Math.max(range.min, 0)), max: Math.log1p(Math.max(range.max, 0)) }
      : range;
    if (autoContrast) {
      const processed = logScale ? applyLogScale(data) : data;
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else if (vminPct > 0 || vmaxPct < 100) {
      ({ vmin, vmax } = sliderRange(logRange.min, logRange.max, vminPct, vmaxPct));
    } else {
      vmin = logRange.min; vmax = logRange.max;
    }

    // Ensure offscreen canvas exists
    if (!mainOffscreenRef.current || mainOffscreenRef.current.width !== width || mainOffscreenRef.current.height !== height) {
      mainOffscreenRef.current = document.createElement("canvas");
      mainOffscreenRef.current.width = width;
      mainOffscreenRef.current.height = height;
      mainImgDataRef.current = mainOffscreenRef.current.getContext("2d")!.createImageData(width, height);
    }

    // GPU zero-copy path
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      engine.uploadData(0, data, width, height);
      engine.uploadLUT(cmap, lut);
      const bitmaps = engine.renderSlotsToImageBitmap([0], [{ vmin, vmax }], logScale);
      if (bitmaps && bitmaps[0]) {
        mainOffscreenRef.current.getContext("2d")!.drawImage(bitmaps[0], 0, 0);
      } else {
        // Fallback
        const processed = logScale ? applyLogScale(data) : data;
        renderToOffscreenReuse(processed, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
      }
    } else {
      // CPU fallback
      const processed = logScale ? applyLogScale(data) : data;
      renderToOffscreenReuse(processed, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
    }

    // Draw to display canvas
    canvas.width = canvasSize * DPR;
    canvas.height = canvasSize * DPR;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(DPR, DPR);
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(mainOffscreenRef.current, 0, 0, canvasSize / zoom, canvasSize / zoom);
    ctx.restore();
  }, [getDisplayIdx, cmap, logScale, autoContrast, vminPct, vmaxPct, canvasSize, frameCount, zoom, panX, panY]);

  // ─── Scale bar (HiDPI UI canvas overlay) ───────────────────────────
  React.useEffect(() => {
    const uiCanvas = uiCanvasRef.current;
    if (!uiCanvas || frameCount === 0) return;
    const ctx = uiCanvas.getContext("2d");
    if (!ctx) return;
    uiCanvas.width = canvasSize * DPR;
    uiCanvas.height = canvasSize * DPR;
    ctx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
    if (pixelSize > 0) {
      const idx = getDisplayIdx();
      const frame = idx >= 0 ? framesRef.current[idx] : null;
      const sliceW = frame ? frame.width : canvasSize;
      drawScaleBarHiDPI(uiCanvas, DPR, zoom, pixelSize, "Å", sliceW);
    }
  }, [canvasSize, zoom, pixelSize, frameCount, getDisplayIdx]);

  // ─── Copy canvas to clipboard ─────────────────────────────────────
  const handleCopy = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) {
        navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]).catch(() => {});
      }
    });
  }, []);

  // ─── Playback ─────────────────────────────────────────────────────
  React.useEffect(() => { playingRef.current = playing; }, [playing]);
  React.useEffect(() => {
    if (mode !== "3d" || !playing || framesRef.current.length < 2) return;
    let lastTime = 0;
    const tick = (now: number) => {
      if (!playingRef.current) return;
      if (now - lastTime >= 1000 / fps) { lastTime = now; setFrameIdx((prev: number) => (prev + 1) % framesRef.current.length); }
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [mode, playing, fps, frameCount]);

  // ─── Zoom (scroll wheel, centered) ────────────────────────────────
  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const nz = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));
      setPanX(prev => mx - (mx - prev) * (nz / zoom));
      setPanY(prev => my - (my - prev) * (nz / zoom));
      setZoom(nz);
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [zoom]);

  // ─── Pan (click + drag) ───────────────────────────────────────────
  const onCanvasMouseDown = React.useCallback((e: React.MouseEvent) => {
    dragRef.current = { wasDrag: false };
    panStartRef.current = { x: e.clientX, y: e.clientY, pX: panX, pY: panY };
    const onMove = (ev: MouseEvent) => {
      if (!panStartRef.current) return;
      const dx = ev.clientX - panStartRef.current.x;
      const dy = ev.clientY - panStartRef.current.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) dragRef.current.wasDrag = true;
      setPanX(panStartRef.current.pX + dx);
      setPanY(panStartRef.current.pY + dy);
    };
    const onUp = () => { panStartRef.current = null; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [panX, panY]);

  // ─── Cursor readout ───────────────────────────────────────────────
  const handleCursorMove = React.useCallback((e: React.MouseEvent) => {
    const idx = getDisplayIdx();
    if (idx < 0) return;
    const frame = framesRef.current[idx];
    if (!frame) { setCursorInfo(null); return; }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const imgCol = Math.floor((mx - panX) / zoom * (frame.width / canvasSize * zoom));
    const imgRow = Math.floor((my - panY) / zoom * (frame.height / canvasSize * zoom));
    if (imgCol >= 0 && imgCol < frame.width && imgRow >= 0 && imgRow < frame.height) {
      const value = frame.data[imgRow * frame.width + imgCol];
      setCursorInfo({ row: imgRow, col: imgCol, value });
    } else {
      setCursorInfo(null);
    }
  }, [getDisplayIdx, frameCount, zoom, panX, panY, canvasSize]);

  // ─── Reset zoom ───────────────────────────────────────────────────
  const resetZoom = React.useCallback(() => { setZoom(1); setPanX(0); setPanY(0); }, []);

  // ─── Resize handle ────────────────────────────────────────────────
  const onResizeMouseDown = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeStartRef.current = { x: e.clientX, y: e.clientY, s: canvasSize };
    const onMove = (ev: MouseEvent) => {
      if (!resizeStartRef.current) return;
      const delta = Math.max(ev.clientX - resizeStartRef.current.x, ev.clientY - resizeStartRef.current.y);
      setCanvasSize(Math.max(MIN_CANVAS, resizeStartRef.current.s + delta));
    };
    const onUp = () => { resizeStartRef.current = null; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [canvasSize]);

  // ─── Keyboard ─────────────────────────────────────────────────────
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.key === "ArrowLeft") {
      e.preventDefault(); e.stopPropagation();
      if (mode === "3d") { setPlaying(false); setFrameIdx((prev: number) => Math.max(0, prev - 1)); }
      else { setSelectedIdx(prev => Math.min(prev + 1, framesRef.current.length - 1)); }
    } else if (e.key === "ArrowRight") {
      e.preventDefault(); e.stopPropagation();
      if (mode === "3d") { setPlaying(false); setFrameIdx((prev: number) => Math.min(prev + 1, framesRef.current.length - 1)); }
      else { setSelectedIdx(prev => Math.max(0, prev - 1)); }
    } else if (e.key === " ") {
      e.preventDefault();
      if (mode === "3d") setPlaying(!playing);
    } else if (e.key === "r" || e.key === "R") {
      e.preventDefault(); resetZoom();
    }
  }, [mode, playing, frameCount]);

  // ─── Display state ────────────────────────────────────────────────
  const displayIdx = getDisplayIdx();
  const displayFrame = displayIdx >= 0 ? framesRef.current[displayIdx] : null;
  const displayLabel = displayFrame?.label || "—";
  const displayDims = displayFrame ? `${displayFrame.height}×${displayFrame.width}` : "—";
  const isSelected = (i: number) => mode === "3d" ? (framesRef.current.length - 1 - i === frameIdx) : i === selectedIdx;
  const accentGreen = "#66bb6a";

  // ─── Render ───────────────────────────────────────────────────────
  return (
    <Box ref={rootRef} className="live-widget" tabIndex={0} onKeyDown={handleKeyDown}
      onMouseDownCapture={() => rootRef.current?.focus()}
      sx={{ ...container.root, color: themeColors.text, outline: "none" }}>

      {/* Title bar */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: `${SPACING.XS}px` }}>
        <Typography sx={{ ...typography.label, fontWeight: "bold" }}>{title}</Typography>
        <Typography sx={{ ...typography.labelSmall, color: themeColors.textSecondary, ml: "auto" }}>
          {frameCount} frame{frameCount !== 1 ? "s" : ""}
          {nFrames > frameCount ? ` (${nFrames} total)` : ""}
        </Typography>
      </Box>

      <Stack direction="row" spacing={`${SPACING.LG}px`}>

        {/* ══════════ LEFT: Thumbnail Panel ══════════ */}
        <Box sx={{ width: thumbPanelWidth, flexShrink: 0 }}>
          <Box ref={thumbListRef} sx={{
            border: `1px solid ${themeColors.border}`,
            bgcolor: themeColors.controlBg,
            p: `${SPACING.SM}px`,
            maxHeight: canvasSize + 150,
            overflowY: "auto",
            overflowX: "hidden",
          }}>
            <Typography sx={{ ...typography.labelSmall, color: themeColors.textSecondary, mb: `${SPACING.XS}px` }}>Scans</Typography>
            <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${THUMB_COLS}, 1fr)`, gap: `${SPACING.XS}px` }}>
              {framesRef.current.slice(0, frameCount).map((frame, i) => (
                <Box key={`thumb-${frameCount - i}`}
                  onClick={() => { if (mode === "3d") { setPlaying(false); setFrameIdx(framesRef.current.length - 1 - i); } else { setSelectedIdx(i); } }}
                  sx={{
                    border: isSelected(i) ? `2px solid ${accentGreen}` : `2px solid transparent`,
                    cursor: "pointer",
                    opacity: isSelected(i) ? 1 : 0.7,
                    "&:hover": { opacity: 1 },
                    position: "relative",
                  }}
                >
                  {frame.thumbCanvas ? (
                    <canvas
                      ref={(el) => {
                        if (el && frame.thumbCanvas) {
                          el.width = THUMB_SIZE * DPR; el.height = THUMB_SIZE * DPR;
                          const ctx = el.getContext("2d");
                          if (ctx) { ctx.imageSmoothingEnabled = true; ctx.drawImage(frame.thumbCanvas, 0, 0, el.width, el.height); }
                        }
                      }}
                      style={{ width: THUMB_SIZE, height: THUMB_SIZE, display: "block" }}
                    />
                  ) : (
                    <Box sx={{ width: THUMB_SIZE, height: THUMB_SIZE, bgcolor: themeColors.border }} />
                  )}
                  <Typography sx={{
                    position: "absolute", bottom: 0, left: 0, right: 0,
                    fontSize: 8, textAlign: "center", bgcolor: "rgba(0,0,0,0.6)",
                    color: "#fff", lineHeight: 1.2, px: 0.25,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {frame.label}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </Box>

        {/* ══════════ RIGHT: Inspect Panel ══════════ */}
        <Box sx={{ width: canvasSize, flexShrink: 0, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>

          {/* Header bar — label + mode + RESET */}
          <Stack direction="row" justifyContent="space-between" alignItems="center"
            sx={{ height: 28, px: 1, bgcolor: themeColors.controlBg, border: `1px solid ${themeColors.border}` }}>
            <Typography sx={{ ...typography.label, fontFamily: "monospace", color: accentGreen, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 1, minWidth: 0 }}>
              {displayLabel}
            </Typography>
            <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ flexShrink: 0 }}>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textSecondary }}>{displayDims}</Typography>
              <Button size="small" sx={{ fontSize: 10, py: 0.25, px: 1, minWidth: 0, color: themeColors.text }}
                onClick={handleCopy}>COPY</Button>
              <Button size="small" sx={{ fontSize: 10, py: 0.25, px: 1, minWidth: 0, color: themeColors.text }}
                disabled={zoom === 1 && panX === 0 && panY === 0}
                onClick={resetZoom}>RESET</Button>
            </Stack>
          </Stack>

          {/* Canvas */}
          <Box ref={canvasContainerRef} sx={{ ...container.imageBox, bgcolor: themeColors.controlBg, border: `1px solid ${themeColors.border}`, width: canvasSize, height: canvasSize }}>
            {frameCount === 0 ? (
              <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Typography sx={{ ...typography.label, color: themeColors.textSecondary }}>Waiting for data...</Typography>
              </Box>
            ) : (
              <canvas ref={canvasRef}
                onMouseDown={onCanvasMouseDown}
                onDoubleClick={resetZoom}
                onMouseMove={handleCursorMove}
                onMouseLeave={() => setCursorInfo(null)}
                style={{ width: canvasSize, height: canvasSize, cursor: "grab", display: "block", imageRendering: "pixelated" }}
              />
            )}
            {/* Scale bar UI canvas (HiDPI overlay) */}
            {pixelSize > 0 && frameCount > 0 && (
              <canvas ref={uiCanvasRef}
                style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            )}
            {/* Cursor readout overlay (top-right, like quantem.live) */}
            {cursorInfo && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            {/* Resize handle */}
            <Box onMouseDown={onResizeMouseDown} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.4, background: `linear-gradient(135deg, transparent 50%, ${themeColors.textSecondary} 50%)`, "&:hover": { opacity: 0.8 } }} />
          </Box>

          {/* 3D playback slider */}
          {mode === "3d" && frameCount > 1 && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <IconButton size="small" onClick={() => setPlaying(!playing)} sx={{ color: themeColors.text, p: 0.25 }}>
                {playing ? <PauseIcon sx={{ fontSize: 16 }} /> : <PlayArrowIcon sx={{ fontSize: 16 }} />}
              </IconButton>
              <Slider value={frameIdx} min={0} max={Math.max(0, frameCount - 1)} step={1}
                onChange={(_, v) => { setPlaying(false); setFrameIdx(v as number); }}
                size="small" sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
              <Typography sx={{ ...typography.value, minWidth: 45, textAlign: "right" }}>{frameIdx + 1}/{frameCount}</Typography>
            </Box>
          )}

          {/* Controls + Histogram (two rows left, histogram right — like quantem.live) */}
          {showControls && (
            <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, width: "100%" }}>
              {/* Left: two rows */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Stats / cursor readout / zoom */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1, py: 0.5, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, flexWrap: "wrap" }}>
                  {cursorInfo ? (
                    <Typography sx={{ ...typography.value, color: themeColors.textSecondary }}>
                      ({cursorInfo.row}, {cursorInfo.col}) = <Box component="span" sx={{ color: accentGreen }}>{formatStat(cursorInfo.value)}</Box>
                    </Typography>
                  ) : (
                    <Typography sx={{ ...typography.value, color: themeColors.textSecondary }}>--</Typography>
                  )}
                  {zoom !== 1 && (
                    <>
                      <Box sx={{ flex: 1 }} />
                      <Typography sx={{ ...typography.value, color: themeColors.textSecondary }}>{zoom.toFixed(1)}×</Typography>
                    </>
                  )}
                </Box>
                {/* Row 2: Display controls */}
                <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, px: 1, py: 0.5, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, flexWrap: "wrap" }}>
                  <Typography sx={{ ...typography.labelSmall }}>Mode:</Typography>
                  <Select value={mode} onChange={(e) => setMode(e.target.value)} size="small"
                    sx={{ fontSize: 10, height: 22, minWidth: 45, color: themeColors.text, "& .MuiSelect-select": { py: 0.25, px: 0.5 } }}>
                    <MenuItem value="2d" sx={{ fontSize: 10 }}>2D</MenuItem>
                    <MenuItem value="3d" sx={{ fontSize: 10 }}>3D</MenuItem>
                  </Select>
                  <Select value={cmap} onChange={(e) => setCmap(e.target.value)} size="small"
                    sx={{ fontSize: 10, height: 22, minWidth: 55, color: themeColors.text, "& .MuiSelect-select": { py: 0.25, px: 0.5 } }}>
                    {COLORMAP_NAMES.map(n => <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>)}
                  </Select>
                  <Typography sx={{ ...typography.labelSmall }}>Log:</Typography>
                  <Switch checked={logScale} onChange={(_, v) => setLogScale(v)} size="small"
                    sx={{ "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } }} />
                  <Typography sx={{ ...typography.labelSmall }}>Auto:</Typography>
                  <Switch checked={autoContrast} onChange={(_, v) => setAutoContrast(v)} size="small"
                    sx={{ "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } }} />
                </Box>
              </Box>
              {/* Right: Histogram */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", flexShrink: 0 }}>
                <Histogram data={histData} vminPct={vminPct} vmaxPct={vmaxPct}
                  onRangeChange={(lo, hi) => { setVminPct(lo); setVmaxPct(hi); }}
                  width={110} height={58} themeColors={themeColors}
                  dataMin={dataRange.min} dataMax={dataRange.max} />
              </Box>
            </Box>
          )}
        </Box>
      </Stack>
    </Box>
  );
});

export default { render };
