/**
 * AtomFinder — Interactive atom column localization.
 *
 * Displays a single 2D image with detected atom positions as colored markers
 * (cyan = sublattice A, magenta = sublattice B, green = single sublattice),
 * plus polarization arrows from B-sites to their displacement direction.
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
import { useTheme } from "../theme";
import { drawScaleBarHiDPI } from "../scalebar";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, sliderRange, applyLogScaleInPlace, percentileClip } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";

// ─── Style constants ────────────────────────────────────────────────────
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;
const DPR = window.devicePixelRatio || 1;
const CANVAS_MIN = 384;
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
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

// ─── Histogram ──────────────────────────────────────────────────────────
interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
}

function Histogram({
  data,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 110,
  height = 50,
  theme = "dark",
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => (data ? computeHistogramFromBytes(data) : null), [data]);
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
      ref={canvasRef}
      width={width}
      height={height}
      style={{ cursor: "ew-resize", display: "block" }}
      onMouseDown={(e) => handleMouse(e, true)}
      onMouseMove={(e) => {
        if (draggingRef.current) handleMouse(e, false);
      }}
      onMouseUp={() => {
        draggingRef.current = null;
      }}
      onMouseLeave={() => {
        draggingRef.current = null;
      }}
    />
  );
}

// ─── Helper: format stat value ──────────────────────────────────────────
function formatStat(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// ─── Main component ────────────────────────────────────────────────────

function AtomFinder() {
  const { themeInfo, colors: themeColors } = useTheme();
  const rootRef = React.useRef<HTMLDivElement>(null);

  const themedSelect = {
    "& .MuiSelect-select": { py: 0.25, px: 1, fontSize: 10, color: themeColors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
    bgcolor: themeColors.controlBg,
    minWidth: 80,
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

  // ── Model state ───────────────────────────────────────────────────────
  const [title] = useModelState<string>("title");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [units] = useModelState<string>("units");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  const [minSigma, setMinSigma] = useModelState<number>("min_sigma");
  const [maxSigma, setMaxSigma] = useModelState<number>("max_sigma");
  const [blobThreshold, setBlobThreshold] = useModelState<number>("blob_threshold");
  const [fitSubpixel, setFitSubpixel] = useModelState<boolean>("fit_gaussian_subpixel");
  const [maskRadiusPx, setMaskRadiusPx] = useModelState<number>("mask_radius_px");
  const [nSublattices, setNSublattices] = useModelState<number>("n_sublattices");
  const [sublatticeFraction, setSublatticeFraction] =
    useModelState<number>("sublattice_fraction");
  const [polarizationActive, setPolarizationActive] =
    useModelState<boolean>("polarization_active");
  const [polarizationScale, setPolarizationScale] =
    useModelState<number>("polarization_scale");

  const [atomPositionsBytes] = useModelState<DataView>("atom_positions_bytes");
  const [subAIdxBytes] = useModelState<DataView>("sublattice_a_indices_bytes");
  const [subBIdxBytes] = useModelState<DataView>("sublattice_b_indices_bytes");
  const [polarizationBytes] = useModelState<DataView>("polarization_bytes");
  const [nAtoms] = useModelState<number>("n_atoms");

  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("AtomFinder", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideStats = toolVisibility.isHidden("stats");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideDisplay = toolVisibility.isHidden("display");
  const hideDetection = toolVisibility.isHidden("detection");
  const hideSublattice = toolVisibility.isHidden("sublattice");
  const hidePolarization = toolVisibility.isHidden("polarization");
  const hideView = toolVisibility.isHidden("view");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockDetection = toolVisibility.isLocked("detection");
  const lockSublattice = toolVisibility.isLocked("sublattice");
  const lockPolarization = toolVisibility.isLocked("polarization");
  const lockView = toolVisibility.isLocked("view");

  // ── Local UI state ────────────────────────────────────────────────────
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_MIN);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{
    x: number;
    y: number;
    size: number;
  } | null>(null);
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);
  const [histData, setHistData] = React.useState<Float32Array | null>(null);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(
    null,
  );

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const offscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const rawDataRef = React.useRef<Float32Array | null>(null);
  const [renderVersion, setRenderVersion] = React.useState(0);

  // ── Decoded outputs (memoized from bytes) ─────────────────────────────
  const atomPositions = React.useMemo(() => {
    if (!atomPositionsBytes || !atomPositionsBytes.byteLength) return null;
    return new Float32Array(
      atomPositionsBytes.buffer,
      atomPositionsBytes.byteOffset,
      atomPositionsBytes.byteLength / 4,
    );
  }, [atomPositionsBytes]);

  const subAIndices = React.useMemo(() => {
    if (!subAIdxBytes || !subAIdxBytes.byteLength) return null;
    return new Int32Array(
      subAIdxBytes.buffer,
      subAIdxBytes.byteOffset,
      subAIdxBytes.byteLength / 4,
    );
  }, [subAIdxBytes]);

  const subBIndices = React.useMemo(() => {
    if (!subBIdxBytes || !subBIdxBytes.byteLength) return null;
    return new Int32Array(
      subBIdxBytes.buffer,
      subBIdxBytes.byteOffset,
      subBIdxBytes.byteLength / 4,
    );
  }, [subBIdxBytes]);

  const polarizationVectors = React.useMemo(() => {
    if (!polarizationBytes || !polarizationBytes.byteLength) return null;
    return new Float32Array(
      polarizationBytes.buffer,
      polarizationBytes.byteOffset,
      polarizationBytes.byteLength / 4,
    );
  }, [polarizationBytes]);

  // ── Canvas resize handle ──────────────────────────────────────────────
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeCanvasStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeCanvasStart ? resizeCanvasStart.size : canvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeCanvasStart) return;
      const delta = Math.max(e.clientX - resizeCanvasStart.x, e.clientY - resizeCanvasStart.y);
      latestSize = Math.max(CANVAS_MIN, resizeCanvasStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setCanvasSize(latestSize);
      setIsResizingCanvas(false);
      setResizeCanvasStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isResizingCanvas, resizeCanvasStart]);

  // ── Render image to offscreen canvas ──────────────────────────────────
  React.useEffect(() => {
    if (!frameBytes || !frameBytes.byteLength || !width || !height) return;
    const raw = new Float32Array(
      frameBytes.buffer,
      frameBytes.byteOffset,
      frameBytes.byteLength / 4,
    );
    rawDataRef.current = raw;
    let scaled: Float32Array;
    if (logScale) {
      scaled = new Float32Array(raw.length);
      applyLogScaleInPlace(raw, scaled);
    } else {
      scaled = raw;
    }
    let vmin: number, vmax: number;
    if (autoContrast) {
      const clip = percentileClip(scaled, percentileLow, percentileHigh);
      vmin = clip.vmin;
      vmax = clip.vmax;
    } else {
      const { min, max } = findDataRange(scaled);
      const sr = sliderRange(min, max, vminPct, vmaxPct);
      vmin = sr.vmin;
      vmax = sr.vmax;
    }
    const lut = COLORMAPS[cmap] || COLORMAPS.gray || COLORMAPS.inferno;
    let offscreen = offscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      offscreenRef.current = offscreen;
    }
    offscreen.width = width;
    offscreen.height = height;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(width, height);
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
    setHistData(scaled);
    setRenderVersion((v) => v + 1);
  }, [frameBytes, cmap, logScale, autoContrast, percentileLow, percentileHigh, vminPct, vmaxPct, width, height]);

  // ── Draw image (cheap zoom/pan) ───────────────────────────────────────
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const offscreen = offscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * zoom) / 2 + panX;
    const offY = (canvasSize - canvasSize * zoom) / 2 + panY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * zoom, canvasSize * zoom);
  }, [renderVersion, zoom, panX, panY, canvasSize]);

  // ── Overlay (atom markers + polarization arrows + scale bar) ──────────
  React.useLayoutEffect(() => {
    const canvas = uiRef.current;
    if (!canvas || !width || !height) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    const scX = (cssW / width) * zoom;
    const scY = (cssW / height) * zoom;
    const offX = (cssW - cssW * zoom) / 2 + panX;
    const offY = (cssW - cssW * zoom) / 2 + panY;

    const toScreen = (row: number, col: number) => ({
      x: offX + (col + 0.5) * scX,
      y: offY + (row + 0.5) * scY,
    });

    // Marker colors
    const colorA = "#00e5ff"; // cyan
    const colorB = "#ff4fd8"; // magenta
    const colorSingle = "#00ff88"; // accentGreen
    const arrowColor = themeInfo.theme === "dark" ? "#ffeb3b" : "#cc8800";

    if (atomPositions && atomPositions.length >= 4 && nAtoms > 0) {
      const hasSublattices =
        nSublattices === 2 &&
        subAIndices !== null &&
        subBIndices !== null &&
        subAIndices.length + subBIndices.length > 0;

      ctx.lineWidth = 0.5;
      ctx.strokeStyle = "rgba(0,0,0,0.6)";

      if (hasSublattices) {
        // Draw A then B so B sits on top
        const aSet = new Set<number>(Array.from(subAIndices!));
        const bSet = new Set<number>(Array.from(subBIndices!));
        for (let i = 0; i < nAtoms; i++) {
          const row = atomPositions[i * 4];
          const col = atomPositions[i * 4 + 1];
          const { x, y } = toScreen(row, col);
          let fill: string | null = null;
          if (aSet.has(i)) fill = colorA;
          else if (bSet.has(i)) fill = colorB;
          if (!fill) continue;
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fillStyle = fill;
          ctx.fill();
          ctx.stroke();
        }
      } else {
        for (let i = 0; i < nAtoms; i++) {
          const row = atomPositions[i * 4];
          const col = atomPositions[i * 4 + 1];
          const { x, y } = toScreen(row, col);
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fillStyle = colorSingle;
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    if (polarizationActive && polarizationVectors && polarizationVectors.length >= 4) {
      const nVecs = polarizationVectors.length / 4;
      ctx.strokeStyle = arrowColor;
      ctx.fillStyle = arrowColor;
      ctx.lineWidth = 1.2;
      for (let i = 0; i < nVecs; i++) {
        const row = polarizationVectors[i * 4];
        const col = polarizationVectors[i * 4 + 1];
        const dRow = polarizationVectors[i * 4 + 2];
        const dCol = polarizationVectors[i * 4 + 3];
        const start = toScreen(row, col);
        const endRow = row + dRow * polarizationScale;
        const endCol = col + dCol * polarizationScale;
        const end = toScreen(endRow, endCol);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
        // Arrowhead
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const len = Math.hypot(dx, dy);
        if (len > 1) {
          const ux = dx / len;
          const uy = dy / len;
          const ahLen = 4;
          const ahW = 2;
          const baseX = end.x - ux * ahLen;
          const baseY = end.y - uy * ahLen;
          ctx.beginPath();
          ctx.moveTo(end.x, end.y);
          ctx.lineTo(baseX - uy * ahW, baseY + ux * ahW);
          ctx.lineTo(baseX + uy * ahW, baseY - ux * ahW);
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    if (scaleBarVisible && pixelSize > 0) {
      const unitArg: "Å" | "mrad" | "px" =
        units === "mrad" ? "mrad" : units === "px" ? "px" : "Å";
      drawScaleBarHiDPI(canvas, DPR, zoom, pixelSize, unitArg, width);
    }

    if (zoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [
    renderVersion,
    zoom,
    panX,
    panY,
    canvasSize,
    width,
    height,
    nAtoms,
    atomPositions,
    subAIndices,
    subBIndices,
    nSublattices,
    polarizationActive,
    polarizationVectors,
    polarizationScale,
    scaleBarVisible,
    pixelSize,
    units,
    themeInfo.theme,
  ]);

  // ── Mouse handlers ────────────────────────────────────────────────────
  const isDraggingRef = React.useRef(false);
  const dragStartRef = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const screenToImage = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const offX = (canvasSize - canvasSize * zoom) / 2 + panX;
    const offY = (canvasSize - canvasSize * zoom) / 2 + panY;
    const col = ((mx - offX) / (canvasSize * zoom)) * width;
    const row = ((my - offY) / (canvasSize * zoom)) * height;
    return { row, col };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    dragStartRef.current = { x: e.clientX, y: e.clientY, panX, panY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingRef.current) {
      setPanX(dragStartRef.current.panX + (e.clientX - dragStartRef.current.x));
      setPanY(dragStartRef.current.panY + (e.clientY - dragStartRef.current.y));
      return;
    }
    if (!rawDataRef.current || !width) return;
    const { row, col } = screenToImage(e);
    const ri = Math.round(row);
    const ci = Math.round(col);
    if (ri >= 0 && ri < height && ci >= 0 && ci < width) {
      setCursorInfo({ row: ri, col: ci, value: rawDataRef.current[ri * width + ci] });
    } else {
      setCursorInfo(null);
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };
  const handleMouseLeave = () => {
    isDraggingRef.current = false;
    setCursorInfo(null);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };

  const resetView = () => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  };

  // ── Wheel scroll prevention ───────────────────────────────────────────
  const containerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const prevent = (e: WheelEvent) => e.preventDefault();
    const el = containerRef.current;
    if (el) el.addEventListener("wheel", prevent, { passive: false });
    return () => {
      if (el) el.removeEventListener("wheel", prevent);
    };
  }, []);

  // ── Keyboard ──────────────────────────────────────────────────────────
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.target instanceof HTMLElement) {
        if (e.target.closest("input, textarea, select, [role='textbox']")) return;
      }
      if (e.key === "r" || e.key === "R") {
        if (!lockView) {
          resetView();
          e.preventDefault();
        }
      }
    },
    [lockView],
  );

  // ── JSX ───────────────────────────────────────────────────────────────
  const canvasBox = {
    position: "relative" as const,
    border: `1px solid ${themeColors.border}`,
    overflow: "hidden",
    width: canvasSize,
    height: canvasSize,
    bgcolor: "#000",
  };

  const nA = subAIndices ? subAIndices.length : 0;
  const nB = subBIndices ? subBIndices.length : 0;
  const nPol = polarizationVectors ? polarizationVectors.length / 4 : 0;

  return (
    <Box
      ref={rootRef}
      sx={{
        p: `${SPACING.LG}px`,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        outline: "none",
      }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {/* Header */}
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: `${SPACING.SM}px` }}
      >
        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Atom Finder"}</Typography>
        <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
          {nAtoms} atoms
          {nSublattices === 2 ? `  (A:${nA} / B:${nB})` : ""}
          {polarizationActive && nPol > 0 ? `  pol:${nPol}` : ""}
        </Typography>
      </Stack>

      {/* Canvas */}
      <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
        {width}×{height}
        {cursorInfo && (
          <span style={{ marginLeft: 8, color: themeColors.accent }}>
            ({cursorInfo.row}, {cursorInfo.col}) {formatStat(cursorInfo.value)}
          </span>
        )}
      </Typography>
      <Box ref={containerRef} sx={canvasBox}>
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: canvasSize,
            height: canvasSize,
            imageRendering: "pixelated",
          }}
        />
        <canvas
          ref={uiRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: canvasSize,
            height: canvasSize,
            pointerEvents: "none",
          }}
        />
        <canvas
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: canvasSize,
            height: canvasSize,
            cursor: isDraggingRef.current ? "grabbing" : "grab",
            opacity: 0,
          }}
          width={canvasSize}
          height={canvasSize}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
          onDoubleClick={resetView}
        />
        {!hideView && (
          <Box
            onMouseDown={handleCanvasResizeStart}
            sx={{
              position: "absolute",
              bottom: 0,
              right: 0,
              width: 16,
              height: 16,
              cursor: lockView ? "default" : "nwse-resize",
              opacity: lockView ? 0.2 : 0.6,
              background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
              "&:hover": { opacity: lockView ? 0.2 : 1 },
            }}
          />
        )}
      </Box>

      {/* Stats */}
      {!hideStats && showStats && (
        <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2 }}>
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            Mean{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {formatStat(statsMean)}
            </Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            Min{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {formatStat(statsMin)}
            </Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            Max{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {formatStat(statsMax)}
            </Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            Std{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {formatStat(statsStd)}
            </Box>
          </Typography>
        </Box>
      )}

      {/* Controls */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap" }}>
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Colormap:</Typography>
                <Select
                  size="small"
                  value={cmap}
                  onChange={(e) => {
                    if (!lockDisplay) setCmap(String(e.target.value));
                  }}
                  sx={themedSelect}
                  MenuProps={themedMenuProps}
                  disabled={lockDisplay}
                >
                  {COLORMAP_NAMES.map((n) => (
                    <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>
                      {n}
                    </MenuItem>
                  ))}
                </Select>
              </Box>
            )}

            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Log:</Typography>
                <Switch
                  size="small"
                  checked={logScale}
                  onChange={(_, v) => {
                    if (!lockDisplay) setLogScale(v);
                  }}
                  sx={switchStyles.small}
                  disabled={lockDisplay}
                />
                <Typography sx={typography.label}>Auto:</Typography>
                <Switch
                  size="small"
                  checked={autoContrast}
                  onChange={(_, v) => {
                    if (!lockDisplay) setAutoContrast(v);
                  }}
                  sx={switchStyles.small}
                  disabled={lockDisplay}
                />
              </Box>
            )}

            {!hideHistogram && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Range:</Typography>
                <Histogram
                  data={histData}
                  vminPct={vminPct}
                  vmaxPct={vmaxPct}
                  onRangeChange={(mn, mx) => {
                    if (!lockHistogram) {
                      setVminPct(mn);
                      setVmaxPct(mx);
                    }
                  }}
                  theme={themeInfo.theme}
                />
              </Box>
            )}
          </Stack>

          {/* Detection params */}
          {!hideDetection && (
            <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap", mt: `${SPACING.XS}px` }}>
              <Box sx={controlRow}>
                <Typography sx={typography.label}>min σ:</Typography>
                <Box sx={{ width: 90 }}>
                  <Slider
                    size="small"
                    value={minSigma}
                    min={0.5}
                    max={20}
                    step={0.5}
                    onChange={(_, v) => {
                      if (!lockDetection) setMinSigma(Array.isArray(v) ? v[0] : v);
                    }}
                    disabled={lockDetection}
                  />
                </Box>
                <Typography sx={typography.value}>{minSigma.toFixed(1)}</Typography>
              </Box>
              <Box sx={controlRow}>
                <Typography sx={typography.label}>max σ:</Typography>
                <Box sx={{ width: 90 }}>
                  <Slider
                    size="small"
                    value={maxSigma}
                    min={1}
                    max={40}
                    step={0.5}
                    onChange={(_, v) => {
                      if (!lockDetection) setMaxSigma(Array.isArray(v) ? v[0] : v);
                    }}
                    disabled={lockDetection}
                  />
                </Box>
                <Typography sx={typography.value}>{maxSigma.toFixed(1)}</Typography>
              </Box>
              <Box sx={controlRow}>
                <Typography sx={typography.label}>thr:</Typography>
                <Box sx={{ width: 100 }}>
                  <Slider
                    size="small"
                    value={blobThreshold}
                    min={0.001}
                    max={0.5}
                    step={0.001}
                    onChange={(_, v) => {
                      if (!lockDetection) setBlobThreshold(Array.isArray(v) ? v[0] : v);
                    }}
                    disabled={lockDetection}
                  />
                </Box>
                <Typography sx={typography.value}>{blobThreshold.toFixed(3)}</Typography>
              </Box>
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Subpixel:</Typography>
                <Switch
                  size="small"
                  checked={fitSubpixel}
                  onChange={(_, v) => {
                    if (!lockDetection) setFitSubpixel(v);
                  }}
                  sx={switchStyles.small}
                  disabled={lockDetection}
                />
                {fitSubpixel && (
                  <>
                    <Typography sx={typography.label}>r:</Typography>
                    <Box sx={{ width: 80 }}>
                      <Slider
                        size="small"
                        value={maskRadiusPx}
                        min={2}
                        max={30}
                        step={0.5}
                        onChange={(_, v) => {
                          if (!lockDetection) setMaskRadiusPx(Array.isArray(v) ? v[0] : v);
                        }}
                        disabled={lockDetection}
                      />
                    </Box>
                    <Typography sx={typography.value}>{maskRadiusPx.toFixed(1)}</Typography>
                  </>
                )}
              </Box>
            </Stack>
          )}

          {/* Sublattice + polarization */}
          {!hideSublattice && (
            <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap", mt: `${SPACING.XS}px` }}>
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Sublattices:</Typography>
                <Select
                  size="small"
                  value={nSublattices}
                  onChange={(e) => {
                    if (!lockSublattice) setNSublattices(Number(e.target.value));
                  }}
                  sx={{ ...themedSelect, minWidth: 50 }}
                  MenuProps={themedMenuProps}
                  disabled={lockSublattice}
                >
                  <MenuItem value={1} sx={{ fontSize: 10 }}>1</MenuItem>
                  <MenuItem value={2} sx={{ fontSize: 10 }}>2</MenuItem>
                </Select>
                {nSublattices === 2 && (
                  <>
                    <Typography sx={typography.label}>A frac:</Typography>
                    <Box sx={{ width: 90 }}>
                      <Slider
                        size="small"
                        value={sublatticeFraction}
                        min={0.1}
                        max={0.9}
                        step={0.01}
                        onChange={(_, v) => {
                          if (!lockSublattice)
                            setSublatticeFraction(Array.isArray(v) ? v[0] : v);
                        }}
                        disabled={lockSublattice}
                      />
                    </Box>
                    <Typography sx={typography.value}>{sublatticeFraction.toFixed(2)}</Typography>
                  </>
                )}
              </Box>
              {nSublattices === 2 && !hidePolarization && (
                <Box sx={controlRow}>
                  <Typography sx={typography.label}>Polarization:</Typography>
                  <Switch
                    size="small"
                    checked={polarizationActive}
                    onChange={(_, v) => {
                      if (!lockPolarization) setPolarizationActive(v);
                    }}
                    sx={switchStyles.small}
                    disabled={lockPolarization}
                  />
                  {polarizationActive && (
                    <>
                      <Typography sx={typography.label}>scale:</Typography>
                      <Box sx={{ width: 80 }}>
                        <Slider
                          size="small"
                          value={polarizationScale}
                          min={0.5}
                          max={50}
                          step={0.5}
                          onChange={(_, v) => {
                            if (!lockPolarization)
                              setPolarizationScale(Array.isArray(v) ? v[0] : v);
                          }}
                          disabled={lockPolarization}
                        />
                      </Box>
                      <Typography sx={typography.value}>{polarizationScale.toFixed(1)}</Typography>
                    </>
                  )}
                </Box>
              )}
            </Stack>
          )}
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(AtomFinder);
