/**
 * OrientationMap — Per-scan-pixel template-matching orientation viewer.
 *
 * Displays a single RGB canvas: hue = best-match rotation,
 * value = best-match correlation score (when show_score is on).
 * On hover, reads the rotation and score for the underlying scan pixel.
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
import { computeToolVisibility } from "../tool-parity";
import "./orientation_map.css";

// ============================================================================
// Style constants
// ============================================================================

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
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

// ============================================================================
// Helper: format a value for stat-like display
// ============================================================================

function formatStat(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// ============================================================================
// Main component
// ============================================================================

function OrientationMap() {
  const { themeInfo, colors: themeColors } = useTheme();

  const themedSelect = {
    "& .MuiSelect-select": { py: 0.25, px: 1, fontSize: 10, color: themeColors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
    bgcolor: themeColors.controlBg,
    minWidth: 80,
  };
  const themedMenuProps = {
    anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
    transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
    PaperProps: {
      sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` },
    },
  };

  // ── Model state ─────────────────────────────────────────────────────
  const [title] = useModelState<string>("title");
  const [shapeRows] = useModelState<number>("shape_rows");
  const [shapeCols] = useModelState<number>("shape_cols");
  const [nTemplates] = useModelState<number>("n_templates");
  const [rgbBytes] = useModelState<DataView>("rgb_bytes");
  const [orientationRadBytes] = useModelState<DataView>("orientation_rad_bytes");
  const [scoreBytes] = useModelState<DataView>("score_bytes");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [showScore, setShowScore] = useModelState<boolean>("show_score");
  const [scoreThreshold, setScoreThreshold] = useModelState<number>("score_threshold");
  const [scoreMin] = useModelState<number>("score_min");
  const [scoreMax] = useModelState<number>("score_max");
  const [nQ] = useModelState<number>("n_q");
  const [nTheta] = useModelState<number>("n_theta");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("OrientationMap", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideStats = toolVisibility.isHidden("stats");
  const hideDisplay = toolVisibility.isHidden("display");
  const hideThreshold = toolVisibility.isHidden("threshold");
  const hideView = toolVisibility.isHidden("view");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockThreshold = toolVisibility.isLocked("threshold");
  const lockView = toolVisibility.isLocked("view");

  // ── Local UI state ──────────────────────────────────────────────────
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_MIN);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; rotDeg: number; score: number } | null>(null);

  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const offscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [version, setVersion] = React.useState(0);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Cache decoded arrays for hover readout
  const orientationArrRef = React.useRef<Float32Array | null>(null);
  const scoreArrRef = React.useRef<Float32Array | null>(null);

  React.useEffect(() => {
    if (!orientationRadBytes || !orientationRadBytes.byteLength) {
      orientationArrRef.current = null;
      return;
    }
    orientationArrRef.current = new Float32Array(
      orientationRadBytes.buffer,
      orientationRadBytes.byteOffset,
      orientationRadBytes.byteLength / 4,
    );
  }, [orientationRadBytes]);

  React.useEffect(() => {
    if (!scoreBytes || !scoreBytes.byteLength) {
      scoreArrRef.current = null;
      return;
    }
    scoreArrRef.current = new Float32Array(
      scoreBytes.buffer,
      scoreBytes.byteOffset,
      scoreBytes.byteLength / 4,
    );
  }, [scoreBytes]);

  // ── Decode RGB bytes into the offscreen canvas ──────────────────────
  React.useEffect(() => {
    if (!rgbBytes || !rgbBytes.byteLength) return;
    if (shapeRows <= 0 || shapeCols <= 0) return;
    const expected = shapeRows * shapeCols * 3;
    if (rgbBytes.byteLength !== expected) return;
    let offscreen = offscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      offscreenRef.current = offscreen;
    }
    offscreen.width = shapeCols;
    offscreen.height = shapeRows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(shapeCols, shapeRows);
    const src = new Uint8Array(rgbBytes.buffer, rgbBytes.byteOffset, rgbBytes.byteLength);
    const dst = imgData.data;
    let si = 0;
    let di = 0;
    const n = shapeRows * shapeCols;
    for (let p = 0; p < n; p++) {
      dst[di] = src[si];
      dst[di + 1] = src[si + 1];
      dst[di + 2] = src[si + 2];
      dst[di + 3] = 255;
      si += 3;
      di += 4;
    }
    ctx.putImageData(imgData, 0, 0);
    setVersion((v) => v + 1);
  }, [rgbBytes, shapeRows, shapeCols]);

  // ── Main canvas draw ────────────────────────────────────────────────
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const offscreen = offscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * zoom) / 2 + panX;
    const offY = (canvasSize - canvasSize * zoom) / 2 + panY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * zoom, canvasSize * zoom);
  }, [version, zoom, panX, panY, canvasSize]);

  // ── UI overlay (zoom indicator) ─────────────────────────────────────
  React.useLayoutEffect(() => {
    const canvas = uiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    if (zoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, 8, cssW - 8);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [version, zoom, panX, panY, canvasSize]);

  // ── Resize handle ───────────────────────────────────────────────────
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
  }, [isResizingCanvas, resizeCanvasStart, canvasSize]);

  // ── Mouse handlers ──────────────────────────────────────────────────
  const isDragging = React.useRef(false);
  const dragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const toImage = (e: React.MouseEvent): { row: number; col: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const offX = (canvasSize - canvasSize * zoom) / 2 + panX;
    const offY = (canvasSize - canvasSize * zoom) / 2 + panY;
    const col = ((mx - offX) / (canvasSize * zoom)) * shapeCols;
    const row = ((my - offY) / (canvasSize * zoom)) * shapeRows;
    return { row, col };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 2 || e.shiftKey) {
      isDragging.current = true;
      dragStart.current = { x: e.clientX, y: e.clientY, panX, panY };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) {
      setPanX(dragStart.current.panX + (e.clientX - dragStart.current.x));
      setPanY(dragStart.current.panY + (e.clientY - dragStart.current.y));
      return;
    }
    const { row, col } = toImage(e);
    const r = Math.floor(row);
    const c = Math.floor(col);
    if (r < 0 || r >= shapeRows || c < 0 || c >= shapeCols) {
      setCursorInfo(null);
      return;
    }
    const idx = r * shapeCols + c;
    const rot = orientationArrRef.current ? orientationArrRef.current[idx] : 0;
    const sc = scoreArrRef.current ? scoreArrRef.current[idx] : 0;
    setCursorInfo({ row: r, col: c, rotDeg: (rot * 180) / Math.PI, score: sc });
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };
  const handleMouseLeave = () => {
    isDragging.current = false;
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

  // Wheel scroll prevention
  React.useEffect(() => {
    const prevent = (e: WheelEvent) => e.preventDefault();
    const el = containerRef.current;
    if (el) el.addEventListener("wheel", prevent, { passive: false });
    return () => {
      if (el) el.removeEventListener("wheel", prevent);
    };
  }, []);

  // ── Score range for the threshold slider ────────────────────────────
  // The slider operates in raw score units. Min/max come from the
  // Python-computed score_min/score_max so threshold is interpretable.
  const sliderMin = scoreMin;
  const sliderMax = Math.max(scoreMax, scoreMin + 1e-6);
  const sliderStep = Math.max((sliderMax - sliderMin) / 200.0, 1e-6);

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
      className="orientation-map-root"
      sx={{ bgcolor: themeColors.bg, color: themeColors.text }}
    >
      {/* Header */}
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: `${SPACING.SM}px` }}
      >
        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>
          {title || "Orientation Map"}
        </Typography>
        <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
          {nTemplates} templates · {nQ}×{nTheta} polar
        </Typography>
      </Stack>

      {/* Hover readout */}
      <Typography
        sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px`, minHeight: 14 }}
      >
        {cursorInfo
          ? `(${cursorInfo.row}, ${cursorInfo.col})  rot = ${cursorInfo.rotDeg.toFixed(1)}°  score = ${formatStat(cursorInfo.score)}`
          : `scan ${shapeRows}×${shapeCols} — hover to inspect`}
      </Typography>

      {/* Canvas */}
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
            cursor: "crosshair",
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

      {/* Stats bar */}
      {!hideStats && showStats && (
        <Box
          sx={{
            mt: `${SPACING.XS}px`,
            px: 1,
            py: 0.25,
            display: "flex",
            gap: 2,
          }}
        >
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            Score min{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {formatStat(scoreMin)}
            </Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            Score max{" "}
            <Box component="span" sx={{ color: themeColors.accent }}>
              {formatStat(scoreMax)}
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
                <Typography sx={typography.label}>Show score:</Typography>
                <Switch
                  size="small"
                  checked={showScore}
                  onChange={(_, v) => {
                    if (!lockDisplay) setShowScore(v);
                  }}
                  sx={switchStyles.small}
                  disabled={lockDisplay}
                />
              </Box>
            )}
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Cmap:</Typography>
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
                  <MenuItem value="hsv" sx={{ fontSize: 10 }}>
                    hsv
                  </MenuItem>
                  <MenuItem value="viridis" sx={{ fontSize: 10 }}>
                    viridis
                  </MenuItem>
                </Select>
              </Box>
            )}
            {!hideThreshold && (
              <Box sx={{ ...controlRow, flexGrow: 1, minWidth: 200 }}>
                <Typography sx={typography.label}>Threshold:</Typography>
                <Slider
                  size="small"
                  value={scoreThreshold}
                  min={sliderMin}
                  max={sliderMax}
                  step={sliderStep}
                  onChange={(_, v) => {
                    if (!lockThreshold) setScoreThreshold(Number(v));
                  }}
                  disabled={lockThreshold}
                  sx={{ color: themeColors.accent, mx: 1, flexGrow: 1, maxWidth: 200 }}
                />
                <Typography sx={typography.value}>{formatStat(scoreThreshold)}</Typography>
              </Box>
            )}
          </Stack>
          <Box sx={{ ...controlRow, mt: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              Hue = best-match rotation · {themeInfo.theme === "dark" ? "dark theme" : "light theme"}
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(OrientationMap);
