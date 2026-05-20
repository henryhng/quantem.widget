/**
 * AberrationExplorer - Live aberration / CTF / probe diagnostics for STEM
 * column tuning.
 *
 * Three panels recompute in real time as the user drags Krivanek polar
 * coefficient sliders:
 *  1) Real-space probe intensity |psi(r)|^2.
 *  2) Aberration phase wheel chi(k, phi) in polar k-space (cyclic colormap).
 *  3) 1D radial CTF sin(chi(k, phi=0)) versus k (mrad).
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
import { extractFloat32, formatNumber } from "../format";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";
import { applyLogScale, findDataRange, percentileClip, sliderRange } from "../stats";
import { computeHistogramFromBytes } from "../histogram";
import "./aberration_explorer.css";

// ============================================================================
// Style constants (match Show3D / ShowComplex2D conventions)
// ============================================================================

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const DPR = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
const DEFAULT_CANVAS_SIZE = 320;

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
  width: "fit-content",
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const sliderSx = {
  width: 100,
  py: 0,
  "& .MuiSlider-thumb": { width: 8, height: 8 },
  "& .MuiSlider-rail": { height: 2 },
  "& .MuiSlider-track": { height: 2 },
  "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
};

// ============================================================================
// Aberration registry — Krivanek polar names + UI metadata
// Magnitudes are angstroms, angles are radians.
// ============================================================================

type AberrationField = {
  key: string;
  label: string;
  kind: "magnitude" | "angle";
  min: number;
  max: number;
  step: number;
};

type AberrationGroup = {
  title: string;
  order: number;
  fields: AberrationField[];
};

const ABERRATION_GROUPS: AberrationGroup[] = [
  {
    title: "1st order",
    order: 1,
    fields: [
      { key: "C10", label: "C10 (defocus)", kind: "magnitude", min: -500, max: 500, step: 1 },
      { key: "C12", label: "C12 (2-fold astig.)", kind: "magnitude", min: 0, max: 500, step: 1 },
      { key: "phi12", label: "phi12", kind: "angle", min: -Math.PI, max: Math.PI, step: 0.01 },
    ],
  },
  {
    title: "2nd order",
    order: 2,
    fields: [
      { key: "C21", label: "C21 (axial coma)", kind: "magnitude", min: 0, max: 50000, step: 50 },
      { key: "phi21", label: "phi21", kind: "angle", min: -Math.PI, max: Math.PI, step: 0.01 },
      { key: "C23", label: "C23 (3-fold astig.)", kind: "magnitude", min: 0, max: 50000, step: 50 },
      { key: "phi23", label: "phi23", kind: "angle", min: -Math.PI, max: Math.PI, step: 0.01 },
    ],
  },
  {
    title: "3rd order",
    order: 3,
    fields: [
      { key: "C30", label: "C30 (sph. aberr.)", kind: "magnitude", min: -1e7, max: 1e7, step: 1e3 },
      { key: "C32", label: "C32 (star)", kind: "magnitude", min: 0, max: 1e7, step: 1e3 },
      { key: "phi32", label: "phi32", kind: "angle", min: -Math.PI, max: Math.PI, step: 0.01 },
      { key: "C34", label: "C34 (4-fold astig.)", kind: "magnitude", min: 0, max: 1e7, step: 1e3 },
      { key: "phi34", label: "phi34", kind: "angle", min: -Math.PI, max: Math.PI, step: 0.01 },
    ],
  },
];

const ALL_ABERRATION_KEYS: string[] = ABERRATION_GROUPS.flatMap((g) => g.fields.map((f) => f.key));

// ============================================================================
// Helper components
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
            maxWidth: 320,
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
// Cyclic HSV colormap for chi phase wheel
// ============================================================================

function hsvToRgbBytes(h: number): [number, number, number] {
  // h in [0, 1)
  const hi = Math.floor(h * 6) % 6;
  const f = h * 6 - Math.floor(h * 6);
  const q = 1 - f;
  let r: number;
  let g: number;
  let b: number;
  switch (hi) {
    case 0:
      r = 1;
      g = f;
      b = 0;
      break;
    case 1:
      r = q;
      g = 1;
      b = 0;
      break;
    case 2:
      r = 0;
      g = 1;
      b = f;
      break;
    case 3:
      r = 0;
      g = q;
      b = 1;
      break;
    case 4:
      r = f;
      g = 0;
      b = 1;
      break;
    default:
      r = 1;
      g = 0;
      b = q;
      break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function renderCyclicChi(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  bgColor: [number, number, number],
  aperturePxRadius: number,
  width: number,
  height: number,
): void {
  // Render chi mod 2*pi as hue. Pixels outside the aperture get the
  // background color so the wheel is visually circular.
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  const rSq = aperturePxRadius * aperturePxRadius;
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const i = row * width + col;
      const dx = col - cx;
      const dy = row - cy;
      const distSq = dx * dx + dy * dy;
      const j = i * 4;
      if (distSq > rSq) {
        rgba[j] = bgColor[0];
        rgba[j + 1] = bgColor[1];
        rgba[j + 2] = bgColor[2];
        rgba[j + 3] = 255;
        continue;
      }
      const chi = data[i];
      // Normalize chi mod 2*pi to [0, 1)
      const TWO_PI = 2 * Math.PI;
      let h = chi - TWO_PI * Math.floor(chi / TWO_PI);
      h = h / TWO_PI;
      const [r, g, b] = hsvToRgbBytes(h);
      rgba[j] = r;
      rgba[j + 1] = g;
      rgba[j + 2] = b;
      rgba[j + 3] = 255;
    }
  }
}

// ============================================================================
// Histogram (probe intensity)
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
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
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
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
      reducedBins.push(sum / Math.max(binRatio, 1));
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  const fmt = (pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    if (!Number.isFinite(val)) return "0";
    return Math.abs(val) >= 1000 || (Math.abs(val) > 0 && Math.abs(val) < 0.01)
      ? val.toExponential(1)
      : val.toFixed(3);
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [lo, hi] = v as number[];
          onRangeChange(Math.min(lo, hi - 1), Math.max(hi, lo + 1));
        }}
        min={0}
        max={100}
        size="small"
        valueLabelDisplay="auto"
        valueLabelFormat={fmt}
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
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {fmt(vminPct)}
        </Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {fmt(vmaxPct)}
        </Typography>
      </Box>
    </Box>
  );
}

// ============================================================================
// Phase wheel ring annotation (axis labels in radians)
// ============================================================================

function drawChiWheelAxes(
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  cutoffMrad: number,
): void {
  const cx = cssW / 2;
  const cy = cssH / 2;
  const r = Math.min(cssW, cssH) / 2 - 4;

  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.6)";
  ctx.shadowBlur = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1;
  // Outer circle = aperture boundary
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Cross axes
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();

  // Labels
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const labelOffset = 12;
  const label = `${cutoffMrad.toFixed(1)} mrad`;
  ctx.fillText(label, cx + r - 20, cy + labelOffset);
  ctx.fillText("0", cx + r + 10, cy);
  ctx.fillText("π/2", cx, cy - r - 10);
  ctx.fillText("π", cx - r - 10, cy);
  ctx.fillText("-π/2", cx, cy + r + 10);
  ctx.restore();
}

// ============================================================================
// 1D CTF panel drawing
// ============================================================================

function drawRadialCTF(
  canvas: HTMLCanvasElement,
  ctf: Float32Array,
  kMaxMrad: number,
  themeIsDark: boolean,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const cssW = parseFloat(canvas.style.width || `${canvas.width}`);
  const cssH = parseFloat(canvas.style.height || `${canvas.height}`);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const bg = themeIsDark ? "#0a0a0a" : "#f8f8f8";
  const axis = themeIsDark ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.45)";
  const gridCol = themeIsDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
  const trace = themeIsDark ? "#4fc3f7" : "#0277bd";
  const zeroLine = themeIsDark ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.35)";

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 40;
  const padR = 10;
  const padT = 10;
  const padB = 26;
  const plotW = Math.max(1, cssW - padL - padR);
  const plotH = Math.max(1, cssH - padT - padB);

  // Y goes from -1.05 to 1.05
  const yMin = -1.05;
  const yMax = 1.05;
  const yToPx = (y: number) => padT + ((yMax - y) / (yMax - yMin)) * plotH;
  const xToPx = (idx: number) => padL + (idx / Math.max(1, ctf.length - 1)) * plotW;

  // Y gridlines at -1, -0.5, 0, 0.5, 1
  ctx.strokeStyle = gridCol;
  ctx.lineWidth = 1;
  const yTicks = [-1, -0.5, 0, 0.5, 1];
  for (const yt of yTicks) {
    const yp = yToPx(yt);
    ctx.beginPath();
    ctx.moveTo(padL, yp);
    ctx.lineTo(padL + plotW, yp);
    ctx.stroke();
  }

  // Zero line emphasised
  ctx.strokeStyle = zeroLine;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, yToPx(0));
  ctx.lineTo(padL + plotW, yToPx(0));
  ctx.stroke();

  // Axes
  ctx.strokeStyle = axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // Trace
  ctx.strokeStyle = trace;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < ctf.length; i++) {
    const x = xToPx(i);
    const y = yToPx(ctf[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = axis;
  ctx.font = "10px -apple-system, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const yt of yTicks) {
    ctx.fillText(yt.toFixed(1), padL - 4, yToPx(yt));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const frac = i / xTicks;
    const kVal = frac * kMaxMrad;
    ctx.fillText(kVal.toFixed(1), padL + frac * plotW, padT + plotH + 4);
  }
  ctx.textAlign = "center";
  ctx.fillText("k (mrad)", padL + plotW / 2, padT + plotH + 16);

  // Y axis title
  ctx.save();
  ctx.translate(12, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("sin(χ)", 0, 0);
  ctx.restore();
}

// ============================================================================
// Aberrations group panel
// ============================================================================

interface AberrationGroupPanelProps {
  group: AberrationGroup;
  aberrations: Record<string, number>;
  onChange: (key: string, value: number) => void;
  themeColors: ReturnType<typeof useTheme>["colors"];
  themedSelect: Record<string, unknown>;
  themedMenuProps: Record<string, unknown>;
  defaultExpanded?: boolean;
}

function AberrationGroupPanel({
  group,
  aberrations,
  onChange,
  themeColors,
  defaultExpanded = true,
}: AberrationGroupPanelProps) {
  const [expanded, setExpanded] = React.useState(defaultExpanded);
  const anyActive = group.fields.some(
    (f) => f.kind === "magnitude" && (aberrations[f.key] || 0) !== 0,
  );
  return (
    <Box
      sx={{
        border: `1px solid ${themeColors.border}`,
        bgcolor: themeColors.controlBg,
        px: 1,
        py: 0.5,
      }}
    >
      <Box
        sx={{ display: "flex", alignItems: "center", gap: 1, cursor: "pointer" }}
        onClick={() => setExpanded((v) => !v)}
      >
        <Typography sx={{ fontSize: 10, fontWeight: "bold", color: themeColors.accent }}>
          {expanded ? "▾" : "▸"} {group.title}
        </Typography>
        {anyActive && (
          <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>(active)</Typography>
        )}
      </Box>
      {expanded && (
        <Box sx={{ mt: 0.5, display: "flex", flexDirection: "column", gap: 0.5 }}>
          {group.fields.map((field) => {
            const val = aberrations[field.key] ?? 0;
            return (
              <Box
                key={field.key}
                sx={{ display: "flex", alignItems: "center", gap: 1 }}
              >
                <Typography
                  sx={{
                    ...typography.labelSmall,
                    minWidth: 110,
                    color: themeColors.text,
                    fontFamily: "monospace",
                  }}
                >
                  {field.label}
                </Typography>
                <Slider
                  value={val}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  onChange={(_, v) => onChange(field.key, v as number)}
                  size="small"
                  sx={sliderSx}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(v: number) =>
                    field.kind === "angle"
                      ? `${v.toFixed(2)} rad`
                      : Math.abs(v) >= 1000
                      ? v.toExponential(1)
                      : v.toFixed(2)
                  }
                />
                <Typography
                  sx={{
                    ...typography.value,
                    minWidth: 70,
                    textAlign: "right",
                    color: val !== 0 ? themeColors.accent : themeColors.textMuted,
                  }}
                >
                  {field.kind === "angle"
                    ? `${val.toFixed(2)}`
                    : Math.abs(val) >= 1000
                    ? val.toExponential(1)
                    : val.toFixed(2)}
                </Typography>
              </Box>
            );
          })}
        </Box>
      )}
    </Box>
  );
}

// ============================================================================
// Main component
// ============================================================================

function AberrationExplorer() {
  const { themeInfo, colors: themeColors } = useTheme();

  const themedSelect = React.useMemo(
    () => ({
      fontSize: 10,
      bgcolor: themeColors.controlBg,
      color: themeColors.text,
      "& .MuiSelect-select": { py: 0.5 },
      "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
      "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
    }),
    [themeColors],
  );

  const themedMenuProps = React.useMemo(
    () => ({
      ...upwardMenuProps,
      PaperProps: {
        sx: {
          bgcolor: themeColors.controlBg,
          color: themeColors.text,
          border: `1px solid ${themeColors.border}`,
        },
      },
    }),
    [themeColors],
  );

  // ---- Model state -----
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [energyKeV, setEnergyKeV] = useModelState<number>("energy_keV");
  const [semiangle, setSemiangle] = useModelState<number>("semiangle_cutoff_mrad");
  const [gpts, setGpts] = useModelState<number>("gpts");
  const [sampling, setSampling] = useModelState<number>("real_space_sampling_A");
  const [apertureSmoothing, setApertureSmoothing] = useModelState<boolean>("aperture_smoothing");
  const [defocusSpread, setDefocusSpread] = useModelState<number>("defocus_spread_A");
  const [aberrationsModel, setAberrationsModel] =
    useModelState<Record<string, number>>("aberrations");

  const [probeBytes] = useModelState<DataView>("probe_intensity_bytes");
  const [chiBytes] = useModelState<DataView>("chi_polar_bytes");
  const [radialBytes] = useModelState<DataView>("radial_ctf_bytes");

  const [radialKMaxMrad] = useModelState<number>("radial_k_max_mrad");
  const [chiMin] = useModelState<number>("chi_min");
  const [chiMax] = useModelState<number>("chi_max");
  const [realSpaceExtent] = useModelState<number>("real_space_extent_A");
  const [wavelengthA] = useModelState<number>("wavelength_A");

  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [canvasSize] = useModelState<number>("canvas_size");

  // ---- Local UI state -----
  const [logScale, setLogScale] = React.useState<boolean>(true);
  const [autoContrast, setAutoContrast] = React.useState<boolean>(true);
  const [percentileLow] = React.useState<number>(1.0);
  const [percentileHigh] = React.useState<number>(99.0);
  const [vminPct, setVminPct] = React.useState<number>(0);
  const [vmaxPct, setVmaxPct] = React.useState<number>(100);

  // Track which aberration sliders the user is currently dragging — we only
  // commit to the Python model on slider commit to avoid 60-Hz round-trips.
  const liveAberrationsRef = React.useRef<Record<string, number>>({});
  React.useEffect(() => {
    liveAberrationsRef.current = { ...aberrationsModel };
  }, [aberrationsModel]);

  // ---- Decoded data refs -----
  const probeFloat = React.useMemo(() => extractFloat32(probeBytes), [probeBytes]);
  const chiFloat = React.useMemo(() => extractFloat32(chiBytes), [chiBytes]);
  const radialFloat = React.useMemo(() => extractFloat32(radialBytes), [radialBytes]);

  // The probe array is (gpts × gpts); chi is rendered on its own dedicated
  // square grid that fits the aperture exactly. Derive both sizes from the
  // bytes themselves so the JS doesn't desync if Python recomputes.
  const gridSize = React.useMemo(() => {
    if (!probeFloat || probeFloat.length === 0) return gpts;
    return Math.round(Math.sqrt(probeFloat.length));
  }, [probeFloat, gpts]);
  const chiSize = React.useMemo(() => {
    if (!chiFloat || chiFloat.length === 0) return gpts;
    return Math.round(Math.sqrt(chiFloat.length));
  }, [chiFloat, gpts]);

  // ---- Canvas refs -----
  const probeCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const probeUiRef = React.useRef<HTMLCanvasElement>(null);
  const chiCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const chiUiRef = React.useRef<HTMLCanvasElement>(null);
  const ctfCanvasRef = React.useRef<HTMLCanvasElement>(null);

  const canvasW = canvasSize > 0 ? canvasSize : DEFAULT_CANVAS_SIZE;
  const canvasH = canvasW;

  // ---- Probe intensity histogram + colormap -----
  const probeProcessed = React.useMemo(() => {
    if (!probeFloat) return null;
    return logScale ? applyLogScale(probeFloat) : probeFloat;
  }, [probeFloat, logScale]);

  const probeRange = React.useMemo(() => {
    if (!probeProcessed) return { min: 0, max: 1 };
    return findDataRange(probeProcessed);
  }, [probeProcessed]);

  React.useEffect(() => {
    const canvas = probeCanvasRef.current;
    if (!canvas || !probeProcessed || gridSize <= 0) return;
    canvas.width = gridSize;
    canvas.height = gridSize;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(gridSize, gridSize);
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    let vmin: number;
    let vmax: number;
    if (autoContrast) {
      const pc = percentileClip(probeProcessed, percentileLow, percentileHigh);
      vmin = pc.vmin;
      vmax = pc.vmax;
    } else {
      const sr = sliderRange(probeRange.min, probeRange.max, vminPct, vmaxPct);
      vmin = sr.vmin;
      vmax = sr.vmax;
    }
    applyColormap(probeProcessed, imgData.data, lut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
  }, [probeProcessed, cmap, autoContrast, percentileLow, percentileHigh, vminPct, vmaxPct, probeRange, gridSize]);

  // ---- Probe UI overlay (scale bar / pixel-size label) -----
  React.useEffect(() => {
    const canvas = probeUiRef.current;
    if (!canvas) return;
    canvas.width = Math.round(canvasW * DPR);
    canvas.height = Math.round(canvasH * DPR);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(DPR, DPR);
    if (realSpaceExtent > 0) {
      // Simple scale bar = 1/4 of FOV, rounded to nice value.
      const fovA = realSpaceExtent;
      const candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
      const target = fovA / 4;
      let chosen = candidates[0];
      for (const c of candidates) if (c <= target) chosen = c;
      const barLenPx = (chosen / fovA) * canvasW;
      const barThickness = 5;
      const margin = 12;
      const x0 = margin;
      const y0 = canvasH - margin - barThickness;
      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur = 2;
      ctx.fillStyle = "white";
      ctx.fillRect(x0, y0, barLenPx, barThickness);
      ctx.font = "12px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      const labelText =
        chosen >= 10 ? `${(chosen / 10).toFixed(1)} nm` : `${chosen.toFixed(1)} Å`;
      ctx.fillText(labelText, x0, y0 - 4);
      ctx.restore();
    }
    ctx.restore();
  }, [canvasW, canvasH, realSpaceExtent]);

  // ---- Chi phase wheel -----
  React.useEffect(() => {
    const canvas = chiCanvasRef.current;
    if (!canvas || !chiFloat || chiSize <= 0) return;
    canvas.width = chiSize;
    canvas.height = chiSize;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(chiSize, chiSize);
    const bgColor: [number, number, number] =
      themeInfo.theme === "dark" ? [12, 12, 12] : [248, 248, 248];
    const aperturePxRadius = chiSize / 2 - 0.5;
    renderCyclicChi(chiFloat, imgData.data, bgColor, aperturePxRadius, chiSize, chiSize);
    ctx.putImageData(imgData, 0, 0);
  }, [chiFloat, chiSize, themeInfo.theme]);

  // ---- Chi UI overlay (axes + cutoff label) -----
  React.useEffect(() => {
    const canvas = chiUiRef.current;
    if (!canvas) return;
    canvas.width = Math.round(canvasW * DPR);
    canvas.height = Math.round(canvasH * DPR);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(DPR, DPR);
    drawChiWheelAxes(ctx, canvasW, canvasH, semiangle);
    ctx.restore();
  }, [canvasW, canvasH, semiangle]);

  // ---- Radial CTF -----
  React.useEffect(() => {
    const canvas = ctfCanvasRef.current;
    if (!canvas || !radialFloat) return;
    canvas.style.width = `${canvasW}px`;
    canvas.style.height = `${Math.round(canvasH * 0.55)}px`;
    drawRadialCTF(canvas, radialFloat, radialKMaxMrad, themeInfo.theme === "dark");
  }, [radialFloat, radialKMaxMrad, canvasW, canvasH, themeInfo.theme]);

  // ---- Aberration handlers -----
  const handleAberrationChange = React.useCallback(
    (key: string, value: number) => {
      const next = { ...liveAberrationsRef.current, [key]: value };
      liveAberrationsRef.current = next;
      setAberrationsModel(next);
    },
    [setAberrationsModel],
  );

  const handleResetAberrations = React.useCallback(() => {
    const cleared: Record<string, number> = {};
    for (const key of ALL_ABERRATION_KEYS) cleared[key] = 0;
    liveAberrationsRef.current = cleared;
    setAberrationsModel(cleared);
  }, [setAberrationsModel]);

  // ============================================================================
  // Render
  // ============================================================================

  const borderColor = themeColors.border;

  return (
    <Box
      className="aberration-explorer-root"
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, overflow: "visible" }}
    >
      {/* Title row */}
      <Typography
        variant="caption"
        sx={{
          ...typography.label,
          color: themeColors.accent,
          mb: `${SPACING.XS}px`,
          display: "block",
        }}
      >
        {title || "AberrationExplorer"}
        <InfoTooltip
          theme={themeInfo.theme}
          text={
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Panels</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                Left: real-space probe intensity |ψ(r)|². Middle: aberration phase χ(k, φ) on
                a polar k-space wheel (cyclic colormap, mod 2π). Right: radial CTF sin(χ(k))
                along φ=0 vs k.
              </Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>
                Conventions
              </Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                Magnitudes in Å, angles in radians. C10 sign follows Krivanek
                (positive C10 ≡ −defocus).
              </Typography>
            </Box>
          }
        />
      </Typography>

      {/* Three-panel row */}
      <Stack direction="row" spacing={`${SPACING.LG}px`} alignItems="flex-start">
        {/* Panel 1: real-space probe intensity */}
        <Box>
          <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, mb: 0.5 }}>
            Probe intensity |ψ(r)|²
          </Typography>
          <Box
            sx={{
              position: "relative",
              border: `1px solid ${borderColor}`,
              bgcolor: "#000",
              width: canvasW,
              height: canvasH,
            }}
          >
            <canvas
              ref={probeCanvasRef}
              width={gridSize || 1}
              height={gridSize || 1}
              style={{
                width: canvasW,
                height: canvasH,
                imageRendering: "pixelated",
              }}
            />
            <canvas
              ref={probeUiRef}
              width={Math.round(canvasW * DPR)}
              height={Math.round(canvasH * DPR)}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: canvasW,
                height: canvasH,
                pointerEvents: "none",
              }}
            />
          </Box>
          {showStats && (
            <Box
              sx={{
                mt: `${SPACING.XS}px`,
                px: 1,
                py: 0.5,
                bgcolor: themeColors.bgAlt,
                display: "flex",
                gap: 1.5,
                alignItems: "center",
                boxSizing: "border-box",
                overflow: "hidden",
                whiteSpace: "nowrap",
              }}
            >
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                Mean{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMean)}
                </Box>
              </Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                Min{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMin)}
                </Box>
              </Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                Max{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMax)}
                </Box>
              </Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                Std{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsStd)}
                </Box>
              </Typography>
            </Box>
          )}
          {/* Display controls + histogram */}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px` }}>
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  gap: `${SPACING.XS}px`,
                  flex: 1,
                }}
              >
                <Box
                  sx={{
                    ...controlRow,
                    border: `1px solid ${borderColor}`,
                    bgcolor: themeColors.controlBg,
                  }}
                >
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select
                    value={cmap}
                    onChange={(e) => setCmap(e.target.value)}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 70 }}
                    MenuProps={themedMenuProps}
                  >
                    {COLORMAP_NAMES.filter((n) => n !== "hsv").map((name) => (
                      <MenuItem key={name} value={name}>
                        {name.charAt(0).toUpperCase() + name.slice(1)}
                      </MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
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
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch
                    checked={autoContrast}
                    onChange={(e) => setAutoContrast(e.target.checked)}
                    size="small"
                    sx={switchStyles.small}
                  />
                </Box>
              </Box>
              <Histogram
                data={probeProcessed}
                vminPct={vminPct}
                vmaxPct={vmaxPct}
                onRangeChange={(lo, hi) => {
                  setVminPct(lo);
                  setVmaxPct(hi);
                  setAutoContrast(false);
                }}
                width={110}
                height={50}
                theme={themeInfo.theme}
                dataMin={probeRange.min}
                dataMax={probeRange.max}
              />
            </Box>
          )}
        </Box>

        {/* Panel 2: chi phase wheel */}
        <Box>
          <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, mb: 0.5 }}>
            χ(k, φ) — cyclic phase wheel (mod 2π)
          </Typography>
          <Box
            sx={{
              position: "relative",
              border: `1px solid ${borderColor}`,
              bgcolor: themeInfo.theme === "dark" ? "#0c0c0c" : "#f8f8f8",
              width: canvasW,
              height: canvasH,
            }}
          >
            <canvas
              ref={chiCanvasRef}
              width={chiSize || 1}
              height={chiSize || 1}
              style={{
                width: canvasW,
                height: canvasH,
                imageRendering: "pixelated",
              }}
            />
            <canvas
              ref={chiUiRef}
              width={Math.round(canvasW * DPR)}
              height={Math.round(canvasH * DPR)}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: canvasW,
                height: canvasH,
                pointerEvents: "none",
              }}
            />
          </Box>
          {showStats && (
            <Box
              sx={{
                mt: `${SPACING.XS}px`,
                px: 1,
                py: 0.5,
                bgcolor: themeColors.bgAlt,
                display: "flex",
                gap: 1.5,
                alignItems: "center",
                boxSizing: "border-box",
                whiteSpace: "nowrap",
              }}
            >
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                χ min{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(chiMin)}
                </Box>{" "}
                rad
              </Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                χ max{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(chiMax)}
                </Box>{" "}
                rad
              </Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                λ{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {wavelengthA.toFixed(4)}
                </Box>{" "}
                Å
              </Typography>
            </Box>
          )}
        </Box>

        {/* Panel 3: radial CTF */}
        <Box sx={{ flex: 1, minWidth: canvasW }}>
          <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, mb: 0.5 }}>
            Radial CTF sin(χ(k, φ=0))
          </Typography>
          <Box sx={{ border: `1px solid ${borderColor}`, width: canvasW }}>
            <canvas
              ref={ctfCanvasRef}
              style={{ width: canvasW, height: Math.round(canvasH * 0.55), display: "block" }}
            />
          </Box>
          <Box
            sx={{
              mt: `${SPACING.XS}px`,
              px: 1,
              py: 0.5,
              bgcolor: themeColors.bgAlt,
              display: "flex",
              gap: 1.5,
              alignItems: "center",
              boxSizing: "border-box",
              whiteSpace: "nowrap",
            }}
          >
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
              k_max{" "}
              <Box component="span" sx={{ color: themeColors.accent }}>
                {radialKMaxMrad.toFixed(2)}
              </Box>{" "}
              mrad
            </Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
              FOV{" "}
              <Box component="span" sx={{ color: themeColors.accent }}>
                {realSpaceExtent.toFixed(2)}
              </Box>{" "}
              Å
            </Typography>
          </Box>
        </Box>
      </Stack>

      {/* Microscope / sampling controls */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.MD}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
          <Box
            sx={{
              ...controlRow,
              border: `1px solid ${borderColor}`,
              bgcolor: themeColors.controlBg,
              flexWrap: "wrap",
              rowGap: 0.5,
            }}
          >
            <Typography sx={{ ...typography.label, fontSize: 10 }}>E:</Typography>
            <Slider
              value={energyKeV}
              min={20}
              max={400}
              step={1}
              onChange={(_, v) => setEnergyKeV(v as number)}
              size="small"
              sx={{ ...sliderSx, width: 100 }}
              valueLabelDisplay="auto"
              valueLabelFormat={(v: number) => `${v.toFixed(0)} keV`}
            />
            <Typography sx={{ ...typography.value, minWidth: 56 }}>
              {energyKeV.toFixed(0)} keV
            </Typography>

            <Typography sx={{ ...typography.label, fontSize: 10 }}>α:</Typography>
            <Slider
              value={semiangle}
              min={1}
              max={80}
              step={0.5}
              onChange={(_, v) => setSemiangle(v as number)}
              size="small"
              sx={{ ...sliderSx, width: 100 }}
              valueLabelDisplay="auto"
              valueLabelFormat={(v: number) => `${v.toFixed(1)} mrad`}
            />
            <Typography sx={{ ...typography.value, minWidth: 60 }}>
              {semiangle.toFixed(1)} mrad
            </Typography>

            <Typography sx={{ ...typography.label, fontSize: 10 }}>gpts:</Typography>
            <Select
              value={gpts}
              onChange={(e) => setGpts(Number(e.target.value))}
              size="small"
              sx={{ ...themedSelect, minWidth: 60 }}
              MenuProps={themedMenuProps}
            >
              <MenuItem value={128}>128</MenuItem>
              <MenuItem value={256}>256</MenuItem>
              <MenuItem value={512}>512</MenuItem>
            </Select>

            <Typography sx={{ ...typography.label, fontSize: 10 }}>dx:</Typography>
            <Slider
              value={sampling}
              min={0.02}
              max={0.5}
              step={0.01}
              onChange={(_, v) => setSampling(v as number)}
              size="small"
              sx={{ ...sliderSx, width: 100 }}
              valueLabelDisplay="auto"
              valueLabelFormat={(v: number) => `${v.toFixed(2)} Å`}
            />
            <Typography sx={{ ...typography.value, minWidth: 60 }}>
              {sampling.toFixed(2)} Å/px
            </Typography>

            <Typography sx={{ ...typography.label, fontSize: 10 }}>Δf σ:</Typography>
            <Slider
              value={defocusSpread}
              min={0}
              max={100}
              step={1}
              onChange={(_, v) => setDefocusSpread(v as number)}
              size="small"
              sx={{ ...sliderSx, width: 100 }}
              valueLabelDisplay="auto"
              valueLabelFormat={(v: number) => `${v.toFixed(0)} Å`}
            />
            <Typography sx={{ ...typography.value, minWidth: 48 }}>
              {defocusSpread.toFixed(0)} Å
            </Typography>

            <Typography sx={{ ...typography.label, fontSize: 10 }}>Soft ap.:</Typography>
            <Switch
              checked={Boolean(apertureSmoothing)}
              onChange={(e) => setApertureSmoothing(e.target.checked)}
              size="small"
              sx={switchStyles.small}
            />

            <Box sx={{ flex: 1 }} />
            <Button size="small" sx={compactButton} onClick={handleResetAberrations}>
              Reset aberrations
            </Button>
          </Box>

          {/* Aberration groups */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
            {ABERRATION_GROUPS.map((group) => (
              <AberrationGroupPanel
                key={group.title}
                group={group}
                aberrations={aberrationsModel}
                onChange={handleAberrationChange}
                themeColors={themeColors}
                themedSelect={themedSelect}
                themedMenuProps={themedMenuProps}
                defaultExpanded={group.order <= 2}
              />
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(AberrationExplorer);
