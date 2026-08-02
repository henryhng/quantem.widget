/// <reference types="@webgpu/types" />

// Browser driver for denova's denoising kernels.
//
// The WGSL is denova's, synced verbatim from denova/webgpu_sources by
// scripts/sync-denova-webgpu.mjs. This file only schedules it, mirroring the
// dispatch order in denova's reference demo: build the Poisson-Gaussian
// variance once, run Adam steps over a ping-pong pair, and pick lambda by
// driving chi2 to 1 (Morozov). The order is part of the algorithm -- the
// kernels carry Adam moments across iterations.
//
// Bind group indices come from the kernels: variance @group(0), Adam steps
// @group(1), chi2 @group(2). Each entry point owns a complete group, which is
// what makes layout:"auto" safe here.

import { DENOISE_WGSL } from "./.generated/denova/denoise.wgsl";
import { getGPUDevice } from "./fft";

export type DenovaMethod = "tv" | "tv2" | "tv12";

export interface DenovaOptions {
  /** Regularization strength. Omit to calibrate against chi2. */
  lambda?: number;
  iterations?: number;
  /** Noise model: var = alpha * max(f, 0) + sigma^2. */
  alpha?: number;
  sigma?: number;
  /** TV2 weight relative to TV, for "tv12". */
  tv2Ratio?: number;
}

// denova's demo constants
const WORKGROUP = 256;
const LEARNING_RATE = 0.01;
const TARGET_CHI2 = 1.0;
const CALIBRATION_ITERATIONS = 48;
const CALIBRATION_STEPS = 8;
const INITIAL_LAMBDA = 0.05;
const MAX_LAMBDA = 102.4;
const DEFAULTS = { iterations: 48, alpha: 0.02, sigma: 0.04, tv2Ratio: 0.5 };

function packParams(
  width: number,
  height: number,
  count: number,
  v: {
    lambda?: number;
    stepSize?: number;
    invSqrtBc2?: number;
    alpha?: number;
    sigma?: number;
    tv2Ratio?: number;
  },
): ArrayBuffer {
  const raw = new ArrayBuffer(48);
  const u32 = new Uint32Array(raw, 0, 4);
  const f32 = new Float32Array(raw);
  u32[0] = width;
  u32[1] = height;
  u32[2] = 1; // depth: one frame at a time
  u32[3] = count;
  f32[4] = v.lambda ?? 0.4;
  f32[5] = 0; // lambdaTemporal: 2D only
  f32[6] = v.stepSize ?? LEARNING_RATE;
  f32[7] = v.invSqrtBc2 ?? 1;
  f32[8] = v.alpha ?? DEFAULTS.alpha;
  f32[9] = v.sigma ?? DEFAULTS.sigma;
  f32[10] = 1.0e-8;
  f32[11] = v.tv2Ratio ?? DEFAULTS.tv2Ratio;
  return raw;
}

function dispatch(pass: GPUComputePassEncoder, n: number): void {
  const groups = Math.ceil(n / WORKGROUP);
  const x = Math.min(groups, 65535);
  pass.dispatchWorkgroups(x, Math.ceil(groups / x));
}

/** Finite and not flat; anything else renders as one solid colour. */
function usable(frame: Float32Array): boolean {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < frame.length; i++) {
    const v = frame[i];
    if (!Number.isFinite(v)) return false;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return max > min;
}

/**
 * Denoise one 2D frame with denova's kernels.
 *
 * Normalizes to [0, 1] and back the way denova does, so lambda and the noise
 * model stay scale invariant and callers can pass raw counts. Returns null when
 * WebGPU is unavailable or rejects the work, leaving fallback to the caller.
 */
export async function denovaDenoiseBrowser(
  data: Float32Array,
  width: number,
  height: number,
  method: DenovaMethod = "tv",
  options: DenovaOptions = {},
): Promise<Float32Array | null> {
  const device = await getGPUDevice();
  if (!device) return null;

  const count = width * height;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  const range = max - min;

  const normalized = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    normalized[i] = Number.isFinite(data[i]) ? (data[i] - min) / range : 0;
  }

  const storage = (bytes: number) =>
    device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  const partialCount = Math.ceil(count / WORKGROUP);

  const fBuffer = storage(count * 4);
  const varianceBuffer = storage(count * 4);
  const uA = storage(count * 4);
  const uB = storage(count * 4);
  const mBuffer = storage(count * 4);
  const vBuffer = storage(count * 4);
  const partials = storage(partialCount * 4);
  const uniform = (bytes: number) =>
    device.createBuffer({ size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const paramsBuffer = uniform(48);
  const reduceParams = uniform(48);
  const readBuffer = device.createBuffer({
    size: Math.max(count, partialCount) * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const alpha = options.alpha ?? DEFAULTS.alpha;
  const sigma = options.sigma ?? DEFAULTS.sigma;
  const tv2Ratio = options.tv2Ratio ?? DEFAULTS.tv2Ratio;

  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    device.queue.writeBuffer(fBuffer, 0, normalized);

    const module = device.createShaderModule({ code: DENOISE_WGSL });
    const entryPoint =
      method === "tv12" ? "adam_tv12_step" : method === "tv2" ? "adam_tv2_step" : "adam_tv_step";
    const variancePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "variance_kernel" },
    });
    const stepPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint },
    });
    const chi2Pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "chi2_reduce" },
    });

    // variance depends only on the measurement, so build it once
    device.queue.writeBuffer(paramsBuffer, 0, packParams(width, height, count, { alpha, sigma }));
    const varianceBind = device.createBindGroup({
      layout: variancePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: fBuffer } },
        { binding: 1, resource: { buffer: varianceBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(variancePipeline);
      pass.setBindGroup(0, varianceBind);
      dispatch(pass, count);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    let current = uA;

    const trial = async (lambda: number, iterations: number): Promise<number> => {
      // u starts at the measurement, Adam moments at zero
      const reset = device.createCommandEncoder();
      reset.copyBufferToBuffer(fBuffer, 0, uA, 0, count * 4);
      reset.copyBufferToBuffer(fBuffer, 0, uB, 0, count * 4);
      reset.clearBuffer(mBuffer);
      reset.clearBuffer(vBuffer);
      device.queue.submit([reset.finish()]);

      let src = uA;
      let dst = uB;
      for (let k = 1; k <= iterations; k++) {
        // Adam bias correction folded into the step size, as denova does
        const bc1 = 1 - Math.pow(0.9, k);
        const bc2 = 1 - Math.pow(0.999, k);
        device.queue.writeBuffer(
          paramsBuffer,
          0,
          packParams(width, height, count, {
            lambda,
            stepSize: LEARNING_RATE / bc1,
            invSqrtBc2: 1 / Math.sqrt(bc2),
            alpha,
            sigma,
            tv2Ratio,
          }),
        );
        const bind = device.createBindGroup({
          layout: stepPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: src } },
            { binding: 1, resource: { buffer: dst } },
            { binding: 2, resource: { buffer: fBuffer } },
            { binding: 3, resource: { buffer: varianceBuffer } },
            { binding: 4, resource: { buffer: mBuffer } },
            { binding: 5, resource: { buffer: vBuffer } },
            { binding: 6, resource: { buffer: paramsBuffer } },
          ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(stepPipeline);
        pass.setBindGroup(1, bind);
        dispatch(pass, count);
        pass.end();
        device.queue.submit([encoder.finish()]);
        [src, dst] = [dst, src];
      }
      current = src;

      // chi2 = mean squared residual over the noise variance; 1 means the fit
      // matches the noise level, which is the Morozov target
      device.queue.writeBuffer(reduceParams, 0, packParams(width, height, count, {}));
      const chiBind = device.createBindGroup({
        layout: chi2Pipeline.getBindGroupLayout(2),
        entries: [
          { binding: 0, resource: { buffer: current } },
          { binding: 1, resource: { buffer: fBuffer } },
          { binding: 2, resource: { buffer: varianceBuffer } },
          { binding: 3, resource: { buffer: partials } },
          { binding: 4, resource: { buffer: reduceParams } },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(chi2Pipeline);
      pass.setBindGroup(2, chiBind);
      dispatch(pass, count);
      pass.end();
      encoder.copyBufferToBuffer(partials, 0, readBuffer, 0, partialCount * 4);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ, 0, partialCount * 4);
      const values = new Float32Array(readBuffer.getMappedRange(0, partialCount * 4).slice(0));
      readBuffer.unmap();
      let total = 0;
      for (const value of values) total += value;
      return total / count;
    };

    let lambda = options.lambda;
    if (lambda === undefined) {
      // denova's calibration: double lambda until chi2 crosses the target, then
      // bisect. This is the zero-parameter behaviour its Python API gives you.
      let low = 0;
      let high = 0;
      let candidate = INITIAL_LAMBDA;
      let bracketed = false;
      while (candidate <= MAX_LAMBDA) {
        if ((await trial(candidate, CALIBRATION_ITERATIONS)) >= TARGET_CHI2) {
          high = candidate;
          bracketed = true;
          break;
        }
        low = candidate;
        candidate *= 2;
      }
      if (bracketed) {
        for (let step = 0; step < CALIBRATION_STEPS; step++) {
          const mid = 0.5 * (low + high);
          if ((await trial(mid, CALIBRATION_ITERATIONS)) < TARGET_CHI2) low = mid;
          else high = mid;
        }
        lambda = 0.5 * (low + high);
      } else {
        lambda = low || INITIAL_LAMBDA;
      }
    }

    await trial(lambda, Math.max(1, Math.round(options.iterations ?? DEFAULTS.iterations)));

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(current, 0, readBuffer, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ, 0, count * 4);
    const result = new Float32Array(readBuffer.getMappedRange(0, count * 4).slice(0));
    readBuffer.unmap();

    scopeOpen = false;
    const error = await device.popErrorScope();
    if (error) {
      console.warn("[denova] WebGPU rejected the denoise pass", error.message);
      return null;
    }

    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = result[i] * range + min;
    return usable(out) ? out : null;
  } catch (err) {
    if (scopeOpen) {
      scopeOpen = false;
      await device.popErrorScope();
    }
    console.warn("[denova] WebGPU denoise failed", err);
    return null;
  } finally {
    for (const buffer of [
      fBuffer, varianceBuffer, uA, uB, mBuffer, vBuffer,
      partials, paramsBuffer, reduceParams, readBuffer,
    ]) {
      buffer.destroy();
    }
  }
}
