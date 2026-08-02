// Sync canonical WGSL denoising kernels from denova into the widget frontend
// tree before bundling. denova owns the kernel source (its Torch and WebGPU
// paths implement the same math); the browser needs it bundled into the
// anywidget JS artifact.
//
// Unlike the quantem.gpu sync this does NOT import the package: denova.device
// raises at import time on a machine without CUDA/MPS, and a laptop docs build
// still has to be able to read the shader text. The sources are plain data
// files inside the package, so find_spec locates them without executing it.

import { spawnSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

export function syncDenovaWebgpuSources({ targetDir = "js/.generated/denova", required = false } = {}) {
  const outputDir = path.isAbsolute(targetDir) ? targetDir : path.join(repoRoot, targetDir);
  const python = process.env.PYTHON || "python3";
  const code = `
import importlib.util, json, pathlib

spec = importlib.util.find_spec("denova")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("denova not importable")
root = pathlib.Path(list(spec.submodule_search_locations)[0]) / "webgpu_sources"
print(json.dumps({p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.wgsl"))}))
`;
  const runExport = (env = process.env) => spawnSync(python, ["-c", code], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
    env,
  });

  let result = runExport();
  if (result.status !== 0) {
    const home = process.env.HOME || "";
    const srcDirs = [
      process.env.DENOVA_SRC,
      path.resolve(repoRoot, "../denoise/src"),
      path.resolve(repoRoot, "../../denoise/src"),
      home ? path.resolve(home, "repos/denoise/src") : "",
      home ? path.resolve(home, "denoise/src") : "",
    ].filter((srcDir) => srcDir && existsSync(srcDir));
    if (srcDirs.length) {
      const pythonPath = [...srcDirs, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter);
      result = runExport({ ...process.env, PYTHONPATH: pythonPath });
    }
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    const message =
      "Unable to sync WGSL sources from denova. Install denova in the active " +
      "Python environment, set PYTHON explicitly, or set DENOVA_SRC to the " +
      `denoise/src directory. ${detail}`;
    if (required) throw new Error(message);
    // denova is optional: the tv/gaussian/anscombe modes do not need it, so a
    // build without it still produces a working bundle minus the denova modes.
    console.log(`skipped denova WGSL sync (${detail.split("\n").pop() || "not found"})`);
    return { skipped: true };
  }

  const sources = JSON.parse(result.stdout);
  mkdirSync(outputDir, { recursive: true });
  let changed = 0;
  let unchanged = 0;
  for (const [name, text] of Object.entries(sources)) {
    const dest = path.join(outputDir, `${name}.ts`);
    const module = `// Generated from denova/webgpu_sources/${name}. Do not edit by hand.\n`
      + `export const ${name.replace(/\W/g, "_").toUpperCase()} = ${JSON.stringify(text)};\n`;
    const current = existsSync(dest) ? readFileSync(dest, "utf8") : null;
    if (current === module) {
      unchanged += 1;
      continue;
    }
    writeFileSync(dest, module, "utf8");
    changed += 1;
  }
  console.log(`synced denova WGSL -> ${targetDir} (${changed} updated, ${unchanged} unchanged)`);
  return { skipped: false, changed, unchanged };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  syncDenovaWebgpuSources();
}
