#!/usr/bin/env python3
"""Record a JupyterLab session scrubbing the denoise-across-time stack.

Starts a throwaway JupyterLab, loads the denova TV progression (raw frame, then
the solver's iterate at successive step counts) into ShowDiffraction, and steps
the frame slider across it while Playwright records the tab. The output is
trimmed to the scrub and cropped to the pattern plus its slider.

The solver runs before the browser starts: on a memory-tight box it and a
recording Chromium compete badly, and the kernel work is not what we are
filming.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEMO_SOURCE = """import numpy as np
from quantem.widget import ShowDiffraction

progression = np.load("_denoise_progression.npy")

ShowDiffraction(
    progression,
    dp_scale_mode="linear",
    title="denova TV: increasing strength",
    verbose=False,
    panel_width_px=560,
)
"""

DEMO_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": DEMO_SOURCE.splitlines(keepends=True),
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _build_progression(tutorials: Path, out: Path, max_frames: int) -> None:
    """Solve once up front so the recorded session only has to render.

    The sweep is over regularization strength, not iteration count. The solver
    converges within about eight iterations, so an iteration sweep is nearly
    static after the first few frames, while the auto-lambda (chi2 = 1, i.e.
    denoise exactly to the noise level) is deliberately conservative. Sweeping
    lambda around that point is what actually shows the solver working.
    """
    import numpy as np
    from PIL import Image
    from denova import denoise

    source = tutorials / "showdiffraction_data" / "fig6_ssb_r_star_phaseFFT_gamma0p7.jpg"
    frame = np.asarray(Image.open(source).convert("L"), dtype=np.float32)

    calibrated = denoise(frame, method="tv")
    frames = max(4, max_frames)
    # geometric from well under the calibrated value to heavily over-smoothed
    # Stop at 5x: past roughly 6x the flat-patch fraction climbs and the result
    # starts reading as posterized rather than denoised.
    lambdas = calibrated.lam * np.geomspace(0.4, 5.0, frames - 1)

    stack = np.empty((frames, *frame.shape), np.float32)
    stack[0] = frame
    for i, lam in enumerate(lambdas, start=1):
        stack[i] = np.asarray(denoise(frame, method="tv", lam=float(lam)).output, np.float32)
    np.save(out, stack)
    print(
        f"progression {stack.shape}, auto lambda {calibrated.lam:.3g}, "
        f"sweep {lambdas[0]:.3g} to {lambdas[-1]:.3g} (0.4x to 5x auto)"
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/tutorials/denoise_widget_demo.mp4")
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--step-ms", type=int, default=900)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    tutorials = REPO / "docs" / "tutorials"
    stack_path = tutorials / "_denoise_progression.npy"
    _build_progression(tutorials, stack_path, args.max_frames)

    demo_path = tutorials / "_denoise_demo.ipynb"
    demo_path.write_text(json.dumps(DEMO_NOTEBOOK, indent=1))

    port = _free_port()
    token = secrets.token_hex(8)
    video_dir = REPO / ".playwright-video"
    if video_dir.exists():
        shutil.rmtree(video_dir)

    lab = subprocess.Popen(
        [
            str(REPO / ".venv" / "bin" / "jupyter"), "lab",
            "--no-browser", f"--port={port}", f"--IdentityProvider.token={token}",
            f"--ServerApp.root_dir={tutorials}", "--ServerApp.open_browser=False",
        ],
        cwd=str(tutorials),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                import urllib.request

                urllib.request.urlopen(f"{base}/api?token={token}", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("JupyterLab did not come up")

        with sync_playwright() as p:
            # /dev/shm is 64 MB in most containers and Chromium OOM-crashes on it
            browser = p.chromium.launch(args=["--disable-dev-shm-usage", "--no-sandbox"])
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                record_video_dir=str(video_dir),
                record_video_size={"width": 1280, "height": 900},
            )
            record_started = time.monotonic()
            page = context.new_page()
            page.goto(
                f"{base}/lab/tree/_denoise_demo.ipynb?token={token}",
                wait_until="domcontentloaded",
                timeout=120_000,
            )
            page.wait_for_selector(".jp-Notebook", timeout=120_000)
            # Lab offers a news opt-in on first run; dismiss it so it does not
            # sit over the widget for the whole recording
            try:
                page.get_by_role("button", name="No", exact=True).click(timeout=15_000)
            except Exception:
                pass
            page.wait_for_timeout(3_000)

            page.keyboard.press("Shift+Enter")

            # Poll rather than one blocking wait: the widget mounts in stages and
            # the frame slider, not the canvas, is what the scrub needs.
            slider = page.get_by_role("slider", name="Frame")
            for _ in range(60):
                page.wait_for_timeout(5_000)
                if slider.count() and slider.first.is_visible():
                    break
            else:
                area = page.locator(".jp-OutputArea")
                detail = area.first.inner_text()[:500] if area.count() else "<no output>"
                raise RuntimeError(f"widget never appeared; cell output: {detail}")

            thumb = slider.first
            root = thumb.locator("xpath=ancestor::*[contains(@class,'MuiSlider-root')][1]")

            # Scroll to the top of the pattern, not the slider: the canvas is
            # taller than half the viewport, so centring the slider pushes the
            # top of the image off-screen and the crop loses it.
            page.locator("canvas").first.evaluate(
                "el => el.scrollIntoView({block: 'start'})"
            )
            page.wait_for_timeout(2_000)

            canvas = page.locator("canvas").first.bounding_box()
            rail = root.bounding_box()
            if canvas is None or rail is None:
                raise RuntimeError("could not measure the widget")
            viewport = page.viewport_size or {"height": 900}
            if canvas["y"] < 0 or rail["y"] + rail["height"] > viewport["height"]:
                raise RuntimeError(
                    f"widget does not fit: canvas top {canvas['y']:.0f}, "
                    f"slider bottom {rail['y'] + rail['height']:.0f}, "
                    f"viewport {viewport['height']}"
                )

            # Keep the pattern and the slider under it; the recording matches the
            # viewport 1:1, so page coordinates crop directly.
            crop_x = max(0, int(min(canvas["x"], rail["x"]) - 16))
            crop_y = max(0, int(canvas["y"] - 8))
            crop_w = int(max(canvas["width"], rail["width"]) + 32)
            crop_h = int(rail["y"] + rail["height"] + 40 - crop_y)

            # MUI hides a 10x10 <input type="range"> behind a 2px track, so a mouse
            # drag is unreliable to land. Focusing the input and stepping it moves
            # the thumb visibly and fires real change events.
            thumb.focus()
            highest = int(thumb.get_attribute("aria-valuemax") or 0)
            seen_max = 0

            page.wait_for_timeout(1_200)
            scrub_start = time.monotonic() - record_started

            def step(key: str, count: int) -> None:
                nonlocal seen_max
                for _ in range(count):
                    page.keyboard.press(key)
                    # each step pulls a frame from the kernel; the pause is also
                    # what makes the scrub readable rather than a flicker
                    page.wait_for_timeout(args.step_ms)
                    # instrumentation only: the widget re-renders on every frame
                    # and can detach the input mid-read, which must not kill the run
                    try:
                        value = thumb.get_attribute("aria-valuenow", timeout=2_000)
                        seen_max = max(seen_max, int(value or 0))
                    except Exception:
                        pass

            for sweep in range(args.sweeps):
                step("ArrowRight" if sweep % 2 == 0 else "ArrowLeft", highest)

            page.wait_for_timeout(1_000)
            scrub_end = time.monotonic() - record_started

            if seen_max < highest * 0.8:
                raise RuntimeError(
                    f"slider only reached {seen_max} of {highest}; it did not scrub"
                )
            print(f"scrubbed to {seen_max}/{highest}")

            context.close()
            browser.close()

        recorded = sorted(video_dir.glob("*.webm"))
        if not recorded:
            raise RuntimeError("no video captured")

        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio_ffmpeg

        # even dimensions only, or libx264 refuses the crop
        crop = f"crop={crop_w // 2 * 2}:{crop_h // 2 * 2}:{crop_x}:{crop_y}"
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-ss", f"{scrub_start:.2f}", "-t", f"{scrub_end - scrub_start:.2f}",
             "-i", str(recorded[0]), "-vf", crop,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            check=True,
        )
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
              f"{scrub_end - scrub_start:.0f}s of scrub, {crop}")
        return 0
    finally:
        lab.terminate()
        lab.wait(timeout=30)
        demo_path.unlink(missing_ok=True)
        stack_path.unlink(missing_ok=True)
        if video_dir.exists():
            shutil.rmtree(video_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
