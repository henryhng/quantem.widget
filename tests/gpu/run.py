#!/usr/bin/env python3
"""
Run GPU E2E tests for one or all widgets.

Usage:
    python tests/gpu/run.py                  # all widgets
    python tests/gpu/run.py show2d           # just Show2D
    python tests/gpu/run.py show2d --build   # build first
    python tests/gpu/run.py show2d --scale   # include scale test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

WIDGETS = {
    "show2d": "tests.gpu.test_show2d",
    "show3d": "tests.gpu.test_show3d",
}


def main():
    parser = argparse.ArgumentParser(description="Run GPU E2E tests")
    parser.add_argument("widget", nargs="?", default="all", help="Widget to test (show2d, show3d, ...) or 'all'")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()

    widgets = list(WIDGETS.keys()) if args.widget == "all" else [args.widget]
    total_pass, total_fail = 0, 0

    for w in widgets:
        if w not in WIDGETS:
            print(f"Unknown widget: {w}. Available: {', '.join(WIDGETS.keys())}")
            sys.exit(1)
        mod = __import__(WIDGETS[w], fromlist=["run_test"])
        results = mod.run_test(build=args.build, scale=args.scale)
        total_pass += results["pass"]
        total_fail += results["fail"]

    if len(widgets) > 1:
        total = total_pass + total_fail
        print(f"\nTotal: {total_pass}/{total} passed, {total_fail} failed")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
