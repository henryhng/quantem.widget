import time as _time
import tempfile

import numpy as np
import pytest

from quantem.widget import Live
from quantem.widget.io import IOResult


# ── Basic construction ──────────────────────────────────────────────────


def test_live_empty():
    w = Live()
    assert w.n_frames == 0
    assert w.mode == "2d"
    assert w.title == "Live"


def test_live_title():
    w = Live(title="SNSF Session")
    assert w.title == "SNSF Session"


def test_live_mode():
    w = Live(mode="3d")
    assert w.mode == "3d"


def test_live_repr_empty():
    w = Live()
    assert "0 frames" in repr(w)
    assert "2D" in repr(w)


# ── Push ────────────────────────────────────────────────────────────────


def test_live_push_single():
    w = Live()
    img = np.random.rand(32, 32).astype(np.float32)
    result = w.push(img)
    assert result is w  # chaining
    assert w.n_frames == 1
    assert w._push_height == 32
    assert w._push_width == 32


def test_live_push_multiple():
    w = Live()
    for i in range(5):
        w.push(np.random.rand(16, 16).astype(np.float32))
    assert w.n_frames == 5
    assert len(w.labels) == 5


def test_live_push_label():
    w = Live()
    w.push(np.random.rand(8, 8).astype(np.float32), label="Gold NP")
    assert w.labels == ["Gold NP"]


def test_live_push_default_labels():
    w = Live()
    w.push(np.random.rand(8, 8).astype(np.float32))
    w.push(np.random.rand(8, 8).astype(np.float32))
    assert w.labels == ["Frame 0", "Frame 1"]


def test_live_push_io_result():
    data = np.random.rand(32, 32).astype(np.float32)
    result = IOResult(data=data, title="capture_001")
    w = Live()
    w.push(result)
    assert w.n_frames == 1
    assert w.labels == ["capture_001"]


def test_live_push_3d_rejects():
    w = Live()
    with pytest.raises(ValueError, match="2D"):
        w.push(np.random.rand(5, 16, 16).astype(np.float32))


def test_live_push_3d_singleton_ok():
    """3D with shape (1, H, W) is squeezed to 2D."""
    w = Live()
    w.push(np.random.rand(1, 16, 16).astype(np.float32))
    assert w.n_frames == 1


def test_live_push_mixed_sizes():
    w = Live()
    w.push(np.random.rand(16, 16).astype(np.float32))
    w.push(np.random.rand(64, 64).astype(np.float32))
    w.push(np.random.rand(32, 48).astype(np.float32))
    assert w.n_frames == 3
    # Each push sends its own dimensions
    assert w._push_height == 32
    assert w._push_width == 48


def test_live_push_counter_increments():
    w = Live()
    assert w._push_counter == 0
    w.push(np.random.rand(8, 8).astype(np.float32))
    assert w._push_counter == 1
    w.push(np.random.rand(8, 8).astype(np.float32))
    assert w._push_counter == 2


def test_live_push_torch():
    import torch
    w = Live()
    t = torch.rand(32, 32)
    w.push(t)
    assert w.n_frames == 1


# ── Constructor with data ──────────────────────────────────────────────


def test_live_init_with_2d_array():
    img = np.random.rand(32, 32).astype(np.float32)
    w = Live(img)
    assert w.n_frames == 1


def test_live_init_with_3d_array():
    stack = np.random.rand(5, 16, 16).astype(np.float32)
    w = Live(stack)
    assert w.n_frames == 5


def test_live_init_with_list():
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Live(imgs)
    assert w.n_frames == 3


def test_live_init_with_io_result():
    data = np.random.rand(3, 16, 16).astype(np.float32)
    result = IOResult(data=data, title="stack", labels=["a", "b", "c"])
    w = Live(result)
    assert w.n_frames == 3
    assert w.title == "stack"
    assert w.labels == ["a", "b", "c"]


# ── Watch ───────────────────────────────────────────────────────────────


def test_live_watch_existing(tmp_path):
    for i in range(3):
        np.save(tmp_path / f"img_{i}.npy", np.random.rand(16, 16).astype(np.float32))
    w = Live.watch(tmp_path, pattern="*.npy", interval=0.2)
    try:
        assert w.n_frames == 3
        assert w.watching
    finally:
        w.stop()


def test_live_watch_new_file(tmp_path):
    np.save(tmp_path / "img_0.npy", np.random.rand(16, 16).astype(np.float32))
    w = Live.watch(tmp_path, pattern="*.npy", interval=0.2)
    try:
        assert w.n_frames == 1
        np.save(tmp_path / "img_1.npy", np.random.rand(16, 16).astype(np.float32))
        _time.sleep(1.5)
        assert w.n_frames == 2
    finally:
        w.stop()


def test_live_watch_stop(tmp_path):
    np.save(tmp_path / "img.npy", np.random.rand(8, 8).astype(np.float32))
    w = Live.watch(tmp_path, pattern="*.npy", interval=0.2)
    assert w.watching
    w.stop()
    assert not w.watching


def test_live_watch_nonexistent():
    with pytest.raises(ValueError, match="does not exist"):
        Live.watch("/nonexistent/folder", pattern="*.npy")


def test_live_watch_title_from_folder(tmp_path):
    np.save(tmp_path / "img.npy", np.random.rand(8, 8).astype(np.float32))
    w = Live.watch(tmp_path, pattern="*.npy", interval=0.2)
    try:
        assert w.title == tmp_path.name
    finally:
        w.stop()


def test_live_watch_custom_title(tmp_path):
    np.save(tmp_path / "img.npy", np.random.rand(8, 8).astype(np.float32))
    w = Live.watch(tmp_path, pattern="*.npy", interval=0.2, title="Custom")
    try:
        assert w.title == "Custom"
    finally:
        w.stop()


# ── Display settings ────────────────────────────────────────────────────


def test_live_cmap():
    w = Live(cmap="viridis")
    assert w.cmap == "viridis"


def test_live_log_scale():
    w = Live(log_scale=True)
    assert w.log_scale is True


def test_live_auto_contrast():
    w = Live(auto_contrast=True)
    assert w.auto_contrast is True


def test_live_buffer_size():
    w = Live(buffer_size=100)
    assert w.buffer_size == 100


def test_live_fps():
    w = Live(fps=10.0)
    assert w.fps == pytest.approx(10.0)


# ── Summary / repr ──────────────────────────────────────────────────────


def test_live_summary(capsys):
    w = Live(title="Test")
    w.push(np.random.rand(8, 8).astype(np.float32))
    w.summary()
    out = capsys.readouterr().out
    assert "Test" in out
    assert "1" in out


def test_live_repr():
    w = Live()
    w.push(np.random.rand(8, 8).astype(np.float32))
    r = repr(w)
    assert "1 frames" in r
    assert "2D" in r


def test_live_repr_3d():
    w = Live(mode="3d")
    assert "3D" in repr(w)


def test_live_repr_watching(tmp_path):
    np.save(tmp_path / "img.npy", np.random.rand(8, 8).astype(np.float32))
    w = Live.watch(tmp_path, pattern="*.npy", interval=0.2)
    try:
        assert "watching" in repr(w)
    finally:
        w.stop()


# ── Widget version ──────────────────────────────────────────────────────


def test_live_widget_version():
    w = Live()
    assert w.widget_version != "unknown"


# ── from_folder ─────────────────────────────────────────────────────────


def test_live_from_folder(tmp_path):
    for i in range(5):
        np.save(tmp_path / f"img_{i}.npy", np.random.rand(16, 16).astype(np.float32))
    w = Live.from_folder(tmp_path, pattern="*.npy")
    assert w.n_frames == 5
    assert w.title == tmp_path.name


def test_live_from_folder_max_files(tmp_path):
    for i in range(10):
        np.save(tmp_path / f"img_{i}.npy", np.random.rand(8, 8).astype(np.float32))
    w = Live.from_folder(tmp_path, pattern="*.npy", max_files=3)
    assert w.n_frames == 3


def test_live_from_folder_mixed_sizes(tmp_path):
    np.save(tmp_path / "small.npy", np.random.rand(16, 16).astype(np.float32))
    np.save(tmp_path / "large.npy", np.random.rand(128, 128).astype(np.float32))
    w = Live.from_folder(tmp_path, pattern="*.npy")
    assert w.n_frames == 2


def test_live_from_folder_custom_title(tmp_path):
    np.save(tmp_path / "img.npy", np.random.rand(8, 8).astype(np.float32))
    w = Live.from_folder(tmp_path, pattern="*.npy", title="Custom")
    assert w.title == "Custom"


def test_live_from_folder_nonexistent():
    with pytest.raises(ValueError, match="does not exist"):
        Live.from_folder("/nonexistent/folder")


def test_live_from_folder_empty(tmp_path):
    w = Live.from_folder(tmp_path, pattern="*.npy")
    assert w.n_frames == 0


def test_live_from_folder_pattern_filter(tmp_path):
    np.save(tmp_path / "keep.npy", np.random.rand(8, 8).astype(np.float32))
    np.savez(tmp_path / "skip.npz", data=np.random.rand(8, 8).astype(np.float32))
    w = Live.from_folder(tmp_path, pattern="*.npy")
    assert w.n_frames == 1


# ── Performance ─────────────────────────────────────────────────────────


def test_live_push_50_frames_fast():
    """50 × 256×256 pushes should complete in under 2 seconds."""
    w = Live(buffer_size=50)
    import time
    t0 = time.perf_counter()
    for i in range(50):
        w.push(np.random.rand(256, 256).astype(np.float32))
    elapsed = time.perf_counter() - t0
    assert w.n_frames == 50
    assert elapsed < 2.0, f"50 pushes took {elapsed:.2f}s, expected < 2s"
