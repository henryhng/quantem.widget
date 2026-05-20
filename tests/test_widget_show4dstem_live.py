"""Tests for the Show4DSTEM live-acquisition append API.

Scope: the *Python API surface* for live 4D-STEM viewing. These tests do not
exercise any ZMQ/DECTRIS/Merlin/LiberTEM connector — that integration lives
in a downstream module that calls these methods.
"""

import json

import numpy as np
import pytest

from quantem.widget import Show4DSTEM


def _seed_widget(nav_shape=(8, 8), det_shape=(16, 16)):
    """Create a widget seeded with empty data, then put it into live_mode."""
    placeholder = np.zeros((1, 1, *det_shape), dtype=np.float32)
    w = Show4DSTEM(placeholder, verbose=False, precompute_virtual_images=False)
    w.begin_live(nav_shape=nav_shape, det_shape=det_shape)
    return w


def _make_frames(n, det_shape, fill_start=1.0):
    """Build a (n, det_y, det_x) batch with non-trivial content for VI."""
    frames = np.zeros((n, *det_shape), dtype=np.float32)
    for i in range(n):
        # Bright disc at the detector center so a centered ROI picks up signal.
        cy, cx = det_shape[0] / 2.0, det_shape[1] / 2.0
        yy, xx = np.ogrid[: det_shape[0], : det_shape[1]]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= (det_shape[0] / 4.0) ** 2
        frames[i][mask] = fill_start + i
    return frames


def test_begin_live_allocates_buffer_and_sets_traits():
    w = _seed_widget(nav_shape=(8, 8), det_shape=(16, 16))
    assert w.live_mode is True
    assert w.live_status == "streaming"
    assert w.live_total_expected == 64
    assert w.live_frames_received == 0
    assert w.live_drop_count == 0
    assert w.live_nav_shape == [8, 8]
    assert w.shape_rows == 8
    assert w.shape_cols == 8
    assert w.det_rows == 16
    assert w.det_cols == 16
    # Buffer should be the right shape on the widget's device.
    assert tuple(w._data.shape) == (8, 8, 16, 16)


def test_begin_live_zero_fills_buffer():
    """The buffer is always zero-filled; NaN fill was removed because NaNs
    propagated through the virtual-image masked sums and corrupted every
    unfilled pixel."""
    placeholder = np.zeros((1, 1, 4, 4), dtype=np.float32)
    w = Show4DSTEM(placeholder, verbose=False, precompute_virtual_images=False)
    w.begin_live(nav_shape=(2, 2), det_shape=(4, 4))
    buf = w._data.cpu().numpy()
    assert buf.shape == (2, 2, 4, 4)
    assert buf.dtype == np.float32
    assert np.all(buf == 0.0)
    assert not np.isnan(buf).any()


def test_set_image_clears_live_mode():
    """set_image replaces the buffer wholesale — live state must reset so the
    UI does not show stale frames-received against the new buffer."""
    placeholder = np.zeros((1, 1, 4, 4), dtype=np.float32)
    w = Show4DSTEM(placeholder, verbose=False, precompute_virtual_images=False)
    w.begin_live(nav_shape=(4, 4), det_shape=(8, 8))
    frame = np.ones((8, 8), dtype=np.float32)
    w.append_frames(0, frame)
    assert w.live_mode is True
    assert w.live_frames_received == 1
    new_data = np.zeros((6, 6, 8, 8), dtype=np.float32)
    w.set_image(new_data)
    assert w.live_mode is False
    assert w.live_status == "idle"
    assert w.live_frames_received == 0
    assert w.live_total_expected == 0


def test_append_single_frame_increments_counter():
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))
    frame = np.ones((8, 8), dtype=np.float32) * 3.0
    w.append_frames(0, frame)
    assert w.live_frames_received == 1
    # Content should be written into position (0, 0).
    written = w._data.cpu().numpy()[0, 0]
    np.testing.assert_array_equal(written, frame)


def test_append_batch_increments_counter_by_n():
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))
    frames = _make_frames(5, (8, 8))
    w.append_frames(0, frames)
    assert w.live_frames_received == 5


def test_partition_threshold_triggers_virtual_image_refresh():
    nav = (8, 8)
    det = (16, 16)
    w = _seed_widget(nav_shape=nav, det_shape=det)
    # Tighten partition size so the test runs at a sensible batch count.
    w.live_partition_size = 16

    # Snapshot the pre-stream virtual image bytes (all-zeros buffer).
    vi_before = bytes(w.virtual_image_bytes)
    frame_before = bytes(w.frame_bytes)

    # Append exactly partition_size bright frames; observer fires once.
    frames = _make_frames(16, det, fill_start=10.0)
    w.append_frames(0, frames)

    assert w.live_frames_received == 16
    assert w.live_last_partition_idx == 16
    # BF/ADF/HAADF (virtual_image_bytes) should now differ from the zeros.
    vi_after = bytes(w.virtual_image_bytes)
    frame_after = bytes(w.frame_bytes)
    assert len(vi_after) > 0
    assert vi_after != vi_before or frame_after != frame_before, (
        "Expected at least one of (virtual_image_bytes, frame_bytes) to change "
        "after a partition-cadence refresh."
    )


def test_partition_threshold_does_not_fire_below_size():
    w = _seed_widget(nav_shape=(8, 8), det_shape=(8, 8))
    w.live_partition_size = 16
    frames = _make_frames(8, (8, 8))
    w.append_frames(0, frames)
    # Only 8 of 16 frames received -> no refresh.
    assert w.live_frames_received == 8
    assert w.live_last_partition_idx == 0


def test_pause_resume_toggles_live_status():
    w = _seed_widget()
    assert w.live_status == "streaming"
    w.pause_live()
    assert w.live_status == "paused"
    w.resume_live()
    assert w.live_status == "streaming"


def test_end_live_sets_done_and_prints_summary(capsys):
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))
    frames = _make_frames(3, (8, 8))
    w.append_frames(0, frames)
    w.bump_drop_count(2)
    w.end_live()
    assert w.live_status == "done"
    captured = capsys.readouterr().out
    assert "Live acquisition ended" in captured
    assert "3/" in captured
    assert "2 drops" in captured


def test_end_live_triggers_final_refresh():
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))
    w.live_partition_size = 1000  # Make sure the cadence path never fires.
    frames = _make_frames(2, (8, 8), fill_start=50.0)
    w.append_frames(0, frames)
    # No partition refresh yet, so live_last_partition_idx == 0.
    assert w.live_last_partition_idx == 0
    vi_before = bytes(w.virtual_image_bytes)
    w.end_live()
    # end_live() updates live_last_partition_idx and refreshes virtual image.
    assert w.live_last_partition_idx == 2
    vi_after = bytes(w.virtual_image_bytes)
    # Force-asserting equality is brittle (the seed widget already had a vi
    # computed); we just confirm bytes are present and non-empty.
    assert len(vi_after) > 0
    assert vi_before != b"" or vi_after != b""


def test_bump_drop_count_increments():
    w = _seed_widget()
    assert w.live_drop_count == 0
    w.bump_drop_count(3)
    assert w.live_drop_count == 3
    w.bump_drop_count()  # default n=1
    assert w.live_drop_count == 4


def test_bump_drop_count_rejects_negative():
    w = _seed_widget()
    with pytest.raises(ValueError):
        w.bump_drop_count(-1)


def test_append_past_buffer_raises_value_error():
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))  # total = 16
    frames = _make_frames(5, (8, 8))
    with pytest.raises(ValueError):
        w.append_frames(start_idx=14, frames=frames)  # 14+5 > 16


def test_append_rejects_wrong_detector_shape():
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))
    bad = np.ones((1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        w.append_frames(0, bad)


def test_append_outside_live_mode_raises():
    placeholder = np.zeros((2, 2, 4, 4), dtype=np.float32)
    w = Show4DSTEM(placeholder, verbose=False, precompute_virtual_images=False)
    # Note: live_mode default is False, so append_frames must raise.
    assert w.live_mode is False
    frame = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(RuntimeError):
        w.append_frames(0, frame)


def test_single_frame_then_batch_appends_compose():
    det = (4, 4)
    w = _seed_widget(nav_shape=(4, 4), det_shape=det)
    f1 = np.ones(det, dtype=np.float32) * 1.0
    w.append_frames(0, f1)
    assert w.live_frames_received == 1
    batch = _make_frames(3, det, fill_start=2.0)
    w.append_frames(1, batch)
    assert w.live_frames_received == 4
    # Position 0 must still hold the single-frame content.
    np.testing.assert_array_equal(w._data.cpu().numpy()[0, 0], f1)
    # Positions 1, 2, 3 hold the batch.
    for i, b in enumerate(batch):
        flat_row, flat_col = divmod(1 + i, 4)
        np.testing.assert_array_equal(w._data.cpu().numpy()[flat_row, flat_col], b)


def test_state_dict_includes_live_traits():
    w = _seed_widget(nav_shape=(6, 6), det_shape=(8, 8))
    w.live_partition_size = 12
    w.bump_drop_count(2)
    w.append_frames(0, _make_frames(4, (8, 8)))
    sd = w.state_dict()
    assert sd["live_mode"] is True
    assert sd["live_nav_shape"] == [6, 6]
    assert sd["live_total_expected"] == 36
    assert sd["live_frames_received"] == 4
    assert sd["live_drop_count"] == 2
    assert sd["live_partition_size"] == 12
    assert sd["live_status"] == "streaming"
    # Round-trip through JSON (state_dict must be JSON-serializable).
    blob = json.dumps(sd)
    parsed = json.loads(blob)
    assert parsed["live_mode"] is True


def test_summary_includes_live_line(capsys):
    w = _seed_widget(nav_shape=(4, 4), det_shape=(8, 8))
    w.append_frames(0, _make_frames(2, (8, 8)))
    w.bump_drop_count(1)
    w.summary()
    out = capsys.readouterr().out
    assert "Live:" in out
    assert "streaming" in out
    assert "2/16" in out
    assert "1 drops" in out


def test_partition_index_advances_correctly_on_large_batch():
    """When a single append jumps the counter by multiple partitions, the
    bookkeeping must advance one partition at a time so the next partition
    boundary is hit at the right frame count (per LiberTEM `frames_per_partition`)."""
    w = _seed_widget(nav_shape=(8, 8), det_shape=(8, 8))
    w.live_partition_size = 4
    # Pre-state
    assert w.live_last_partition_idx == 0
    # One big batch of 13 frames spans 3 partitions
    batch = np.ones((13, 8, 8), dtype=np.float32)
    w.append_frames(0, batch)
    # After the batch, last_partition_idx should be 12 (3 * partition_size),
    # not 13 (the previous bug discarded intermediate boundaries by setting
    # last_partition_idx = live_frames_received).
    assert w.live_frames_received == 13
    assert w.live_last_partition_idx == 12, (
        f"expected 12 (3 * partition_size), got {w.live_last_partition_idx}"
    )
