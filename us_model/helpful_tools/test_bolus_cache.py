"""
Tests for bolus_cache module.
Verifies save / load / validate / clear round-trip behaviour
using synthetic data (no real .mat files required).
"""

import os
import sys
import shutil
import tempfile

import numpy as np

# Ensure us_model is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolus_cache import (
    BOLUS_CACHE_VERSION,
    get_bolus_cache_dir,
    save_bolus_result,
    load_bolus_result,
    save_bolus_cache_metadata,
    load_bolus_cache_metadata,
    validate_bolus_cache,
    clear_bolus_cache,
    get_bolus_cache_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_result():
    """Return a dictionary that mimics LogNormalBolusSolver.fit() output."""
    n = 500
    time = np.linspace(0, 60, n)
    observed = np.random.rand(n).astype(np.float32)
    predicted = np.random.rand(n).astype(np.float32)
    losses = list(np.random.rand(100).astype(np.float64))
    return {
        "params": {"I0": 0.12, "A": 5.5, "t0": 8.0, "mu": 2.1, "sigma": 0.6},
        "derived": {"t_peak": 12.3, "MTT": 18.7, "AUC": 5.5},
        "predicted_curve": predicted,
        "losses": losses,
        "time": time,
        "observed": observed,
    }


_DATA_DIR = "/fake/data/dir"
_N_SVD = 5
_ROI = (100, 300, 150, 350)
_SMOOTH = 301


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_bolus_cache_dir_deterministic():
    d1 = get_bolus_cache_dir(_DATA_DIR, _N_SVD, _ROI, _SMOOTH)
    d2 = get_bolus_cache_dir(_DATA_DIR, _N_SVD, _ROI, _SMOOTH)
    assert d1 == d2, "Same inputs must produce the same cache directory"
    assert d1.startswith(f"bolus_cache_v{BOLUS_CACHE_VERSION}_")


def test_get_bolus_cache_dir_varies_with_params():
    d_base = get_bolus_cache_dir(_DATA_DIR, _N_SVD, _ROI, _SMOOTH)
    d_diff_svd = get_bolus_cache_dir(_DATA_DIR, 10, _ROI, _SMOOTH)
    d_diff_roi = get_bolus_cache_dir(_DATA_DIR, _N_SVD, (0, 50, 0, 50), _SMOOTH)
    d_diff_win = get_bolus_cache_dir(_DATA_DIR, _N_SVD, _ROI, 151)
    assert d_base != d_diff_svd
    assert d_base != d_diff_roi
    assert d_base != d_diff_win


def test_save_and_load_round_trip():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "bolus_test_cache")
        result = _make_fake_result()

        save_bolus_result(cache_dir, result)
        loaded = load_bolus_result(cache_dir)

        assert loaded is not None, "load_bolus_result returned None"
        assert loaded["params"] == result["params"]
        assert loaded["derived"] == result["derived"]
        np.testing.assert_array_almost_equal(
            loaded["predicted_curve"], result["predicted_curve"]
        )
        np.testing.assert_array_almost_equal(
            loaded["time"], result["time"]
        )
        np.testing.assert_array_almost_equal(
            loaded["observed"], result["observed"]
        )
        assert isinstance(loaded["losses"], list)
        np.testing.assert_array_almost_equal(
            loaded["losses"], result["losses"]
        )
    finally:
        shutil.rmtree(tmpdir)


def test_load_missing_returns_none():
    assert load_bolus_result("/nonexistent/path") is None


def test_metadata_round_trip():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "meta_test")
        fit_cfg = {"n_iterations": 150, "lr": 0.01, "patience": 20}
        save_bolus_cache_metadata(
            cache_dir, _DATA_DIR, _N_SVD, _ROI, _SMOOTH, fit_config=fit_cfg
        )
        meta = load_bolus_cache_metadata(cache_dir)

        assert meta is not None
        assert meta["version"] == BOLUS_CACHE_VERSION
        assert meta["data_dir"] == _DATA_DIR
        assert meta["n_svd"] == _N_SVD
        assert tuple(meta["roi_coords"]) == _ROI
        assert meta["smooth_window"] == _SMOOTH
        assert meta["fit_config"] == fit_cfg
    finally:
        shutil.rmtree(tmpdir)


def test_validate_valid_cache():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "valid_cache")
        save_bolus_cache_metadata(cache_dir, _DATA_DIR, _N_SVD, _ROI, _SMOOTH)
        save_bolus_result(cache_dir, _make_fake_result())

        is_valid, msg = validate_bolus_cache(
            cache_dir, _DATA_DIR, _N_SVD, _ROI, _SMOOTH
        )
        assert is_valid, f"Expected valid but got: {msg}"
    finally:
        shutil.rmtree(tmpdir)


def test_validate_missing_dir():
    ok, msg = validate_bolus_cache("/no/such/dir", _DATA_DIR, _N_SVD, _ROI, _SMOOTH)
    assert not ok
    assert "not found" in msg.lower()


def test_validate_param_mismatch():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "mismatch")
        save_bolus_cache_metadata(cache_dir, _DATA_DIR, _N_SVD, _ROI, _SMOOTH)
        save_bolus_result(cache_dir, _make_fake_result())

        # Wrong n_svd
        ok, _ = validate_bolus_cache(cache_dir, _DATA_DIR, 99, _ROI, _SMOOTH)
        assert not ok

        # Wrong ROI
        ok, _ = validate_bolus_cache(
            cache_dir, _DATA_DIR, _N_SVD, (0, 0, 0, 0), _SMOOTH
        )
        assert not ok

        # Wrong smooth_window
        ok, _ = validate_bolus_cache(cache_dir, _DATA_DIR, _N_SVD, _ROI, 51)
        assert not ok
    finally:
        shutil.rmtree(tmpdir)


def test_validate_missing_result_files():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "no_result")
        # Save metadata only — no result files
        save_bolus_cache_metadata(cache_dir, _DATA_DIR, _N_SVD, _ROI, _SMOOTH)

        ok, msg = validate_bolus_cache(
            cache_dir, _DATA_DIR, _N_SVD, _ROI, _SMOOTH
        )
        assert not ok
        assert "missing" in msg.lower() or "Missing" in msg
    finally:
        shutil.rmtree(tmpdir)


def test_clear_bolus_cache():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "to_clear")
        save_bolus_result(cache_dir, _make_fake_result())
        assert os.path.exists(cache_dir)

        cleared = clear_bolus_cache(cache_dir)
        assert cleared
        assert not os.path.exists(cache_dir)

        # Clearing again returns False
        assert not clear_bolus_cache(cache_dir)
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


def test_get_bolus_cache_size():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_dir = os.path.join(tmpdir, "sized")
        assert get_bolus_cache_size(cache_dir) == 0  # doesn't exist yet

        save_bolus_result(cache_dir, _make_fake_result())
        size_mb = get_bolus_cache_size(cache_dir)
        assert size_mb > 0
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
