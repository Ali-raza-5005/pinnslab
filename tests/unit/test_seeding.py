"""Determinism is the whole basis of every later claim (DESIGN.md §5)."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pinnslab.utils.seeding import (
    CUBLAS_WORKSPACE_CONFIG,
    capture_rng_state,
    derive_seed,
    make_generator,
    restore_rng_state,
    set_seed,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pinned outputs of ``derive_seed``. These are not arbitrary regression values:
#: every checkpoint ever written derives its sub-streams from this function, so
#: changing the separator, the digest size or the encoding would silently give
#: every resumed run a different RNG stream than the run it is continuing.
#: If a change here is deliberate, it is a breaking change — bump the version and
#: do not resume old checkpoints across it.
DERIVE_SEED_GOLDEN = {
    (0,): 1137565557,
    (0, "trainer"): 1530466301,
    (0, "member", 12): 985595060,
    (7, "trainer", "fefdd93509b81b18"): 1332618140,
}


def _draw_all() -> tuple[float, float, float]:
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_same_seed_gives_identical_streams():
    set_seed(123)
    first = [_draw_all() for _ in range(3)]
    set_seed(123)
    assert [_draw_all() for _ in range(3)] == first


def test_different_seeds_diverge():
    set_seed(1)
    first = _draw_all()
    set_seed(2)
    assert _draw_all() != first


def test_capture_restore_round_trips_every_stream():
    set_seed(99)
    _draw_all()  # advance past the seeded position
    state = capture_rng_state()
    expected = [_draw_all() for _ in range(4)]

    restore_rng_state(state)
    assert [_draw_all() for _ in range(4)] == expected


def test_capture_is_serialisable_without_numpy_arrays():
    """torch.load(weights_only=True) rejects ndarrays; the numpy state must be plain."""
    set_seed(3)
    state = capture_rng_state()
    _, keys, pos, has_gauss, cached = state["numpy"]
    assert isinstance(keys, list) and all(isinstance(k, int) for k in keys)
    assert isinstance(pos, int) and isinstance(has_gauss, int)
    assert isinstance(cached, float)


def test_restore_accepts_lists_where_tuples_were_captured():
    """Round-tripping through a serialiser can turn tuples into lists."""
    set_seed(11)
    state = capture_rng_state()
    expected = random.random()
    state["python"] = list(state["python"])  # what a JSON/torch round-trip may do
    restore_rng_state(state)
    assert random.random() == expected


# --- CUDA state guards in restore_rng_state ------------------------------------
# A checkpoint's CUDA RNG state must not be resumed on hardware it wasn't
# captured on (DESIGN.md §5: comparison groups must not span hardware). These
# run on CPU via monkeypatch, mirroring the cuBLAS guard tests above.


def test_restore_raises_if_checkpoint_has_cuda_state_but_process_has_none(monkeypatch):
    """Resuming a GPU-captured checkpoint on a CPU-only process must fail
    loudly, not silently drop the CUDA streams."""
    state = capture_rng_state()
    state["cuda"] = [torch.get_rng_state()]  # fake single-device CUDA state
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no CUDA"):
        restore_rng_state(state)


def test_restore_raises_on_cuda_device_count_mismatch(monkeypatch):
    """Resuming across a different device count would violate the
    hardware-uniformity rule (DESIGN.md §5)."""
    state = capture_rng_state()
    state["cuda"] = [torch.get_rng_state()]  # captured on 1 device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="hardware configurations"):
        restore_rng_state(state)


def test_derive_seed_is_stable_and_tag_sensitive():
    assert derive_seed(5, "sampler") == derive_seed(5, "sampler")
    assert derive_seed(5, "sampler") != derive_seed(5, "weighting")
    assert derive_seed(5, "sampler") != derive_seed(6, "sampler")
    assert 0 <= derive_seed(5, "sampler") < 2**31


def test_derive_seed_matches_its_pinned_values():
    """Pins the algorithm, not just its self-consistency.

    The previous version of this test compared two calls in one process, which
    passes just as happily for a ``hash()``-based implementation — it could not
    detect the very property the function exists to provide.
    """
    for args, expected in DERIVE_SEED_GOLDEN.items():
        assert derive_seed(*args) == expected, args


def test_derive_seed_ignores_python_hash_randomisation():
    """The real property: identical across processes with different hash seeds.

    ``hash()`` varies per process unless PYTHONHASHSEED is pinned, so a
    hash-based implementation would give a resumed run in a new Kaggle session
    different sub-streams than the session it is continuing.
    """
    code = (
        f"import sys; sys.path.insert(0, r'{REPO_ROOT}');"
        "from pinnslab.utils.seeding import derive_seed;"
        "print(derive_seed(0, 'trainer'), hash('trainer'))"
    )
    results = []
    for hash_seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        derived, hashed = out.stdout.split()
        results.append((int(derived), int(hashed)))

    assert results[0][0] == results[1][0] == derive_seed(0, "trainer")
    # Sanity check that the two subprocesses really did differ in hash seed —
    # otherwise the assertion above proves nothing.
    assert results[0][1] != results[1][1]


def test_derive_seed_distinguishes_no_tag_from_an_empty_tag():
    assert derive_seed(0) != derive_seed(0, "")


def test_make_generator_is_independent_of_global_rng():
    gen = make_generator(42)
    expected = torch.rand(4, generator=gen)

    set_seed(999)  # disturb the global streams
    torch.rand(100)

    gen = make_generator(42)
    assert torch.equal(torch.rand(4, generator=gen), expected)


def test_set_seed_rejects_negative():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_set_seed_enables_deterministic_algorithms():
    set_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False


# --- the cuBLAS workspace guard ------------------------------------------------
# Deterministic cuBLAS reductions need CUBLAS_WORKSPACE_CONFIG set *before* the
# CUDA context exists. Set it afterwards and cuBLAS ignores it silently: the run
# reports itself deterministic and is not. These tests are the only thing
# standing between us and that state, and they run on CPU via monkeypatch.


def test_set_seed_sets_the_cublas_workspace_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_set_seed_refuses_once_cuda_is_initialised(monkeypatch):
    """Too late to set it: fail loudly rather than claim a determinism we lack."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        set_seed(0)


def test_an_already_correct_cublas_config_is_not_a_conflict(monkeypatch):
    """The route the error message recommends must actually work.

    Setting the variable in the environment before importing torch is the
    documented fix; if the guard fired on that too, the advice would be a dead
    end and the only escape would be deterministic=False.
    """
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    set_seed(0)  # must not raise
    assert torch.are_deterministic_algorithms_enabled()


def test_a_wrong_preexisting_cublas_config_is_overwritten_before_cuda_starts(
    monkeypatch,
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    set_seed(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


# --- the deterministic / warn_only switches ------------------------------------


def test_deterministic_false_seeds_without_touching_global_flags(monkeypatch):
    """Opt-out must not silently leave determinism half-applied."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    set_seed(5, deterministic=False)

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_deterministic_false_still_seeds_every_stream():
    set_seed(5, deterministic=False)
    first = _draw_all()
    set_seed(5, deterministic=False)
    assert _draw_all() == first


def test_deterministic_false_skips_the_cuda_guard(monkeypatch):
    """Nothing to guard when we are not claiming determinism in the first place."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    set_seed(0, deterministic=False)  # must not raise


def test_warn_only_downgrades_the_error_to_a_warning():
    set_seed(0, warn_only=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()

    set_seed(0)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
