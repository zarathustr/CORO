from __future__ import annotations

import numpy as np

from deep_coro_znn_parallel.calibration import (
    generate_axbycz_instance,
    generate_hand_eye_instance,
    generate_rwhe_instance,
    solve_axbycz,
    solve_hand_eye,
    solve_rwhe,
)


def test_hand_eye_smoke() -> None:
    rng = np.random.default_rng(0)
    instance = generate_hand_eye_instance(rng, num_pairs=3)
    result = solve_hand_eye(instance, dynamics="znn", projector="coro", steps=5)
    assert result.estimate.shape == (4, 4)



def test_rwhe_smoke() -> None:
    rng = np.random.default_rng(1)
    instance = generate_rwhe_instance(rng, num_pairs=3)
    result = solve_rwhe(instance, dynamics="znn", projector="coro", steps=5)
    assert result.x_estimate.shape == (4, 4)
    assert result.y_estimate.shape == (4, 4)



def test_axbycz_smoke() -> None:
    rng = np.random.default_rng(2)
    instance = generate_axbycz_instance(rng, num_pairs=3)
    result = solve_axbycz(instance, dynamics="znn", projector="coro", steps=5)
    assert result.x_estimate.shape == (4, 4)
    assert result.y_estimate.shape == (4, 4)
    assert result.z_estimate.shape == (4, 4)
