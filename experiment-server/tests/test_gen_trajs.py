from typing import List, Tuple

import numpy as np
from experiment_server.gen_trajectory import compute_diffs
from experiment_server.types import FeatureTrajectory
from hypothesis import given
from hypothesis.strategies import lists, tuples

from .strategies import actions, feature_trajs, states


@given(
    questions=lists(
        tuples(
            feature_trajs(states(max_grid_size=20), actions),
            feature_trajs(states(max_grid_size=20), actions),
        ),
        min_size=1,
    )
)
def test_compute_diffs_odd(
    questions: List[Tuple[FeatureTrajectory, FeatureTrajectory]]
):
    diffs = compute_diffs(questions)
    reverse_diffs = compute_diffs(
        [(question[1], question[0]) for question in questions]
    )
    assert np.allclose(diffs, -reverse_diffs)


@given(trajs=lists(feature_trajs(states(max_grid_size=20), actions), min_size=1))
def test_compute_diffs_zero(trajs: List[FeatureTrajectory]):
    questions = [
        (
            traj,
            FeatureTrajectory(
                traj.start_state,
                traj.actions,
                traj.env_name,
                traj.modality,
                np.zeros_like(traj.features),
            ),
        )
        for traj in trajs
    ]
    diffs = compute_diffs(questions)
    assert np.allclose(
        diffs, np.stack([np.sum(traj.features, axis=0) for traj in trajs])
    )


@given(trajs=lists(feature_trajs(states(max_grid_size=20), actions), min_size=1))
def test_compute_diffs_same(trajs: List[FeatureTrajectory]):
    questions = [(traj, traj) for traj in trajs]
    diffs = compute_diffs(questions)
    assert np.allclose(diffs, 0)
