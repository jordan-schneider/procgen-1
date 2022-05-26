import dataclasses
from typing import List, Tuple

import numpy as np
from experiment_server.gen_trajectory import (
    FeatureTrajectory,
    State,
    Trajectory,
    compute_diffs,
)
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import (
    binary,
    booleans,
    characters,
    composite,
    floats,
    integers,
    lists,
    tuples,
)

from .strategies import floats_1000


@composite
def states(draw, max_grid_size: int = 10) -> State:
    grid_size = draw(integers(min_value=1, max_value=max_grid_size))
    grid = draw(
        arrays(dtype=np.int32, shape=(grid_size, grid_size), elements=integers(1, 100))
    )
    pos_val = integers(min_value=0, max_value=grid_size - 1)
    position = tuples(pos_val, pos_val)
    agent_pos = draw(position)
    exit_pos = draw(position)
    return State(grid, agent_pos, exit_pos)


actions = arrays(
    dtype=np.int32, shape=array_shapes(max_dims=1), elements=integers(0, 4)
)


@composite
def trajs(draw, states, action_strategy):
    start_state = draw(states)
    actions = draw(action_strategy)
    env_name = draw(characters())
    return Trajectory(start_state, actions, env_name)


@composite
def feature_trajs(draw, states, action_strategy):
    start_state = draw(states)
    actions = draw(action_strategy)
    env_name = draw(characters())
    features = draw(
        arrays(dtype=np.float32, shape=(actions.shape[0], 4), elements=floats_1000)
    )
    return FeatureTrajectory(start_state, actions, env_name, features)


@given(state=states(max_grid_size=20))
def test_state_eq_reflexive(state: State):
    assert state == state


@given(
    state=states(max_grid_size=20),
    integer=integers(),
    real=floats(),
    string=characters(),
    boolean=booleans(),
    byte=binary(),
)
def test_state_neq_primitives(
    state: State,
    integer: int,
    real: float,
    string: str,
    boolean: bool,
    byte: bytes,
):
    assert state != integer
    assert state != real
    assert state != string
    assert state != boolean
    assert state != byte


@given(state=states(max_grid_size=20))
def test_state_neq_other_state(state: State):
    state1 = State(state.grid + 1, state.agent_pos, state.exit_pos)
    state2 = State(
        state.grid, (state.agent_pos[0] + 1, state.agent_pos[1]), state.exit_pos
    )
    state3 = State(
        state.grid, state.agent_pos, (state.exit_pos[0] + 1, state.exit_pos[1])
    )

    assert state != state1
    assert state != state2
    assert state != state3


@given(traj=trajs(states(max_grid_size=20), actions))
def test_traj_eq_reflexive(traj: Trajectory):
    assert traj == traj


@given(traj=trajs(states(max_grid_size=20), actions))
def test_traj_neq_other_traj(
    traj: Trajectory,
):
    grid, agent_pos, exit_pos = dataclasses.astuple(traj.start_state)
    actions = traj.actions
    env_name = traj.env_name
    traj1 = Trajectory(
        start_state=State(grid + 1, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name,
    )
    traj2 = Trajectory(
        start_state=State(grid, agent_pos, exit_pos),
        actions=actions + 1,
        env_name=env_name,
    )
    traj3 = Trajectory(
        start_state=State(grid, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name + "hello world",
    )

    assert traj != traj1
    assert traj != traj2
    assert traj != traj3


@given(
    traj=trajs(states(max_grid_size=20), actions),
    integer=integers(),
    real=floats(),
    string=characters(),
    boolean=booleans(),
    byte=binary(),
)
def test_traj_neq_primitives(
    traj: Trajectory,
    integer: int,
    real: float,
    string: str,
    boolean: bool,
    byte: bytes,
):
    assert traj != integer
    assert traj != real
    assert traj != string
    assert traj != boolean
    assert traj != byte


@given(traj=trajs(states(max_grid_size=20), actions))
def test_feature_traj_eq_reflexive(traj: FeatureTrajectory):
    assert traj == traj


@given(traj=feature_trajs(states(max_grid_size=20), actions))
def test_feature_traj_neq_other_traj(
    traj: FeatureTrajectory,
):
    grid, agent_pos, exit_pos = dataclasses.astuple(traj.start_state)
    actions = traj.actions
    env_name = traj.env_name
    features = traj.features
    traj1 = FeatureTrajectory(
        start_state=State(grid + 1, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name,
        features=features,
    )
    traj2 = FeatureTrajectory(
        start_state=State(grid, agent_pos, exit_pos),
        actions=actions + 1,
        env_name=env_name,
        features=features,
    )
    traj3 = FeatureTrajectory(
        start_state=State(grid, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name + "hello world",
        features=features,
    )
    traj4 = FeatureTrajectory(
        start_state=State(grid, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name,
        features=features / 2 + 1,
    )
    assert traj != traj1
    assert traj != traj2
    assert traj != traj3
    assert traj != traj4


@given(
    traj=feature_trajs(states(max_grid_size=20), actions),
    integer=integers(),
    real=floats(),
    string=characters(),
    boolean=booleans(),
    byte=binary(),
)
def test_feature_traj_neq_primitives(
    traj: FeatureTrajectory,
    integer: int,
    real: float,
    string: str,
    boolean: bool,
    byte: bytes,
):
    assert traj != integer
    assert traj != real
    assert traj != string
    assert traj != boolean
    assert traj != byte


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
