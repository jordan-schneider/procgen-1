import dataclasses

from experiment_server.type import FeatureTrajectory, State, Trajectory
from hypothesis import given
from hypothesis.strategies import binary, booleans, characters, floats, integers

from .strategies import actions, feature_trajs, states, trajs


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
    state1 = State(
        state.grid + 1,
        state.grid_width,
        state.grid_height,
        state.agent_pos,
        state.exit_pos,
    )
    state2 = State(
        state.grid,
        state.grid_width,
        state.grid_height,
        (state.agent_pos[0] + 1, state.agent_pos[1]),
        state.exit_pos,
    )
    state3 = State(
        state.grid,
        state.grid_width,
        state.grid_height,
        state.agent_pos,
        (state.exit_pos[0] + 1, state.exit_pos[1]),
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
    grid, grid_width, grid_height, agent_pos, exit_pos = dataclasses.astuple(
        traj.start_state
    )
    actions = traj.actions
    assert actions is not None
    env_name = traj.env_name
    modality = traj.modality
    traj1 = Trajectory(
        start_state=State(grid + 1, grid_width, grid_height, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name,
        modality=modality,
    )
    traj2 = Trajectory(
        start_state=traj.start_state,
        actions=actions + 1,
        env_name=env_name,
        modality=modality,
    )
    traj3 = Trajectory(
        start_state=traj.start_state,
        actions=actions,
        env_name=env_name + "hello world",
        modality=modality,
    )
    traj4 = Trajectory(
        start_state=traj.start_state,
        actions=actions,
        env_name=env_name,
        modality=modality + "hello world",
    )
    assert traj != traj1
    assert traj != traj2
    assert traj != traj3
    assert traj != traj4


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
    grid, grid_width, grid_height, agent_pos, exit_pos = dataclasses.astuple(
        traj.start_state
    )
    actions = traj.actions
    assert actions is not None
    env_name = traj.env_name
    modality = traj.modality
    features = traj.features
    traj1 = FeatureTrajectory(
        start_state=State(grid + 1, grid_width, grid_height, agent_pos, exit_pos),
        actions=actions,
        env_name=env_name,
        features=features,
        modality=modality,
    )
    traj2 = FeatureTrajectory(
        start_state=traj.start_state,
        actions=actions + 1,
        env_name=env_name,
        features=features,
        modality=modality,
    )
    traj3 = FeatureTrajectory(
        start_state=traj.start_state,
        actions=actions,
        env_name=env_name + "hello world",
        features=features,
        modality=modality,
    )
    traj4 = FeatureTrajectory(
        start_state=traj.start_state,
        actions=actions,
        env_name=env_name,
        features=features / 2 + 1,
        modality=modality,
    )
    traj5 = FeatureTrajectory(
        start_state=traj.start_state,
        actions=actions,
        env_name=env_name,
        features=features,
        modality=modality + "hello world",
    )
    assert traj != traj1
    assert traj != traj2
    assert traj != traj3
    assert traj != traj4
    assert traj != traj5


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
