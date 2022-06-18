import numpy as np
from experiment_server.serialize import serialize
from experiment_server.type import Question, State, Trajectory


def test_serialize_question():
    state = State(grid=np.eye(20), grid_height=20, grid_width=20, agent_pos=(0,0), exit_pos=(1,1))
    traj = Trajectory(start_state=state, actions=np.arange(5), env_name="test", modality="traj")
    question = Question(id=1, first_traj=traj, second_traj=traj)
    serialize(question)
