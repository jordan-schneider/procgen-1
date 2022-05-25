import numpy as np
from experiment_server.biyik import infogain, successive_elimination
from hypothesis import given

from strategies import halfplanes_strategy, rewards_strategy


@given(rewards=rewards_strategy(), halfplanes=halfplanes_strategy())
def test_infogain_even(rewards: np.ndarray, halfplanes: np.ndarray):
    forward_infogain = infogain(halfplanes, rewards)
    backward_infogain = infogain(-halfplanes, rewards)
    assert np.allclose(forward_infogain, backward_infogain)

    backward_infogain = infogain(halfplanes, -rewards)
    assert np.allclose(forward_infogain, backward_infogain)


@given(rewards=rewards_strategy(), halfplanes=halfplanes_strategy())
def test_infogain_nonnegative(rewards: np.ndarray, halfplanes: np.ndarray):
    infogains = infogain(halfplanes, rewards)
    assert np.all(infogains >= 0)


@given(rewards=rewards_strategy(), questions=halfplanes_strategy())
def test_successive_elimination_noop(rewards: np.ndarray, questions: np.ndarray):
    n_questions = questions.shape[0]
    indices = successive_elimination(
        reward_samples=rewards,
        question_samples=questions,
        n_out_questions=n_questions,
        initial_questions=n_questions,
    )
    final_questions = questions[np.sort(indices)]
    assert np.array_equal(questions, final_questions)
