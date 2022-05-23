import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def infogain(feature_diffs: np.ndarray, reward_posterior: np.ndarray) -> np.ndarray:
    """Computes the infogain of each question.

    Args:
        feature_diffs (np.ndarray): Difference between reward features of trajectories being compared in each question.
        reward_posterior (np.ndarray): Point based monte carlo estimate of the reward posterior.

    Returns:
        np.ndarray: Expected infogain of each question.
    """
    assert len(feature_diffs.shape) == 2
    assert len(reward_posterior.shape) == 2
    delta_return = feature_diffs @ reward_posterior.T
    p1 = 1 / (1 + np.exp(-delta_return))
    p2 = 1 / (1 + np.exp(delta_return))

    n_reward_samples = reward_posterior.shape[0]

    return (
        1.0
        / n_reward_samples
        * (
            np.sum(p1 * np.log2(n_reward_samples * p1 / p1.sum(axis=0)), axis=0)
            + np.sum(p2 * np.log2(n_reward_samples * p2 / p2.sum(axis=0)), axis=0)
        )
    )


def successive_elimination(
    question_samples: np.ndarray,
    reward_samples: np.ndarray,
    n_out_questions: int,
    inital_questions=200,
) -> np.ndarray:
    """Selects a batch of questions to ask using the successive elimination algorithm from Bikiy's
    Batch Active Learning from Preferences paper.

    Args:
        question_samples (np.ndarray): Unique feature differences of trajectories being compared in each question.
        reward_samples (np.ndarray): Point based monte carlo estimate of the reward posterior.
        n_out_questions (int): Desired batch size.
        inital_questions (int, optional): Number of questions to greedily select in the first step. Defaults to 200.

    Returns:
        np.ndarray: Index array of selected questions.
    """
    infogains = infogain(question_samples, reward_samples)
    indices = np.argpartition(infogains, inital_questions)[:inital_questions]

    greedy_questions = question_samples[indices]

    dists = pairwise_distances(greedy_questions, metric="euclidean")

    # Every question is distance 0 to itself, but we don't want to consider that a real distance.
    dists[np.where(np.eye(dists.shape[0]))] = np.inf

    while len(indices) > n_out_questions:
        mins = np.where(dists == np.min(dists))
        assert (
            len(mins) > 1 and len(mins[0]) == 2
        ), "Distance matrix is symmetric, there should be at least two minimums"
        min_index = mins[0]

        delete_dist_index = min_index[np.argmin(infogains[indices][min_index])]
        dists = np.delete(dists, delete_dist_index, axis=0)
        dists = np.delete(dists, delete_dist_index, axis=1)

        indices = np.delete(indices, delete_dist_index)

    return indices
