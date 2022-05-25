import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers

seed = integers(0, 2**32 - 1)

small_int_strategy = integers(1, 5)

finite_floats = floats(allow_infinity=False, allow_nan=False, width=32)
floats_1000 = floats(min_value=-1000, max_value=1000, allow_nan=False, width=32)


@composite
def halfplanes_strategy(draw, n_halfplanes=small_int_strategy):
    return draw(
        arrays(
            np.float32,
            (draw(n_halfplanes), 4),
            elements=floats_1000,
        )
    )


reward_strategy = (
    arrays(np.float32, (4,), elements=floats_1000)
    .filter(lambda r: np.any(r != 0.0))
    .map(lambda r: r / np.linalg.norm(r))
)


@composite
def rewards_strategy(draw, n_rewards=small_int_strategy):
    return draw(
        arrays(np.float32, (draw(n_rewards), 4), elements=floats_1000)
        .filter(lambda r: np.linalg.norm(r) > 0.0)
        .map(lambda r: r / np.linalg.norm(r))
    )
