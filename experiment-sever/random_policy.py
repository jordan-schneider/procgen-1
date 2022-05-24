import gym3
from numpy.random import Generator


class RandomPolicy:
    def __init__(self, env: gym3.Env, rng: Generator):
        self.actions = env.ac_space.eltype.n
        self.num = env.num
        self.rng = rng

    def __call__(self, ob):
        return self.rng.integers(low=0, high=self.actions, size=(self.num,))
