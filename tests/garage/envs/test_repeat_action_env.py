import unittest

import gym
import numpy as np

from garage.envs.wrappers import RepeatAction


class TestRepeatActionEnv(unittest.TestCase):
    def test_repeat_action(self):
        env_repeat = RepeatAction(gym.make("Breakout-v0"), frame_to_repeat=4)
        env = gym.make("Breakout-v0")

        env.seed(0)
        env_repeat.seed(0)

        env.reset()
        env_repeat.reset()

        obs_repeat, _, _, _ = env_repeat.step(1)

        for i in range(4):
            obs, _, _, _ = env.step(1)

        np.testing.assert_almost_equal(obs, obs_repeat)
