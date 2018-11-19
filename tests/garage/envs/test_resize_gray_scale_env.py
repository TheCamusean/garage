import unittest

import gym

from garage.envs.wrappers import ResizeAndGrayscale


class TestResizeAndGrayScaleEnv(unittest.TestCase):
    def test_resize_and_gray_scale(self):
        # resize to (84, 84)
        # grayscale: channel is converted from 3 (84, 84, 3)
        # to 1 (84, 84)
        env_r = ResizeAndGrayscale(gym.make("Breakout-v0"), w=84, h=84)
        assert env_r.observation_space.shape == (84, 84)
