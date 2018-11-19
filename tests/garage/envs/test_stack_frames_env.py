import unittest

import gym
import numpy as np

from garage.envs.wrappers import ResizeAndGrayscale
from garage.envs.wrappers import StackFrames


class TestStackFramesEnv(unittest.TestCase):
    def test_stack_frames(self):
        # StackFrames only supports input with channel=1
        with self.assertRaises(ValueError):
            StackFrames(gym.make("Breakout-v0"), n_frames_stacked=4)

        env_stack = StackFrames(
            ResizeAndGrayscale(gym.make("Breakout-v0"), w=84, h=84),
            n_frames_stacked=4)
        env = ResizeAndGrayscale(gym.make("Breakout-v0"), w=84, h=84)

        env.seed(0)
        env_stack.seed(0)

        # after reset
        obs = env.reset()
        obs_stack = env_stack.reset()

        frame_stack = obs
        for i in range(3):
            frame_stack = np.dstack((frame_stack, obs))

        assert obs_stack.shape == (84, 84, 4)
        np.testing.assert_almost_equal(obs_stack, frame_stack)

        # stack next frame
        frame_stack = frame_stack[:, :, 1:]
        obs, _, _, _ = env.step(1)
        frame_stack = np.dstack((frame_stack, obs))

        obs_stack, _, _, _ = env_stack.step(1)
        np.testing.assert_almost_equal(obs_stack, frame_stack)

        # after a few steps
        for i in range(10):
            frame_stack = frame_stack[:, :, 1:]
            obs, _, _, _ = env.step(1)
            frame_stack = np.dstack((frame_stack, obs))

            obs_stack, _, _, _ = env_stack.step(1)
        np.testing.assert_almost_equal(obs_stack, frame_stack)
