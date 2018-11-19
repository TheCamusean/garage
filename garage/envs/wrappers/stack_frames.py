"""This module implements the stack frames wrapper for gym env."""

from collections import deque

import gym
from gym.spaces import Box
import numpy as np


class StackFrames(gym.Wrapper):
    """
    Stack frames wrapper.

    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, n_frames_stacked):
        """Construct the class."""
        super(StackFrames, self).__init__(env)
        if len(env.observation_space.shape) != 2:
            raise ValueError(
                'Stack frames only works with 2D single channel images')
        self._n_stack_past = n_frames_stacked
        self._frames = None

        new_obs_space_shape = env.observation_space.shape + (
            n_frames_stacked, )
        self.observation_space = Box(
            0.0, 1.0, shape=new_obs_space_shape, dtype=np.float32)

    def _stack_frames(self):
        return np.stack(self._frames, axis=2)

    def reset(self):
        """Gym env reset function."""
        observation = self.env.reset()
        self._frames = deque([observation] * self._n_stack_past)
        return self._stack_frames()

    def step(self, action):
        """Gym env step function."""
        new_observation, reward, done, info = self.env.step(action)
        self._frames.popleft()
        self._frames.append(new_observation)
        return self._stack_frames(), reward, done, info
