"""This module implements the resize and grayscale wrapper for gym env."""

import gym
from gym.spaces import Box
import numpy as np
from scipy.misc import imresize


class ResizeAndGrayscale(gym.Wrapper):
    """
    Resize and grayscale wrapper.

    Gym env wrapper for resizing to (w, h) and converting frames to
    grayscale.
    """

    def __init__(self, env, w, h):
        """Construct the class."""
        super(ResizeAndGrayscale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, shape=[w, h], dtype=np.float32)
        self.w = w
        self.h = h

    def _observation(self, obs):
        obs = imresize(obs, (self.w, self.h))
        obs = np.dot(obs[:, :, :3], [0.299, 0.587, 0.114]) / 255.0
        return obs

    def reset(self):
        """Gym env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """Gym env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
