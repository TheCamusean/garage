"""This module implements the repeat action wrapper for gym env."""

import gym


class RepeatAction(gym.Wrapper):
    """
    Repeat action wrapper.

    Gym env wrapper for repeating action for n frames.
    """

    def __init__(self, env, frame_to_repeat):
        """Construct the class."""
        super(RepeatAction, self).__init__(env)
        self.frame_to_repeat = frame_to_repeat

    def step(self, action):
        """Gym env step function."""
        for i in range(self.frame_to_repeat):
            obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        """To suppress the warning from gym."""
        return self.env.reset()
