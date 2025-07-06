import os
import skvideo.io
import cv2
import gym
import numpy as np
import random
import tqdm
from .od_envs.mujoco.call_mujoco_env import call_mujoco_env

class OffDynamicsMujocoEnv:

    def __init__(self, name, shift_level=None, action_repeat=1, size=(64, 64)):
        os.environ["MUJOCO_GL"] = "egl"
        env_config = {
            'env_name': name, #'walker2d-friction'
            'shift_level': shift_level,
        }
        self._env = call_mujoco_env(env_config)
        self._action_repeat = action_repeat
        self._size = size

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        #if self._obs_is_dict:
        #    spaces = self._env.observation_space.spaces.copy()
        #else:
        #    spaces = {self._obs_key: self._env.observation_space}
        return {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "state": self._env.observation_space,
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {'action': self._env.action_space}

    def step(self, action):
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action['action'])
            reward += rew or 0.0
            if done:
                break
        obs = {'image': self._env.render(mode='rgb_array', width=self._size[1], height=self._size[0]),
               'reward': float(reward),
               'is_first': False,
               'is_last': done,
               'is_terminal': done,
               'state': state,
        }
        info['success'] = int(obs['is_terminal'])
        return obs

    def reset(self):
        state = self._env.reset()
        obs = {'image': self._env.render(mode='rgb_array', width=self._size[1], height=self._size[0]),
               'reward': 0.0,
               'is_first': True,
               'is_last': False,
               'is_terminal': False,
               'state': state,
               }
        return obs
