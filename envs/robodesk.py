import os

import gym
import numpy as np
import cv2

'''
Tasks                   | Descriptions
open_slide	            | Push the sliding door all the way to the right, navigating around the other objects.
open_drawer	            | Pull the dark brown drawer all the way open.
push_green	            | Push the green button to turn the green light on.
stack_blocks	        | Stack the upright blue block on top of the flat green block.
upright_block_off_table	| Push the blue upright block off the table.
flat_block_in_bin	    | Push the green flat block into the blue bin.
flat_block_in_shelf	    | Push the green flat block into the shelf, navigating around the other blocks.
lift_upright_block	    | Grasp the blue upright block and lift it above the table.
lift_ball	            | Grasp the magenta ball and lift it above the table.
'''

class RoboDesk:

    def __init__(self, task, action_repeat=1, time_limit=500, obs_key='image', act_key='action', reward='dense', render_size=(64, 64)):
        import robodesk
        self.task = task
        self._env = robodesk.RoboDesk(task=task, reward=reward, action_repeat=action_repeat, episode_length=time_limit, image_size=render_size[0])
        self._size = render_size
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key

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
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool_)
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            action = self._env.action_space.spaces.copy()[self._act_key]
        else:
            action = self._env.action_space
        return {'action': action}

    def step(self, action):
        state, reward, done, info = self._env.step(action[self._act_key])
        obs = {'reward': float(reward),
               'is_first': False,
               'is_last': done,
               'is_terminal': info.get('is_terminal', done)}
        obs['image'] = state['image']
        obs['success'] = np.bool_(self._env._get_task_reward(self.task, 'success'))#int(obs['is_terminal'])
        return obs

    def reset(self):
        state = self._env.reset()
        obs = {'reward': 0.0,
               'is_first': True,
               'is_last': False,
               'is_terminal': False}
        obs['image'] = state['image']
        obs['success'] = False
        return obs
    
    # def _get_obs(self):
    #     '''
    #     returns: state
    #     image	Box(0, 255, (64, 64, 3), np.uint8)
    #     qpos_robot	Box(-np.inf, np.inf, (9,), np.float32)
    #     qvel_robot	Box(-np.inf, np.inf, (9,), np.float32)
    #     qpos_objects	Box(-np.inf, np.inf, (26,), np.float32)
    #     qvel_objects	Box(-np.inf, np.inf, (26,), np.float32)
    #     end_effector	Box(-np.inf, np.inf, (3,), np.float32)
    #     '''
    #     return {'image': self._env.render(resize=True),
    #             'qpos_robot': self._env.physics.data.qpos[:self._env.num_joints].copy(),
    #             'qvel_robot': self._env.physics.data.qvel[:self._env.num_joints].copy(),
    #             'end_effector': self._env.physics.named.data.site_xpos['end_effector'],
    #             'qpos_objects': self._env.physics.data.qvel[self._env.num_joints:].copy(),
    #             'qvel_objects': self._env.physics.data.qvel[self._env.num_joints:].copy()}