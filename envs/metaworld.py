import os
import gymnasium
from gymnasium.core import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import gym
import numpy as np
import cv2
import mujoco


class MetaWorld:

    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera='corner', use_gripper=False):
        '''
        camera: one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        '''
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env.render_mode = "rgb_array"
        self._env.weight = size[1]
        self._env.height = size[0]
        self._env.camera_name = camera
        # self._env.mujoco_renderer.camera_id = mujoco.mj_name2id(
        #     self._env.model,
        #     mujoco.mjtObj.mjOBJ_CAMERA,
        #     camera,
        # )
        # self._env.mujoco_renderer.height = size[0]
        # self._env.mujoco_renderer.width = size[1]
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._use_gripper = use_gripper

        self._camera = camera

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            # "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        if self._use_gripper:
            spaces["gripper_image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, _, info = self._env.step(action["action"])
            success = float(info["success"])
            reward += rew or 0.0
            if done or success == 1.0:
                break
        assert success in [0.0, 1.0]
        image = self.render()
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": image,
            # "image": self._env.sim.render(
            #     *self._size, mode="offscreen", camera_name=self._camera
            # ),
            # "state": state,
            "success": success,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state, info = self._env.reset()
        image = self.render()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": image,
            # "image": self._env.sim.render(
            #     *self._size, mode="offscreen", camera_name=self._camera
            # ),
            # "state": state,
            "success": False,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def render(self, mode='offscreen'):
        return cv2.resize(self._env.render()[::-1], self._size)
    

class MultiViewMetaWorld:

    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera_keys='corner|topview', use_gripper=False):
        '''
        camera: one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        '''
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env.render_mode = "rgb_array"
        self._env.weight = size[1]
        self._env.height = size[0]
        self.camera_keys = camera_keys.split('|')
        # self._env.mujoco_renderer.camera_id = mujoco.mj_name2id(
        #     self._env.model,
        #     mujoco.mjtObj.mjOBJ_CAMERA,
        #     camera,
        # )
        # self._env.mujoco_renderer.height = size[0]
        # self._env.mujoco_renderer.width = size[1]
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._use_gripper = use_gripper

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, (self._size[0], self._size[1] * len(self.camera_keys), 3), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        if self._use_gripper:
            spaces["gripper_image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, _, info = self._env.step(action["action"])
            success = float(info["success"])
            reward += rew or 0.0
            if done or success == 1.0:
                break
        assert success in [0.0, 1.0]
        image = self.render()
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": image,
            "success": success,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def reset(self):
        state, info = self._env.reset()
        image = self.render()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": image,
            "success": False,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def render(self, mode='offscreen'):
        imgs = []
        for camera in self.camera_keys:
            self._env.camera_name = camera
            imgs.append(cv2.resize(self._env.render()[::-1], self._size))
        imgs = np.concatenate(imgs, axis=1)
        return imgs

class MetaWorldMovableCameraWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, seed:int, size=224, viewpoint_mode='controlled', viewpoint_randomization_type='weak'):
        super().__init__(env)

        ##################################################################################
        '''
        controlled mode: the viewpoint is controlled by a sensory policy, smoothly changes in a valid range;
        '''
        self.viewpoint_mode = viewpoint_mode
        self.viewpoint_randomization_type = viewpoint_randomization_type
        assert self.viewpoint_randomization_type in ['weak', 'medium', 'strong'], viewpoint_randomization_type
        assert self.viewpoint_mode in ['controlled', 'shake', 'rotation', 'translation', 'zoom'], viewpoint_mode
        self.init_camera_config = {
            "distance": 1.25,
            "azimuth": 145,
            "elevation": -25.0,
            "lookat": np.array([0.0, 0.65, 0.0]),
            }
        self.curr_camera_config = {
            "distance": 1.25,
            "azimuth": 145,
            "elevation": -25.0,
            "lookat": np.array([0.0, 0.65, 0.0]),
            }
        self.camera_range_config = {
            "distance_min": 0.50,
            "distance_max": 2.50,
            "azimuth_min": 115,
            "azimuth_max": 175,
            "elevation_min": -85.0,
            "elevation_max": -5.0,
            "lookat_min": np.array([0.0, 0.15, 0.0]),
            "lookat_max": np.array([0.8, 1.00, 0.8]),
            }
        self.viewpoint_step = 0
        self.viewpoint_period = 100
        if self.viewpoint_mode == 'controlled':
            self.camera_unit_config = {
                "distance": (self.camera_range_config['distance_max'] - self.camera_range_config['distance_min']) / 10,
                "azimuth": (self.camera_range_config['azimuth_max'] - self.camera_range_config['azimuth_min']) / 10,
                "elevation": (self.camera_range_config['elevation_max'] - self.camera_range_config['elevation_min']) / 10,
                "lookat": (self.camera_range_config['lookat_max'] - self.camera_range_config['lookat_min']) / 10,
                }
            
        ##################################################################################

        self.size = size
        self.unwrapped.model.vis.global_.offwidth = size
        self.unwrapped.model.vis.global_.offheight = size
        
        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)

    def reset(self):
        ##################################################################################
        for k, v in self.curr_camera_config.items():
            self.curr_camera_config[k] = self.init_camera_config[k]
        self.viewpoint_step = 0
        ##################################################################################
        obs, info = super().reset()
        
        return obs, info

    def step(self, action):
        next_obs, reward, done, truncate, info = self.env.step(action) 
        
        return next_obs, reward, done, truncate, info
        
    def render(self, camera_config=None, sensory_action=None):
        if camera_config == None:
            if self.viewpoint_mode == 'controlled':
                assert sensory_action is not None
                for i, k in enumerate(self.curr_camera_config.keys()):
                    self.curr_camera_config[k] += sensory_action[i] * self.camera_unit_config[k]
            elif self.viewpoint_mode == 'rotation':
                self.viewpoint_rotation = dict(weak=[135, 155], medium=[125, 165], strong=[115, 175])[self.viewpoint_randomization_type]
                self.curr_camera_config['azimuth'] = self.init_camera_config['azimuth'] + \
                        np.sin( self.viewpoint_step / self.viewpoint_period * 2 * np.pi) * \
                              (self.viewpoint_rotation[1] - self.viewpoint_rotation[0])
                for i, k in enumerate(self.curr_camera_config.keys()):
                    if k != 'azimuth':
                        self.curr_camera_config[k] = self.init_camera_config[k] + np.random.normal(0.0, (self.camera_range_config[k+'_max'] - self.camera_range_config[k+'_min'])/50)
            elif self.viewpoint_mode == 'shake':
                self.viewpoint_shake = dict(weak=50, medium=25, strong=10)[self.viewpoint_randomization_type]
                for i, k in enumerate(self.curr_camera_config.keys()):
                    self.curr_camera_config[k] = self.init_camera_config[k] + np.random.normal(0.0, (self.camera_range_config[k+'_max'] - self.camera_range_config[k+'_min'])/self.viewpoint_shake)
            elif self.viewpoint_mode == 'translation':
                self.viewpoint_translation = dict(weak=[0.45, 0.80], medium=[0.30, 0.90], strong=[0.15, 1.00])[self.viewpoint_randomization_type]
                self.curr_camera_config['lookat'][1] = self.init_camera_config['lookat'][1] + \
                        np.sin( self.viewpoint_step / self.viewpoint_period * 2 * np.pi) * \
                              (self.viewpoint_translation[1] - self.viewpoint_translation[0])
                for i, (k, v) in enumerate(self.curr_camera_config.items()):
                    if k != 'lookat':
                        self.curr_camera_config[k] = self.init_camera_config[k] + np.random.normal(0.0, (self.camera_range_config[k+'_max'] - self.camera_range_config[k+'_min'])/50)
                    else:
                        for lookat_idx in range(len(v)):
                            if lookat_idx != 1:
                                self.curr_camera_config[k][lookat_idx] = self.init_camera_config[k][lookat_idx] + np.random.normal(0.0, (self.camera_range_config[k+'_max'][lookat_idx] - self.camera_range_config[k+'_min'][lookat_idx])/50)
            elif self.viewpoint_mode == 'zoom':
                self.viewpoint_zoom = dict(weak=[1.0, 1.5], medium=[0.75, 2.0], strong=[0.5, 2.5])[self.viewpoint_randomization_type]
                self.curr_camera_config['distance'] = self.init_camera_config['distance'] + \
                        np.sin( self.viewpoint_step / self.viewpoint_period * 2 * np.pi) * \
                              (self.viewpoint_zoom[1] - self.viewpoint_zoom[0])
                for i, k in enumerate(self.curr_camera_config.keys()):
                    if k != 'distance':
                        self.curr_camera_config[k] = self.init_camera_config[k] + np.random.normal(0.0, (self.camera_range_config[k+'_max'] - self.camera_range_config[k+'_min'])/50)
        else:
            self.curr_camera_config = camera_config
        for k, v in self.curr_camera_config.items():
            self.curr_camera_config[k] = np.clip(self.curr_camera_config[k], self.camera_range_config[k+'_min'], self.camera_range_config[k+'_max'])
        self.viewpoint_step += 1
        self.unwrapped.mujoco_renderer = MujocoRenderer(self.env.model, self.env.data, self.curr_camera_config)
        return self.unwrapped.mujoco_renderer.render(render_mode = "rgb_array")

# def setup_metaworld_env(task_name:str, seed:int, size:int, viewpoint_mode:str, viewpoint_randomization_type:str):
#     env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
#     env = MetaWorldMovableCameraWrapper(env_cls(), seed, size, viewpoint_mode=viewpoint_mode, viewpoint_randomization_type=viewpoint_randomization_type)
#     return env

class ViewMetaWorld:

    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None, viewpoint_mode='shake', viewpoint_randomization_type='strong', use_gripper=False):
        '''
        camera: one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        '''
        os.environ["MUJOCO_GL"] = "egl"
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = MetaWorldMovableCameraWrapper(env_cls(seed=seed), seed, size[0], viewpoint_mode=viewpoint_mode, viewpoint_randomization_type=viewpoint_randomization_type)
        self._size = size
        self._action_repeat = action_repeat
        self._use_gripper = use_gripper

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            # "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        if self._use_gripper:
            spaces["gripper_image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, _, info = self._env.step(action["action"])
            success = float(info["success"])
            reward += rew or 0.0
            if done or success == 1.0:
                break
        assert success in [0.0, 1.0]
        image = self.render()
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": image,
            # "image": self._env.sim.render(
            #     *self._size, mode="offscreen", camera_name=self._camera
            # ),
            # "state": state,
            "success": success,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def reset(self):
        # if self._camera == "corner2":
        #     self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state, info = self._env.reset()
        image = self.render()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": image,
            # "image": self._env.sim.render(
            #     *self._size, mode="offscreen", camera_name=self._camera
            # ),
            # "state": state,
            "success": False,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def render(self, mode='offscreen'):
        return self._env.render()