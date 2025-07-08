import cv2
import numpy as np
import gymnasium as gym
from envs.wrappers import EnvSpec, Box


class CarRacingEnv:
    def __init__(self, name="CarRacing-v2", seed=None, action_repeat=1, size=(64, 64), flatten_obs=False):
        """
        Gym Car Racing V2 环境包装器
        """
        self._env = gym.make(name, render_mode='rgb_array')
        self._seed = seed
        self._size = size
        self._action_repeat = action_repeat
        
        # 初始化状态跟踪
        self._current_obs = None
        self._done = False

        self.flatten_obs = flatten_obs
        
    @property
    def obs_space(self):
        spaces = {
            "image": Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        return {"action": Box(self._env.action_space.low, self._env.action_space.high, self._env.action_space.shape, self._env.action_space.dtype)}
    
    @property
    def spec(self):
        return EnvSpec(
            obs_space=self.obs_space,
            act_space=self.act_space
        )

    def step(self, action):
        total_reward = 0.0
        terminal = False
        
        for _ in range(self._action_repeat):
            if self._done:
                break
                
            obs, reward, done, truncated, info = self._env.step(action["action"])
            total_reward += reward
            self._current_obs = obs
            self._done = done or truncated
            
            if self._done:
                terminal = True
                break
        
        # Car Racing 没有显式的"success"概念
        success = False
        
        return {
            "image": self._process_image(self._current_obs),
            "reward": np.float32(total_reward),
            "is_first": False,
            "is_last": self._done,
            "is_terminal": terminal,
            "success": success
        }

    def reset(self):
        if self._seed is not None:
            obs, info = self._env.reset(seed=self._seed)
            # 重置后清除种子以确保后续随机性
            self._seed = None  
        else:
            obs, info = self._env.reset()
            
        self._current_obs = obs
        self._done = False
        
        return {
            "image": self._process_image(obs),
            "reward": np.float32(0.0),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "success": False
        }

    def render(self, mode='offscreen'):
        """渲染当前帧并调整大小"""
        img = self._env.render()
        return self._resize_image(img)

    def _process_image(self, obs):
        """处理并调整图像大小"""

        obs = self._resize_image(obs)
        if self.flatten_obs:
            obs = obs.flatten()
        return obs

    def _resize_image(self, img):
        """调整图像大小并转换颜色格式"""
        # 注意: OpenCV 使用 (宽, 高) 而 numpy 数组是 (高, 宽)
        resized = cv2.resize(img, (self._size[1], self._size[0]))
        return resized

    def close(self):
        self._env.close()