import os
import skvideo.io
import cv2
import gym
import numpy as np
import random
import tqdm

class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ["MUJOCO_GL"] = "egl"
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if domain == "manip":
            from dm_control import manipulation

            self._env = manipulation.load(task + "_vision")
        elif domain == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite

            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                pentaped_walk=2,
                pentaped_run=2,
                pentaped_escape=2,
                pentaped_fetch=2,
                biped_walk=2,
                biped_run=2,
                biped_escape=2,
                biped_fetch=2,
                triped_walk=2,
                triped_run=2,
                triped_escape=2,
                triped_fetch=2,
                hexaped_walk=2,
                hexaped_run=2,
                hexaped_escape=2,
                hexaped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, self._env.physics.get_state().shape, dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action["action"])
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        obs = {
            "reward": reward,
            "state": self._env.physics.get_state().copy(),
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "state": self._env.physics.get_state().copy(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs


class RandomVideoSource:
    def __init__(self, env, video_dir, shape, total_frames=None, grayscale=False):
        """
        Args:
            filelist: a list of video files
        """
        self._env = env
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = [os.path.join(video_dir, file) for file in os.listdir(video_dir)]
        self.build_arr()
        self.current_idx = 0
        self._pixels_key = 'image'
        self.reset()

    def build_arr(self):
        if not self.total_frames:
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:              frames = skvideo.io.vread(fname)
                local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = cv2.resize(frames[i], (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0: random.shuffle(self.filelist)
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:              frames = skvideo.io.vread(fname)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames: break
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[..., None] ## THIS IS NOT A BUG! cv2 uses (width, height)
                        else:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0])) 
                        pbar.update(1)
                        total_frame_i += 1


    def step(self, action):
        obs = self._env.step(action)
        pixels = self._extract_pixels(obs)
        img = self.get_obs(pixels)
        obs['image'] = img
        return obs

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)
        obs = self._env.reset()
        pixels = self._extract_pixels(obs)
        img = self.get_obs(pixels)
        obs['image'] = img
        return obs

    def get_obs(self, img):
        img = img.transpose(1, 2, 0)
        mask = np.logical_and((img[:, :, 2] > img[:, :, 1]), (img[:, :, 2] > img[:, :, 0]))  # hardcoded for dmc
        bg = self.get_image()
        img[mask] = bg[mask]
        # img = img.transpose(2, 0, 1).copy()
        img = img.copy()
        # CHW to HWC for tensorflow
        return img

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def act_space(self):
        return self._env.act_space

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def _extract_pixels(self, obs):
        pixels = obs[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()
