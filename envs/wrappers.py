import atexit
import sys
import traceback
from collections import OrderedDict, deque
# from qwen_vl_utils import process_vision_info
import cloudpickle
import cv2
import gym
import numpy as np
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_sinusoidal_positional_encoding(max_length, d_model=512):
    """
    Generate sinusoidal position encoding。
    
    Params:
    - max_length: The maximum length of position encoding。
    - d_model: The dimension of feature。
    
    Returns:
    - Positional encoding: shape [max_length, d_model]
    """
    positional_encoding = np.zeros((max_length, d_model))
    position = np.arange(0, max_length, dtype=np.float32).reshape((-1, 1))
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)
    
    return positional_encoding

class GymWrapper:

    def __init__(self, env, obs_key="image", act_key="action"):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._act_is_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = float(reward)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", done)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs

class LLMActionWrapper:
    def __init__(self, env, config, llm_packages, gap_frame=4, device='cuda:0'):
        '''
        llm_packages: (logdir, model, processor, viclip_global_instance)
        '''
        self._env = env
        self.device = device
        self.gap_frame = gap_frame
        self.config = config
        #############################################################
        self.logdir, self.model, self.processor, self.viclip_global_instance = llm_packages
        # Messages containing a images list as a video and a text query
        self.prompt = "Please describe possible actions (if there appears significant action then describe it else output null) in this video with format: [Executing body][Action][Target object][Moving direction][Displacement generated]. Example 1: A hand lifted a brick and moved it the distance of one arm. Example 2: A leg kicked the ball two meters away. Example 3: null."
        
        (self.logdir / '.obs_cache').mkdir(parents=True, exist_ok=True)
        self.img_list = [str(self.logdir / '.obs_cache' / f'{i}.png') for i in range(0, self.gap_frame)]
        self.pos = 0
        self.pos_emb = get_sinusoidal_positional_encoding(max_length=config.replay.minlen)
        self.acs_pointer = 0
        self.emb_acs = None
        self.output_text = None
        #############################################################

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)
    
    def step(self, action):
        obs = self._env.step(action)
        self.acs_pointer = (self.acs_pointer + 1) % self.gap_frame
        self.pos = (self.pos + 1) % self.config.replay.minlen
        if self.acs_pointer == 0:
            self.emb_acs, self.output_text = self.encode_acs()
        self.emb_acs += self.pos_emb[self.pos]
        obs['llm_acs'] = self.emb_acs
        # obs['llm_text'] = self.output_text
        # print(self.output_text)
        # save image
        cv2.imwrite(self.logdir / '.obs_cache' / f"{self.acs_pointer}.png", obs['image'][:, :, ::-1])
        return obs
    
    def reset(self):
        obs = self._env.reset()
        self.acs_pointer = 0
        self.pos = 0
        # save image
        cv2.imwrite(self.logdir / '.obs_cache' / f"{self.acs_pointer}.png", obs['image'][:, :, ::-1])
        self.emb_acs, self.output_text = self.encode_acs(['null'])
        self.emb_acs += self.pos_emb[self.pos]
        obs['llm_acs'] = self.emb_acs
        # obs['llm_text'] = self.output_text
        # print(self.output_text)
        return obs

    @torch.no_grad()
    def encode_acs(self, output_text=None):
        if output_text is None:
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": self.img_list,
                            },
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                # fps=fps,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(self.device)

            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        emb_acs = self.viclip_global_instance.viclip.get_txt_feat(output_text)
        return emb_acs.detach().cpu().numpy().squeeze(0), output_text[0]

class ViClipWrapper:
    def __init__(self, env, hd_rendering=False, device='cuda'):
        self._env = env
        from nets import ViCLIPGlobalInstance
        viclip_global_instance = ViCLIPGlobalInstance()

        if not viclip_global_instance._instantiated:
            viclip_global_instance.instantiate(device)
        self.viclip_model = viclip_global_instance.viclip
        self.n_frames = self.viclip_model.n_frames
        self.viclip_emb_dim = viclip_global_instance.viclip_emb_dim
        self.n_frames = self.viclip_model.n_frames
        self.buffer = deque(maxlen=self.n_frames)
        # NOTE: these are hardcoded for now, as they are the best settings
        self.accumulate = True
        self.accumulate_buffer = []
        self.anticipate_conv1 = False
        self.hd_rendering = hd_rendering

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def hd_render(self, obs):
        if not self.hd_rendering:
            return obs['observation']
        if self._env._domain_name in ['mw', 'kitchen', 'mujoco']:
            return self.get_visual_obs((224,224,))
        else:
            render_kwargs = {**getattr(self, '_render_kwargs', {})}
            render_kwargs.update({'width' : 224, 'height' : 224})
            return self._env.physics.render(**render_kwargs).transpose(2,0,1)

    def preprocess(self, x):
        return x

    def process_accumulate(self, process_at_once=4): # NOTE: this could be varied for increasing FPS, depending on the size of the GPU
        self.accumulate = False
        x = np.stack(self.accumulate_buffer, axis=0)
        # Splitting in chunks
        chunks = []
        chunk_idxs = list(range(0, x.shape[0] + 1, process_at_once))
        if chunk_idxs[-1] != x.shape[0]:
            chunk_idxs.append(x.shape[0])
        start = 0
        for end in chunk_idxs[1:]:
            embeds = self.clip_process(x[start:end], bypass=True)
            chunks.append(embeds.cpu())
            start = end
        embeds = torch.cat(chunks, dim=0)
        assert embeds.shape[0] == len(self.accumulate_buffer)
        self.accumulate = True
        self.accumulate_buffer = []
        return [*embeds.cpu().numpy()], 'clip_video'
    
    def process_episode(self, obs, process_at_once=8):
        self.accumulate = False
        sequences = []
        for j in range(obs.shape[0] - self.n_frames + 1):
            sequences.append(obs[j:j+self.n_frames].copy())
        sequences = np.stack(sequences, axis=0)

        idx_start = 0
        clip_vid = []
        for idx_end in range(process_at_once, sequences.shape[0] + process_at_once, process_at_once):
            x = sequences[idx_start:idx_end]
            with torch.no_grad(): # , torch.cuda.amp.autocast():
                x = self.clip_process(x, bypass=True) 
            clip_vid.append(x)
            idx_start = idx_end
        if len(clip_vid) == 1: # process all at once
            embeds = clip_vid[0]
        else:
            embeds = torch.cat(clip_vid, dim=0)
        pad = torch.zeros( (self.n_frames - 1, *embeds.shape[1:]), device=embeds.device, dtype=embeds.dtype)
        embeds = torch.cat([pad, embeds], dim=0)
        assert embeds.shape[0] == obs.shape[0], f"Shapes are different {embeds.shape[0]} {obs.shape[0]}"
        return embeds.cpu().numpy()

    def get_sequence(self,):
        return np.expand_dims(np.stack(self.buffer, axis=0), axis=0)
    
    def clip_process(self, x, bypass=False):
        if len(self.buffer) == self.n_frames or bypass:
            if self.accumulate:
                self.accumulate_buffer.append(self.preprocess(x)[0])
                return torch.zeros(self.viclip_emb_dim)
            with torch.no_grad():
                B, n_frames, C, H, W = x.shape
                obs = torch.from_numpy(x.copy().reshape(B * n_frames, C, H, W)).to(self.viclip_model.device)
                processed_obs = self.viclip_model.preprocess_transf(obs / 255)
                reshaped_obs = processed_obs.reshape(B, n_frames, 3, processed_obs.shape[-2], processed_obs.shape[-1])
                video_embed = self.viclip_model.get_vid_features(reshaped_obs)
            return video_embed.detach()
        else:
            return torch.zeros(self.viclip_emb_dim)

    def step(self, action):
        ts, obs = self._env.step(action)
        self.buffer.append(self.hd_render(obs))
        obs['clip_video'] = self.clip_process(self.get_sequence()).cpu().numpy()
        return ts, obs

    def reset(self,):
        # Important to reset the buffer        
        self.buffer = deque(maxlen=self.n_frames)

        ts, obs = self._env.reset()
        self.buffer.append(self.hd_render(obs))
        obs['clip_video'] = self.clip_process(self.get_sequence()).cpu().numpy()
        return ts, obs

    def __getattr__(self, name):
        if name == 'obs_space':
            space = self._env.obs_space
            space['clip_video'] = gym.spaces.Box(-np.inf, np.inf, (self.viclip_emb_dim,), dtype=np.float32)  
            return space
        return getattr(self._env, name)


class Dummy:

    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        return {"action": gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

    def step(self, action):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
        }

    def reset(self):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            obs["is_terminal"] = True
            self._step = 0
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()
    
    @property
    def obs_space(self):
        return self._env.obs_space
    
    @property
    def act_space(self):
        return self._env.act_space


class NormalizeAction:

    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})
    
    @property
    def obs_space(self):
        return self._env.obs_space



class OneHotAction:

    def __init__(self, env, key="action"):
        assert hasattr(env.act_space[key], "n")
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

class MultiDiscreteAction:

    def __init__(self, env, key="action"):
        assert hasattr(env.act_space[key], "nvec")
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        return self._env.obs_space


    @property
    def act_space(self):
        shape = self._env.act_space[self._key].nvec
        space = gym.spaces.Box(low=0, high=1, shape=(int(shape.sum()), ), dtype=np.float32)
        space.nvec = shape
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        output_action = []
        cum_sum = 0
        reference = np.zeros_like(action[self._key])
        for i in self._env.act_space[self._key].nvec:
            index = np.argmax(action[self._key][cum_sum:cum_sum + i])
            output_action.append(index)
            reference[cum_sum + index] = 1
            cum_sum += i
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f"Invalid multi-discrete action:\n{action[self._key]}")

        return self._env.step({**action, self._key: output_action})

    def reset(self):
        return self._env.reset()

class ResizeImage:

    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces
    
    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:

    def __init__(self, env, key="image"):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces
    
    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render("rgb_array")
        return obs

class FrameStack:
    def __init__(self, env, k, key='image'):
        self._env = env
        self._k = k
        self._key = key
        self._frames = deque([], maxlen=k)


    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        shp = spaces["image"].shape
        spaces[self._key] = gym.spaces.Box(0, 255, (shp[:-1] + (shp[-1] * self._k,)), np.uint8)
        return spaces
    

    @property
    def act_space(self):
        return self._env.act_space

    def reset(self):
        timestep = self._env.reset()
        obs = timestep['image']
        for _ in range(self._k):
            self._frames.append(obs)
        timestep['image'] = self._get_obs()
        return timestep

    def step(self, action):
        timestep = self._env.step(action)
        self._frames.append(timestep['image'])
        timestep['image'] = self._get_obs()
        return timestep

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=-1)


class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.
