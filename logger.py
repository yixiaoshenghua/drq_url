import csv
import json
import os
import shutil
from collections import defaultdict

import numpy as np

import torch
import torchvision
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

COMMON_TRAIN_FORMAT = [('episode', 'E', 'int'), ('step', 'S', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('duration', 'D', 'time')]

COMMON_EVAL_FORMAT = [('episode', 'E', 'int'), ('step', 'S', 'int'),
                      ('episode_reward', 'R', 'float')]

AGENT_TRAIN_FORMAT = {
    'drq': [('batch_reward', 'BR', 'float'), ('actor_loss', 'ALOSS', 'float'),
            ('critic_loss', 'CLOSS', 'float'),
            ('alpha_loss', 'TLOSS', 'float'), ('alpha_value', 'TVAL', 'float'),
            ('actor_entropy', 'AENT', 'float')]
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=False,
                 log_frequency=10000,
                 action_repeat=1,
                 agent='drq'):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        self._action_repeat = action_repeat
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)

        # Define fieldnames for the new detailed per-episode CSV log
        self._defined_episode_csv_fieldnames = ['type', 'episode', 'step', 'duration',
                                                'episode_extrinsic_reward', 'skill_id',
                                                'episode_intrinsic_reward_sum']

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _update_step(self, step):
        return step * self._action_repeat

    def _try_sw_log(self, key, value, step):
        step = self._update_step(step)
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        step = self._update_step(step)
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step, fps=25):
        """Internal method to log video to TensorBoard."""
        step = self._update_step(step)
        if self._sw is not None:
            # Input frames is expected to be a list of HWC uint8 numpy arrays (e.g., from VideoRecorder)
            video_tensor = torch.from_numpy(np.array(frames))  # Converts list of (H,W,C) to (T,H,W,C)
            # TensorBoard's add_video expects (N, T, C, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (T,H,W,C) -> (T,C,H,W) -> (1,T,C,H,W)
            self._sw.add_video(key, video_tensor, step, fps=fps)

    def _try_sw_log_histogram(self, key, histogram, step):
        step = self._update_step(step)
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step, fps=25, log_frequency=None):
        """Logs a video to TensorBoard."""
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval'), "Key for video log must start with 'train' or 'eval'"
        self._try_sw_log_video(key, frames, step, fps=fps)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        step = self._update_step(step)
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        else:
            raise f'invalid log type: {ty}'

    def log_episode_to_csv(self, episode_data):
        """
        Logs detailed per-episode data to a dedicated 'episodes.csv' file.
        This allows for fine-grained analysis of episode-level performance,
        including extrinsic rewards, intrinsic rewards, and skill IDs.

        Args:
            episode_data (dict): A dictionary containing data for a single episode.
                                 Expected keys are defined in self._defined_episode_csv_fieldnames.
        """
        log_path = os.path.join(self._log_dir, 'episodes.csv')
        file_exists = os.path.exists(log_path)

        # Ensure all expected fieldnames are present in episode_data, filling with defaults if not.
        # This makes the CSV robust to missing data points for an episode.
        for field in self._defined_episode_csv_fieldnames:
            if field not in episode_data:
                if field == 'skill_id':
                    episode_data[field] = -1  # Default for non-DIAYN or if skill is not applicable
                elif field == 'episode_intrinsic_reward_sum':
                    episode_data[field] = 0.0  # Default if intrinsic rewards are not used or applicable
                elif field == 'duration':
                    episode_data[field] = -1.0 # Default if duration is not tracked for this episode type (e.g., eval)
                # Other fields are expected to be present.
                # Consider adding a warning or error for other critical missing fields if necessary.

        try:
            with open(log_path, 'a', newline='') as f:
                # Use DictWriter for robust CSV writing, ensuring columns align with headers.
                writer = csv.DictWriter(f, fieldnames=self._defined_episode_csv_fieldnames)
                if not file_exists or os.path.getsize(log_path) == 0:
                    writer.writeheader()  # Write header only if file is new or empty
                writer.writerow(episode_data)
        except IOError as e:
            print(f"Error writing to episodes.csv: {e}")
