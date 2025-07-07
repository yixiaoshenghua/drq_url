import numpy as np
from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time
import utils
from dowel import Histogram
from replay_buffer import TrajectoryBatch, obtain_exact_trajectories
import sac_utils
from collections import defaultdict
import tqdm

class DictBatchDataset:
    """Use when the input is the dict type."""
    def __init__(self, inputs, batch_size):
        self._inputs = inputs
        self._batch_size = batch_size
        self._size = list(self._inputs.values())[0].shape[0]
        if batch_size is not None:
            self._ids = np.arange(self._size)
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._size * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield self._inputs
        else:
            if update:
                self.update()
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = {
                    k: v[batch_ids]
                    for k, v in self._inputs.items()
                }
                yield batch

    def update(self):
        np.random.shuffle(self._ids)

class OptimizerGroupWrapper:
    """A wrapper class to handle torch.optim.optimizer.
    """

    def __init__(self,
                 optimizers,
                 max_optimization_epochs=1,
                 minibatch_size=None):
        self._optimizers = optimizers
        self._max_optimization_epochs = max_optimization_epochs
        self._minibatch_size = minibatch_size

    def get_minibatch(self, data, max_optimization_epochs=None):
        batch_dataset = DictBatchDataset(data, self._minibatch_size)

        if max_optimization_epochs is None:
            max_optimization_epochs = self._max_optimization_epochs

        for _ in range(max_optimization_epochs):
            for dataset in batch_dataset.iterate():
                yield dataset

    def zero_grad(self, keys=None):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        # optimize to param = None style.
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            self._optimizers[key].zero_grad()

    def step(self, keys=None, **closure):
        """Performs a single optimization step.

        Arguments:
            **closure (callable, optional): A closure that reevaluates the
                model and returns the loss.

        """
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            self._optimizers[key].step(**closure)

    def target_parameters(self, keys=None):
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            for pg in self._optimizers[key].param_groups:
                for p in pg['params']:
                    yield p

class MeasureAndAccTime:
    def __init__(self, target):
        assert isinstance(target, list)
        assert len(target) == 1
        self._target = target

    def __enter__(self):
        self._time_enter = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target[0] += (time.time() - self._time_enter)

def compute_total_norm(parameters, norm_type=2):
    # Code adopted from clip_grad_norm_().
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class DRQ_METRAAgent(object):
    """
    DrQ agent with METRA skill discovery integration.
    Manages the actor, critic, and their updates.
    Configuration is now passed via explicit arguments (formerly Hydra).
    """
    def __init__(self,
            env,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,
            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,
            dual_reg,
            dual_slack,
            dual_dist,
            pixel_shape,
            env_name,
            algo,
            env_spec,
            option_policy,
            traj_encoder,
            skill_dynamics,
            dist_predictor,
            dual_lam,
            optimizer,
            alpha,
            max_path_length,
            n_epochs_per_eval,
            n_epochs_per_log,
            n_epochs_per_tb,
            n_epochs_per_save,
            n_epochs_per_pt_save,
            n_epochs_per_pkl_update,
            dim_option,
            num_random_trajectories,
            num_video_repeats,
            eval_record_video,
            video_skip_frames,
            eval_plot_axis,
            name='IOD',
            device=torch.device('cuda'),
            sample_cpu=True,
            num_train_per_epoch=1,
            discount=0.99,
            sd_batch_norm=False,
            skill_dynamics_obs_dim=None,
            trans_minibatch_size=None,
            trans_optimization_epochs=None,
            discrete=False,
            unit_length=False,
            batch_size=32,
            snapshot_dir=None,
    ):
        self._env = env
        self.env_name = env_name
        self.algo = algo

        self.step_itr = 0
        self.snapshot_dir = snapshot_dir

        self.discount = discount
        self.max_path_length = max_path_length

        self.device = device
        self.sample_cpu = sample_cpu
        self.option_policy = option_policy.to(self.device)
        self.traj_encoder = traj_encoder.to(self.device)
        self.dual_lam = dual_lam.to(self.device)
        self.param_modules = {
            'traj_encoder': self.traj_encoder,
            'option_policy': self.option_policy,
            'dual_lam': self.dual_lam,
        }
        if skill_dynamics is not None:
            self.skill_dynamics = skill_dynamics.to(self.device)
            self.param_modules['skill_dynamics'] = self.skill_dynamics
        if dist_predictor is not None:
            self.dist_predictor = dist_predictor.to(self.device)
            self.param_modules['dist_predictor'] = self.dist_predictor

        self.alpha = alpha
        self.name = name

        self.dim_option = dim_option

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_log = n_epochs_per_log
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.n_epochs_per_pkl_update = n_epochs_per_pkl_update
        self.num_random_trajectories = num_random_trajectories
        self.num_video_repeats = num_video_repeats
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames
        self.eval_plot_axis = eval_plot_axis

        assert isinstance(optimizer, OptimizerGroupWrapper)
        self._optimizer = optimizer

        self._sd_batch_norm = sd_batch_norm
        self._skill_dynamics_obs_dim = skill_dynamics_obs_dim

        if self._sd_batch_norm:
            self._sd_input_batch_norm = torch.nn.BatchNorm1d(self._skill_dynamics_obs_dim, momentum=0.01).to(self.device)
            self._sd_target_batch_norm = torch.nn.BatchNorm1d(self._skill_dynamics_obs_dim, momentum=0.01, affine=False).to(self.device)
            self._sd_input_batch_norm.eval()
            self._sd_target_batch_norm.eval()

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self.discrete = discrete
        self.unit_length = unit_length
        self.batch_size = batch_size

        self.traj_encoder.eval()

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None

        self._start_time = time.time()
        self.total_env_steps = 0

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    def _get_concat_obs(self, obs, option):
        return utils.get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self):
        if self.discrete: # False
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, self.batch_size)])
        else:
            random_options = np.random.randn(self.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            extras = self._generate_option_extras(random_options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)
    
    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data
    
    def train(self, n_epochs):
        last_return = None
        self.total_epoch = 0
        with utils.GlobalContext({'phase': 'train', 'policy': 'sampling'}):
            for self.total_epoch in tqdm.tqdm(n_epochs, desc=f'Training {self.name}'):
                '''
                full_tb_epochs=0,
                log_period=self.n_epochs_per_log,
                tb_period=self.n_epochs_per_tb,
                pt_save_period=self.n_epochs_per_pt_save,
                pkl_update_period=self.n_epochs_per_pkl_update,
                new_save_period=self.n_epochs_per_save,
                '''
                for p in self.policy.values():
                    p.eval()
                self.traj_encoder.eval()

                if self.n_epochs_per_eval != 0 and self.step_itr % self.n_epochs_per_eval == 0:
                    self._evaluate_policy()

                for p in self.policy.values():
                    p.train()
                self.traj_encoder.train()

                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        step_path = self._get_train_trajectories()
                    # TODO: increment the total_env_steps
                    self.total_env_steps += ...
                    last_return = self.train_once(
                        self.step_itr,
                        step_path,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )

                self.step_itr += 1

                # TODO: logging

        return last_return

    def train_once(self, itr, paths, extra_scalar_metrics={}):
        logging_enabled = ((self.step_itr + 1) % self.n_epochs_per_log == 0)

        data = self.process_samples(paths)

        time_computing_metrics = [0.0]
        time_training = [0.0]

        with MeasureAndAccTime(time_training):
            tensors = self._train_once_inner(data)

        performence = utils.log_performance_ex(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )
        discounted_returns = performence['discounted_returns']
        undiscounted_returns = performence['undiscounted_returns']

        if logging_enabled:
            prefix_tabular = utils.get_metric_prefix()
            with utils.get_tabular().prefix(prefix_tabular + self.name + '/'), utils.get_tabular(
                    'plot').prefix(prefix_tabular + self.name + '/'):
                def _record_scalar(key, val):
                    utils.get_tabular().record(key, val)

                def _record_histogram(key, val):
                    utils.get_tabular('plot').record(key, Histogram(val))

                for k in tensors.keys():
                    if tensors[k].numel() == 1:
                        _record_scalar(f'{k}', tensors[k].item())
                    else:
                        _record_scalar(f'{k}', np.array2string(tensors[k].detach().cpu().numpy(), suppress_small=True))
                with torch.no_grad():
                    total_norm = compute_total_norm(self.all_parameters())
                    _record_scalar('TotalGradNormAll', total_norm.item())
                    for key, module in self.param_modules.items():
                        total_norm = compute_total_norm(module.parameters())
                        _record_scalar(f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}', total_norm.item())
                for k, v in extra_scalar_metrics.items():
                    _record_scalar(k, v)
                _record_scalar('TimeComputingMetrics', time_computing_metrics[0])
                _record_scalar('TimeTraining', time_training[0])

                path_lengths = [
                    len(path['actions'])
                    for path in paths
                ]
                _record_scalar('PathLengthMean', np.mean(path_lengths))
                _record_scalar('PathLengthMax', np.max(path_lengths))
                _record_scalar('PathLengthMin', np.min(path_lengths))

                _record_histogram('ExternalDiscountedReturns', np.asarray(discounted_returns))
                _record_histogram('ExternalUndiscountedReturns', np.asarray(undiscounted_returns))

        return np.mean(undiscounted_returns)

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_te(tensors, v)
            self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)

        return tensors
    
    def _gradient_descent(self, loss, optimizer_keys):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._optimizer.step(keys=optimizer_keys)

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            if self.discrete:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['options']).sum(dim=1)
                rewards = inner

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            else:
                raise NotImplementedError

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(v['obs'], v['options'])
        next_processed_cat_obs = self._get_concat_obs(v['next_obs'], v['next_options'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(v['obs'], v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _get_mini_tensors(self, epoch_data):
        num_transitions = len(epoch_data['actions'])
        idxs = np.random.choice(num_transitions, self._trans_minibatch_size)

        data = {}
        for key, value in epoch_data.items():
            data[key] = value[idxs]

        return data

    def _evaluate_policy(self):
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = utils.get_option_colors(random_options * 4)
        random_trajectories = self._get_trajectories(
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with utils.FigManager(self.snapshot_dir, self.step_itr, 'TrajPlot_RandomZ') as fm:
            self._env.render_trajectories( # TODO:
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with utils.FigManager(self.snapshot_dir, self.step_itr, f'PhiPlot') as fm:
            utils.draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            utils.draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            utils.record_video(self.snapshot_dir, self.step_itr, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(self._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True)) # TODO:
        with utils.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            utils.log_performance_ex(
                self.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics()

    def _generate_option_extras(self, options):
        return [{'option': option} for option in options]

    def _get_train_trajectories(self):
        default_kwargs = dict(
            update_stats=True,
            worker_update=dict(
                _render=False,
                _deterministic_policy=False,
            ),
            env_update=dict(_action_noise_std=None),
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs())

        paths = self._get_trajectories(**kwargs)

        return paths

    def _get_trajectories(self,
                          sampler_key,
                          batch_size=None,
                          extras=None,
                          update_stats=False,
                          worker_update=None,
                          env_update=None):
        if batch_size is None:
            batch_size = len(extras)
        policy_sampler_key = sampler_key[6:] if sampler_key.startswith('local_') else sampler_key
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories, infos = obtain_exact_trajectories( # TODO:
                self.step_itr,
                sampler_key=sampler_key,
                batch_size=batch_size,
                agent_update=self._get_policy_param_values(policy_sampler_key),
                env_update=env_update,
                worker_update=worker_update,
                extras=extras,
                update_stats=update_stats,
            )
        print(f'_get_trajectories({sampler_key}) {time_get_trajectories[0]}s')

        for traj in trajectories:
            for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
                if key not in traj['env_infos']:
                    continue

        return trajectories

    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = param_dict[k].detach().cpu()
            else:
                param_dict[k] = param_dict[k].detach()
        return param_dict


    def _log_eval_metrics(self):
        self.eval_log_diagnostics()
        self.plot_log_diagnostics()

    def process_samples(self, paths):
        data = defaultdict(list)
        for path in paths:
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(utils.discount_cumsum(path['rewards'], self.discount))
            data['ori_obs'].append(path['env_infos']['ori_obs'])
            data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])
            if 'pre_tanh_value' in path['agent_infos']:
                data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
            if 'log_prob' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_prob'])
            if 'option' in path['agent_infos']:
                data['options'].append(path['agent_infos']['option'])
                data['next_options'].append(np.concatenate([path['agent_infos']['option'][1:], path['agent_infos']['option'][-1:]], axis=0))

        return data


    def eval_log_diagnostics(self):
        total_time = (time.time() - self._start_time)
        utils.get_tabular('eval').record('TotalEnvSteps', self.total_env_steps)
        utils.get_tabular('eval').record('TotalEpoch', self.total_epoch)
        utils.get_tabular('eval').record('TimeTotal', total_time)
        utils.get_logger('eval').log(utils.get_tabular('eval'))
        utils.get_logger('eval').dump_all(self.step_itr)
        utils.get_tabular('eval').clear()

    def plot_log_diagnostics(self):
        utils.get_tabular('plot').record('TotalEnvSteps', self.total_env_steps)
        utils.get_tabular('plot').record('TotalEpoch', self.total_epoch)
        utils.get_logger('plot').log(utils.get_tabular('plot'))
        utils.get_logger('plot').dump_all(self.step_itr)
        utils.get_tabular('plot').clear()
