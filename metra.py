import numpy as np
import os
from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time
import utils
from collections import defaultdict
import tqdm
import functools
from torch.utils.tensorboard import SummaryWriter
import logging
from replay_buffer import TrajectoryBatch, SkillRolloutWorker
import sac_utils
from networks import PolicyEx, ContinuousMLPQFunctionEx, GaussianMLPIndependentStdModuleEx, GaussianMLPTwoHeadedModuleEx, Encoder, WithEncoder

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

def _finalize_lr(lr, common_lr=1e-4):
    if lr is None:
        lr = common_lr
    else:
        assert bool(lr), 'To specify a lr of 0, use a negative value'
    if lr < 0.0:
        print(f'Setting lr to ZERO given {lr}')
        lr = 0.0
    return lr


class DRQ_METRAAgent:
    """
    DrQ agent with METRA skill discovery integration.
    Manages the actor, critic, and their updates.
    Configuration is now passed via explicit arguments (formerly Hydra).
    """
    def __init__(self,
            env,
            tau,
            scale_reward,
            target_coef,
            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,
            dual_lam,
            dual_reg,
            dual_slack,
            dual_dist,
            pixel_shape,
            env_name,
            algo,
            env_spec,
            skill_dynamics,
            dist_predictor,
            alpha,
            time_limit,
            n_epochs_per_eval,
            n_epochs_per_log,
            n_epochs_per_tb,
            n_epochs_per_save,
            n_epochs_per_pt_save,
            n_epochs_per_pkl_update,
            dim_skill,
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
            use_encoder=True,
            spectral_normalization=False,
            model_master_nonlinearity=None,
            model_master_dim=1024,
            model_master_num_layers=2,
            lr_op=None,
            lr_te=None,
            dual_lr=None,
            sac_lr_q=None,
            sac_lr_a=None,
            seed=0,
    ):
        self._env = env
        self.env_name = env_name
        self.algo = algo
        self.seed = seed
        self.enable_logging = True

        self.step_itr = 0
        self.snapshot_dir = snapshot_dir

        self.discount = discount
        self.time_limit = time_limit

        self.device = device
        self.sample_cpu = sample_cpu
        # skill_policy
        if model_master_nonlinearity == 'relu':
            nonlinearity = torch.relu
        elif model_master_nonlinearity == 'tanh':
            nonlinearity = torch.tanh
        else:
            nonlinearity = None
        self.use_encoder = use_encoder
        example_ob = env.reset()
        if self.use_encoder: # for pixels input
            def make_encoder(**kwargs):
                return Encoder(pixel_shape=pixel_shape, **kwargs)

            def with_encoder(module, encoder=None):
                if encoder is None:
                    encoder = make_encoder()

                return WithEncoder(encoder=encoder, module=module)

            example_encoder = make_encoder()
            module_obs_dim = example_encoder(torch.as_tensor(example_ob["image"]).float().unsqueeze(0)).shape[-1]
        policy_q_input_dim = module_obs_dim + dim_skill
        action_dim = self._env.spec.action_space.flat_dim
        master_dims = [model_master_dim] * model_master_num_layers
        self.skill_policy = PolicyEx(
            name='skill_policy',
            env_spec=env_spec,
            module=with_encoder(GaussianMLPTwoHeadedModuleEx(
                input_dim=policy_q_input_dim,
                output_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity,
                layer_normalization=False,
                max_std=np.exp(2.),
                normal_distribution_cls=utils.TanhNormal,
                output_w_init=functools.partial(utils.xavier_normal_ex, gain=1.),
                init_std=1.,
            )) if self.use_encoder else GaussianMLPTwoHeadedModuleEx(
                input_dim=policy_q_input_dim,
                output_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity,
                layer_normalization=False,
                max_std=np.exp(2.),
                normal_distribution_cls=utils.TanhNormal,
                output_w_init=functools.partial(utils.xavier_normal_ex, gain=1.),
                init_std=1.,
            ),
            skill_info={'dim_skill': dim_skill}
        ).to(self.device)
        # traj_encoder
        self.spectral_normalization = spectral_normalization
        traj_encoder_obs_dim = module_obs_dim
        traj_encoder = GaussianMLPIndependentStdModuleEx(
            input_dim=traj_encoder_obs_dim,
            output_dim=dim_skill,
            std_hidden_sizes=master_dims,
            std_hidden_nonlinearity=nonlinearity or torch.relu,
            std_hidden_w_init=torch.nn.init.xavier_uniform_,
            std_output_w_init=torch.nn.init.xavier_uniform_,
            init_std=1.0,
            min_std=1e-6,
            max_std=None,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            hidden_w_init=torch.nn.init.xavier_uniform_,
            output_w_init=torch.nn.init.xavier_uniform_,
            std_parameterization='exp',
            bias=True,
            spectral_normalization=self.spectral_normalization,
        )
        if self.use_encoder:
            if self.spectral_normalization:
                te_encoder = make_encoder(spectral_normalization=True)
            else:
                te_encoder = None
            traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)
        self.traj_encoder = traj_encoder.to(self.device)
        # dual_lambda
        self.dual_lam = utils.ParameterModule(torch.Tensor([np.log(dual_lam)])).to(self.device)
        
        self.param_modules = {
            'traj_encoder': self.traj_encoder,
            'skill_policy': self.skill_policy,
            'dual_lam': self.dual_lam,
        }
        if skill_dynamics is not None:
            self.skill_dynamics = skill_dynamics.to(self.device)
            self.param_modules['skill_dynamics'] = self.skill_dynamics
        if dist_predictor is not None:
            self.dist_predictor = dist_predictor.to(self.device)
            self.param_modules['dist_predictor'] = self.dist_predictor

        optimizers = {
            'skill_policy': torch.optim.Adam([
                {'params': self.skill_policy.parameters(), 'lr': _finalize_lr(lr_op)},
            ]),
            'traj_encoder': torch.optim.Adam([
                {'params': self.traj_encoder.parameters(), 'lr': _finalize_lr(lr_te)},
            ]),
            'dual_lam': torch.optim.Adam([
                {'params': self.dual_lam.parameters(), 'lr': _finalize_lr(dual_lr)},
            ]),
        }
        if skill_dynamics is not None:
            optimizers.update({
                'skill_dynamics': torch.optim.Adam([
                    {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(lr_te)},
                ]),
            })
        if dist_predictor is not None:
            optimizers.update({
                'dist_predictor': torch.optim.Adam([
                    {'params': dist_predictor.parameters(), 'lr': _finalize_lr(lr_op)},
                ]),
            })

        self.alpha = alpha
        self.name = name

        self.dim_skill = dim_skill

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

        # self._sd_batch_norm = sd_batch_norm
        # self._skill_dynamics_obs_dim = skill_dynamics_obs_dim

        # if self._sd_batch_norm:
        #     self._sd_input_batch_norm = torch.nn.BatchNorm1d(self._skill_dynamics_obs_dim, momentum=0.01).to(self.device)
        #     self._sd_target_batch_norm = torch.nn.BatchNorm1d(self._skill_dynamics_obs_dim, momentum=0.01, affine=False).to(self.device)
        #     self._sd_input_batch_norm.eval()
        #     self._sd_target_batch_norm.eval()

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self.discrete = discrete
        self.unit_length = unit_length
        self.batch_size = batch_size

        self.traj_encoder.eval()

        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            )
        if self.use_encoder:
            qf1 = with_encoder(qf1)
        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if self.use_encoder:
            qf2 = with_encoder(qf2)
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        log_alpha = utils.ParameterModule(torch.Tensor([np.log(self.alpha)]))
        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(sac_lr_a)},
            ])
        })

        self._optimizer = OptimizerGroupWrapper(
            optimizers=optimizers,
            max_optimization_epochs=None,
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
        self.total_epoch = 0

        self.rollout_worker = SkillRolloutWorker(self.seed, time_limit=self.time_limit, cur_extra_keys=['skill'])

        self.writer = None
        self.logger = None

    @property
    def policy(self):
        return {
            'skill_policy': self.skill_policy,
        }

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p


    #########################################################################################################
    #                                                                                                       #
    #                                        interacting with env                                           #
    #                                                                                                       #
    #########################################################################################################
    def _generate_skill_extras(self, skills):
        return [{'skill': skill} for skill in skills]

    def _get_train_trajectories_kwargs(self):
        if self.discrete: # False
            extras = self._generate_skill_extras(np.eye(self.dim_skill)[np.random.randint(0, self.dim_skill, self.batch_size)])
        else:
            random_skills = np.random.randn(self.batch_size, self.dim_skill)
            if self.unit_length:
                random_skills /= np.linalg.norm(random_skills, axis=-1, keepdims=True)
            extras = self._generate_skill_extras(random_skills)

        return dict(
            extras=extras,
        )

    def _get_train_trajectories(self):
        default_kwargs = dict(
            batch_size=self.batch_size,
            deterministic_policy=False,
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs())

        paths = self._get_trajectories(**kwargs)

        return paths

    def _get_trajectories(self,
                          batch_size=None,
                          deterministic_policy=False,
                          extras=None):
        if batch_size is None:
            batch_size = len(extras)
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories = self.obtain_exact_trajectories( 
                env=self._env,
                policy=self.skill_policy,
                batch_size=batch_size,
                extras=extras,
                deterministic_policy=deterministic_policy,
            )
        print(f'_get_trajectories {time_get_trajectories[0]}s')

        # for traj in trajectories:
        #     for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
        #         if key not in traj['env_infos']:
        #             continue

        return trajectories

    def obtain_exact_trajectories(self, env, policy, batch_size, extras, deterministic_policy=False): 
        batches = []
        for i in range(batch_size):
            extra = extras[i]
            batch = self.rollout_worker.rollout(env, policy, extra, deterministic_policy=deterministic_policy)
            batches.append(batch)
        trajectories = TrajectoryBatch.concatenate(*batches)
        paths = trajectories.to_trajectory_list()
        return paths

    #########################################################################################################
    #                                                                                                       #
    #                                        processing data                                                #
    #                                                                                                       #
    #########################################################################################################
    def _get_mini_tensors(self, epoch_data):
        num_transitions = len(epoch_data['actions'])
        idxs = np.random.choice(num_transitions, self._trans_minibatch_size)

        data = {}
        for key, value in epoch_data.items():
            data[key] = value[idxs]

        return data

    def _get_concat_obs(self, obs, skill):
        return utils.get_torch_concat_obs(obs, skill)

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = param_dict[k].detach().cpu()
            else:
                param_dict[k] = param_dict[k].detach()
        return param_dict

    def process_samples(self, paths):
        data = defaultdict(list)
        for path in paths:
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(utils.discount_cumsum(path['rewards'], self.discount))
            # if 'ori_obs' in path['env_infos']:
            #     data['ori_obs'].append(path['env_infos']['ori_obs'])
            # if 'next_ori_obs' in path['env_infos']:
            #     data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])
            if 'pre_tanh_value' in path['agent_infos']:
                data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
            if 'log_prob' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_prob'])
            if 'skill' in path['agent_infos']:
                data['skills'].append(path['agent_infos']['skill'])
                data['next_skills'].append(np.concatenate([path['agent_infos']['skill'][1:], path['agent_infos']['skill'][-1:]], axis=0))

        return data

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
            if value.shape[1] == 1 and 'skill' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data
    
    #########################################################################################################
    #                                                                                                       #
    #                                               training                                                #
    #                                                                                                       #
    #########################################################################################################
    def train(self, n_epochs):
        last_return = None
        with utils.GlobalContext({'phase': 'train', 'policy': 'sampling'}):
            self.logger.info('Obtaining samples...')
            for epoch in range(n_epochs):
                self.logger.info('epoch #%d | ' % epoch)
                self._itr_start_time = time.time()
                self.total_epoch = epoch

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
                        step_paths = self._get_train_trajectories()
                    self.total_env_steps += sum([step_path['dones'].shape[0] for step_path in step_paths])
                    last_return = self.train_once(
                        self.step_itr,
                        step_paths,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )

                self.step_itr += 1

                new_save = (self.n_epochs_per_save != 0 and self.step_itr % self.n_epochs_per_save == 0)
                pt_save = (self.n_epochs_per_pt_save != 0 and self.step_itr % self.n_epochs_per_pt_save == 0)
                if new_save or pt_save:
                    self.save(epoch, new_save=new_save, pt_save=pt_save)

                if self.enable_logging:
                    if self.step_itr % self.n_epochs_per_log == 0:
                        self.log_diagnostics(pause_for_plot=False)
                        if self.n_epochs_per_tb is None:
                            self.writer.flush()
                        else:
                            if self.step_itr <= 0 or (self.n_epochs_per_tb != 0 and self.step_itr % self.n_epochs_per_tb == 0):
                                self.writer.flush()
                            else:
                                print('Dump text csv std at', self.step_itr)

        return last_return

    def train_once(self, itr, paths, extra_scalar_metrics={}):
        logging_enabled = ((self.step_itr + 1) % self.n_epochs_per_log == 0)

        data = self.process_samples(paths)

        time_computing_metrics = [0.0]
        time_training = [0.0]

        with MeasureAndAccTime(time_training):
            metrics = self._train_once_inner(data)

        performance = utils.log_performance_ex(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )
        discounted_returns = performance['discounted_returns']
        undiscounted_returns = performance['undiscounted_returns']

        prefix = utils.get_metric_prefix() + self.name + '/'
        self.writer.add_scalar(prefix + 'AverageDiscountedReturn', np.mean(discounted_returns), self.step_itr)
        self.writer.add_scalar(prefix + 'AverageReturn', np.mean(undiscounted_returns), self.step_itr)

        if logging_enabled:
            for k in metrics.keys():
                if metrics[k].numel() == 1:
                    self.writer.add_scalar(prefix + f'{k}', metrics[k].item(), self.step_itr)
                else:
                    self.writer.add_scalar(prefix + f'{k}', metrics[k].mean(), self.step_itr)  # Use mean for arrays
            with torch.no_grad():
                total_norm = compute_total_norm(self.all_parameters())
                self.writer.add_scalar(prefix + 'TotalGradNormAll', total_norm.item(), self.step_itr)
                for key, module in self.param_modules.items():
                    total_norm = compute_total_norm(module.parameters())
                    self.writer.add_scalar(prefix + f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}', total_norm.item(), self.step_itr)
            for k, v in extra_scalar_metrics.items():
                self.writer.add_scalar(prefix + k, v, self.step_itr)
            self.writer.add_scalar(prefix + 'TimeComputingMetrics', time_computing_metrics[0], self.step_itr)
            self.writer.add_scalar(prefix + 'TimeTraining', time_training[0], self.step_itr)

            path_lengths = [
                len(path['actions'])
                for path in paths
            ]
            self.writer.add_scalar(prefix + 'PathLengthMean', np.mean(path_lengths), self.step_itr)
            self.writer.add_scalar(prefix + 'PathLengthMax', np.max(path_lengths), self.step_itr)
            self.writer.add_scalar(prefix + 'PathLengthMin', np.min(path_lengths), self.step_itr)

            self.writer.add_histogram(prefix + 'ExternalDiscountedReturns', np.asarray(discounted_returns), self.step_itr)
            self.writer.add_histogram(prefix + 'ExternalUndiscountedReturns', np.asarray(undiscounted_returns), self.step_itr)

        return np.mean(undiscounted_returns)

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        metrics = self._train_components(epoch_data)

        return metrics

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            print(f"Current buffer size: {self.replay_buffer.n_transitions_stored}")
            return {}

        for _ in range(self._trans_optimization_epochs):
            metrics = {}

            if self.replay_buffer is None: # on policy training
                v = self._get_mini_tensors(epoch_data)
            else: # off policy training
                v = self._sample_replay_buffer()

            self._optimize_te(metrics, v)
            self._update_rewards(metrics, v)
            self._optimize_op(metrics, v)

        return metrics
    
    def _gradient_descent(self, loss, optimizer_keys):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._optimizer.step(keys=optimizer_keys)

    def _optimize_te(self, metrics, internal_vars):
        self._update_loss_te(metrics, internal_vars)

        self._gradient_descent(
            metrics['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            self._update_loss_dual_lam(metrics, internal_vars)
            self._gradient_descent(
                metrics['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    metrics['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    def _optimize_op(self, metrics, internal_vars):
        self._update_loss_qf(metrics, internal_vars)

        self._gradient_descent(
            metrics['LossQf1'] + metrics['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_loss_op(metrics, internal_vars)
        self._gradient_descent(
            metrics['LossSacp'],
            optimizer_keys=['skill_policy'],
        )

        self._update_loss_alpha(metrics, internal_vars)
        self._gradient_descent(
            metrics['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    def _update_rewards(self, metrics, v):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            if self.discrete:
                masks = (v['skills'] - v['skills'].mean(dim=1, keepdim=True)) * self.dim_skill / (self.dim_skill - 1 if self.dim_skill != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['skills']).sum(dim=1)
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
                rewards = -torch.nn.functional.cross_entropy(logits, v['skills'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['skills'])

        metrics.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards

    def _update_loss_te(self, metrics, v):
        self._update_rewards(metrics, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            metrics.update({
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

                metrics.update({
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
            metrics.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        metrics.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, metrics, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        metrics.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, metrics, v):
        processed_cat_obs = self._get_concat_obs(self.skill_policy.process_observations(v['obs']), v['skills'])
        next_processed_cat_obs = self._get_concat_obs(self.skill_policy.process_observations(v['next_obs']), v['next_skills'])

        sac_utils.update_loss_qf(
            self, metrics, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.skill_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, metrics, v):
        processed_cat_obs = self._get_concat_obs(self.skill_policy.process_observations(v['obs']), v['skills'])
        sac_utils.update_loss_sacp(
            self, metrics, v,
            obs=processed_cat_obs,
            policy=self.skill_policy,
        )

    def _update_loss_alpha(self, metrics, v):
        sac_utils.update_loss_alpha(
            self, metrics, v,
        )


    #########################################################################################################
    #                                                                                                       #
    #                                        logging and eval                                               #
    #                                                                                                       #
    #########################################################################################################
    def setup_logger(self, log_dir):
        tabular_log_file = os.path.join(log_dir, 'progress.csv')
        text_log_file = os.path.join(log_dir, 'debug.log')
        tb_dir = os.path.join(log_dir, 'tb')

        self.writer = SummaryWriter(tb_dir)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DRQ_METRAAgent')
        handler = logging.FileHandler(text_log_file)
        self.logger.addHandler(handler)

        print('Logging to {}'.format(log_dir))

    def _evaluate_policy(self):
        if self.discrete:
            eye_skills = np.eye(self.dim_skill)
            random_skills = []
            colors = []
            for i in range(self.dim_skill):
                num_trajs_per_skill = self.num_random_trajectories // self.dim_skill + (i < self.num_random_trajectories % self.dim_skill)
                for _ in range(num_trajs_per_skill):
                    random_skills.append(eye_skills[i])
                    colors.append(i)
            random_skills = np.array(random_skills)
            colors = np.array(colors)
            num_evals = len(random_skills)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_skill <= 10 else 'tab20'
            random_skill_colors = []
            for i in range(num_evals):
                random_skill_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_skill_colors = np.array(random_skill_colors)
        else:
            random_skills = np.random.randn(self.num_random_trajectories, self.dim_skill)
            if self.unit_length:
                random_skills = random_skills / np.linalg.norm(random_skills, axis=1, keepdims=True)
            random_skill_colors = utils.get_skill_colors(random_skills * 4)
        random_trajectories = self._get_trajectories(
            batch_size=self.num_random_trajectories,
            extras=self._generate_skill_extras(random_skills),
            deterministic_policy=True,
        )

        if False: # TODO:
            with utils.FigManager(self.snapshot_dir, self.step_itr, 'TrajPlot_RandomZ', writer=self.writer, global_step=self.step_itr) as fm:
                self._env.render_trajectories(
                    random_trajectories, random_skill_colors, self.eval_plot_axis, fm.ax
                )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        skill_dists = self.traj_encoder(last_obs)

        skill_means = skill_dists.mean.detach().cpu().numpy()
        if self.inner:
            skill_stddevs = torch.ones_like(skill_dists.stddev.detach().cpu()).numpy()
        else:
            skill_stddevs = skill_dists.stddev.detach().cpu().numpy()
        skill_samples = skill_dists.mean.detach().cpu().numpy()

        skill_colors = random_skill_colors

        with utils.FigManager(self.snapshot_dir, self.step_itr, f'PhiPlot', writer=self.writer, global_step=self.step_itr) as fm: # PhiPlot just plots ϕ(s). The phi trajectories in the paper are also ϕ(s) trajectories from randomly sampled z's.
            utils.draw_2d_gaussians(skill_means, skill_stddevs, skill_colors, fm.ax)
            utils.draw_2d_gaussians(
                skill_samples,
                [[0.03, 0.03]] * len(skill_samples),
                skill_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_skill_metrics = {}

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_skills = np.eye(self.dim_skill)
                video_skills = video_skills.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_skill == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_skills = []
                    for angle in [3, 2, 1, 4]:
                        video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_skills.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_skills = np.array(video_skills)
                else:
                    video_skills = np.random.randn(9, self.dim_skill)
                    if self.unit_length:
                        video_skills = video_skills / np.linalg.norm(video_skills, axis=1, keepdims=True)
                video_skills = video_skills.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                batch_size=len(video_skills),
                deterministic_policy=True,
                extras=self._generate_skill_extras(video_skills),
            )
            utils.record_video(self.snapshot_dir, self.step_itr, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_skill_metrics.update(self.calc_eval_metrics(random_trajectories, is_skill_trajectories=True))
        with utils.GlobalContext({'phase': 'eval', 'policy': 'skill'}):
            performance = utils.log_performance_ex(
                self.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_skill_metrics,
            )
            # Log performance metrics with 'eval/' prefix
            for k, v in performance['scalars'].items():
                self.writer.add_scalar('eval/' + k, v, self.step_itr)
            for k, v in performance['histograms'].items():
                self.writer.add_histogram('eval/' + k, v, self.step_itr)
        self._log_eval_metrics()

    def calc_eval_metrics(self, trajectories, is_skill_trajectories=True):
        eval_metrics = {}
        sum_returns = 0
        for traj in trajectories:
            sum_returns += traj['rewards'].sum()
        eval_metrics[f'ReturnOverall'] = sum_returns

        return eval_metrics
    
    def log_diagnostics(self, pause_for_plot=False):
        total_time = (time.time() - self._start_time)
        self.logger.info('Time %.2f s' % total_time)
        epoch_time = (time.time() - self._itr_start_time)
        self.logger.info('EpochTime %.2f s' % epoch_time)
        self.writer.add_scalar('TotalEnvSteps', self.total_env_steps, self.total_epoch)
        self.writer.add_scalar('TotalEpoch', self.total_epoch, self.total_epoch)
        self.writer.add_scalar('TimeEpoch', epoch_time, self.total_epoch)
        self.writer.add_scalar('TimeTotal', total_time, self.total_epoch)
        self.writer.flush()

    def _log_eval_metrics(self):
        self.eval_log_diagnostics()
        self.plot_log_diagnostics()

    def eval_log_diagnostics(self):
        total_time = (time.time() - self._start_time)
        self.writer.add_scalar('eval/TotalEnvSteps', self.total_env_steps, self.step_itr)
        self.writer.add_scalar('eval/TotalEpoch', self.total_epoch, self.step_itr)
        self.writer.add_scalar('eval/TimeTotal', total_time, self.step_itr)
        self.writer.flush()

    def plot_log_diagnostics(self):
        self.writer.add_scalar('plot/TotalEnvSteps', self.total_env_steps, self.step_itr)
        self.writer.add_scalar('plot/TotalEpoch', self.total_epoch, self.step_itr)
        self.writer.flush()

    #########################################################################################################
    #                                                                                                       #
    #                                        save and restore model                                         #
    #                                                                                                       #
    #########################################################################################################
    def save(self, epoch, new_save=False, pt_save=False):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the runner is set up.

        """

        self.logger.info('Saving snapshot...')

        if new_save and epoch != 0:
            os.makedirs(os.path.join(self.snapshot_dir, f'models/epoch-{epoch}'), exist_ok=True)
            file_name = os.path.join(self.snapshot_dir, f'models/epoch-{epoch}/skill_policy.pt')
            torch.save({
                'discrete': self.discrete,
                'dim_skill': self.dim_skill,
                'policy': self.skill_policy,
            }, file_name)
            file_name = os.path.join(self.snapshot_dir, f'models/epoch-{epoch}/traj_encoder.pt')
            torch.save({
                'discrete': self.discrete,
                'dim_skill': self.dim_skill,
                'traj_encoder': self.traj_encoder,
            }, file_name)

        if pt_save and epoch != 0:
            os.makedirs(os.path.join(self.snapshot_dir, f'models/epoch-{epoch}'), exist_ok=True)
            file_name = os.path.join(self.snapshot_dir, f'models/epoch-{epoch}/skill_policy.pt')
            torch.save({
                'discrete': self.discrete,
                'dim_skill': self.dim_skill,
                'policy': self.skill_policy,
            }, file_name)

        self.logger.info('Saved')