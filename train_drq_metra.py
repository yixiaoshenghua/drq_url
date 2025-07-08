import copy
import math
import os
import pickle as pkl
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message='ing')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import argparse

# import dmc2gym
# import hydra # Hydra configuration is replaced by argparse
import tqdm
import utils
from logger import Logger # Handles logging to console, CSVs, and TensorBoard
from replay_buffer import PathBufferEx
from video import VideoRecorder
import metra # Import drq directly for DRQ_METRAAgent
import json
from envs import make_env


torch.backends.cudnn.benchmark = True

'''
# METRA on pixel-based car racing for debug
python train_drq_metra.py --task debug_dummy --time_limit 50 --seed 0 --traj_batch_size 8 --video_skip_frames 2 --framestack 3 --sac_min_buffer_size 300 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 2 --n_epochs_per_log 5 --n_epochs_per_eval 5 --n_epochs_per_save 1 --n_epochs_per_pt_save 1 --discrete 0 --dim_skill 4 --encoder 1 --sample_cpu 0 --action_repeat 1 --n_epochs 10
'''

class Workspace(object):
    """
    Manages the training and evaluation lifecycle for the DrQ(+METRA) agent.
    Handles environment creation, agent instantiation, replay buffer, logging, and video recording.
    Configuration is passed via an argparse Namespace.
    """
    def __init__(self, args):
        self.args = args
        # Create working directory for logs and outputs
        self.work_dir = f'./runs/{args.algo}-{args.task}/{time.strftime("%Y%m%d-%H%M%S")}_seed{args.seed}'
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.work_dir + '/models', exist_ok=True)
        print(f'Workspace directory: {self.work_dir}')
        with open(os.path.join(self.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        utils.set_seed_everywhere(args.seed) # Set random seeds for reproducibility
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device (cuda/cpu)
        self.env = make_env(mode="train", config=args) # Create the DMC environment

        # Determine camera_id for VideoRecorder based on environment
        camera_id = 2 if args.task.startswith('quadruped') else 0

        # Initialize Replay Buffer
        self.replay_buffer = PathBufferEx(capacity_in_transitions=int(args.sac_max_buffer_size), 
                                          pixel_shape=self.env.obs_space['image'].shape)
        # Prepare parameters for DRQ_METRAAgent instantiation from args
        agent_params = dict(
                env=self.env,
                tau=args.sac_tau,
                scale_reward=args.sac_scale_reward,
                target_coef=args.sac_target_coef,
                replay_buffer=self.replay_buffer,
                min_buffer_size=args.sac_min_buffer_size,
                inner=args.inner,
                num_alt_samples=args.num_alt_samples,
                split_group=args.split_group,
                dual_reg=args.dual_reg,
                dual_slack=args.dual_slack,
                dual_dist=args.dual_dist,
                pixel_shape=self.env.spec.observation_space.shape,
                env_name=args.task,
                algo=args.algo,
                env_spec=self.env.spec,
                skill_dynamics=None,
                dist_predictor=None,
                dual_lam=args.dual_lam,
                alpha=args.alpha,
                time_limit=args.time_limit,
                n_epochs_per_eval=args.n_epochs_per_eval,
                n_epochs_per_log=args.n_epochs_per_log,
                n_epochs_per_tb=args.n_epochs_per_log,
                n_epochs_per_save=args.n_epochs_per_save,
                n_epochs_per_pt_save=args.n_epochs_per_pt_save,
                n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
                dim_skill=args.dim_skill,
                num_random_trajectories=args.num_random_trajectories,
                num_video_repeats=args.num_video_repeats,
                eval_record_video=args.eval_record_video,
                video_skip_frames=args.video_skip_frames,
                eval_plot_axis=args.eval_plot_axis,
                name='METRA',
                device=self.device,
                sample_cpu=args.sample_cpu,
                num_train_per_epoch=1,
                sd_batch_norm=args.sd_batch_norm, # True, no use
                skill_dynamics_obs_dim=self.env.spec.observation_space.flat_dim, # no use
                trans_minibatch_size=args.trans_minibatch_size,
                trans_optimization_epochs=args.trans_optimization_epochs,
                discount=args.sac_discount,
                discrete=args.discrete,
                unit_length=args.unit_length,
                batch_size=args.traj_batch_size,
                snapshot_dir=self.work_dir,
                use_encoder=True,
                spectral_normalization=args.spectral_normalization,
                model_master_nonlinearity=args.model_master_nonlinearity,
                model_master_dim=args.model_master_dim,
                model_master_num_layers=args.model_master_num_layers,
                lr_op=args.lr_op,
                lr_te=args.lr_te,
                dual_lr=args.dual_lr,
                sac_lr_q=args.sac_lr_q,
                sac_lr_a=args.sac_lr_a,
                seed=args.seed,
            )
        # Instantiate DRQ_METRAAgent with parameters derived from args
        self.agent = metra.DRQ_METRAAgent(**agent_params)
        self.agent.setup_logger(self.work_dir)  # Setup logging for the agent

    def run(self):
        """Main training loop."""
        self.agent.train(n_epochs=self.args.n_epochs)  # Start training the agent

        # TODO: save final model
        self.agent.save(self.work_dir + '/models/final_model.pt')

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument('--normalizer_type', type=str, default='off', choices=['off', 'preset'])
    parser.add_argument('--encoder', type=int, default=1)
    parser.add_argument('--task', type=str, default='dmc_walker_walk')
    parser.add_argument('--framestack', type=int, default=None)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--render_size', type=int, default=64)
    parser.add_argument('--flatten_obs', type=int, default=1, choices=[0, 1])

    parser.add_argument('--time_limit', type=int, default=200)

    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=4)
    parser.add_argument('--n_thread', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--traj_batch_size', type=int, default=8)
    parser.add_argument('--trans_minibatch_size', type=int, default=256)
    parser.add_argument('--trans_optimization_epochs', type=int, default=200)

    parser.add_argument('--n_epochs_per_eval', type=int, default=125)
    parser.add_argument('--n_epochs_per_log', type=int, default=25)
    parser.add_argument('--n_epochs_per_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    parser.add_argument('--num_random_trajectories', type=int, default=48)
    parser.add_argument('--num_video_repeats', type=int, default=2)
    parser.add_argument('--eval_record_video', type=int, default=1)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)

    parser.add_argument('--dim_skill', type=int, default=2)

    parser.add_argument('--common_lr', type=float, default=1e-4)
    parser.add_argument('--lr_op', type=float, default=None)
    parser.add_argument('--lr_te', type=float, default=None)

    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--algo', type=str, default='metra', choices=['metra', 'dads'])

    parser.add_argument('--sac_tau', type=float, default=5e-3)
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_min_buffer_size', type=int, default=10000)
    parser.add_argument('--sac_max_buffer_size', type=int, default=300000)

    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])

    parser.add_argument('--model_master_dim', type=int, default=1024)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    parser.add_argument('--sd_const_std', type=int, default=1)
    parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])

    parser.add_argument('--num_alt_samples', type=int, default=100)
    parser.add_argument('--split_group', type=int, default=65536)

    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1])
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1])
    parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1])  # Only for continuous skills

    parser.add_argument('--dual_reg', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=30)
    parser.add_argument('--dual_slack', type=float, default=1e-3)
    parser.add_argument('--dual_dist', type=str, default='one', choices=['l2', 's2_from_s', 'one'])
    parser.add_argument('--dual_lr', type=float, default=None)

    return parser


def main():
    args = get_argparser().parse_args()

    # Correcting boolean arg parsing for store_true like behavior with defaults from YAML
    # The type=lambda x: (str(x).lower() == 'true') handles this for default=True cases.
    # If a flag like --no-save-video is preferred for a default True, then action='store_false' and dest='save_video' would be used.

    workspace = Workspace(args)
    workspace.run()


if __name__ == '__main__':
    main()
