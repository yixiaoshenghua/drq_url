import copy
import math
import os
import pickle as pkl
import sys
import time

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
from replay_buffer import ReplayBuffer
from video import VideoRecorder
import drq # Import drq directly for DRQ_DIAYNAgent
import json
from envs import make_env

torch.backends.cudnn.benchmark = True

# TODO:
class Workspace(object):
    """
    Manages the training and evaluation lifecycle for the DrQ(+METRA) agent.
    Handles environment creation, agent instantiation, replay buffer, logging, and video recording.
    Configuration is passed via an argparse Namespace.
    """
    def __init__(self, args):
        self.args = args
        # Create working directory for logs and outputs
        self.work_dir = f'./runs/{args.task}/{time.strftime("%Y%m%d-%H%M%S")}_seed{args.seed}'
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.work_dir + '/models', exist_ok=True)
        print(f'Workspace directory: {self.work_dir}')
        with open(os.path.join(self.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        # Initialize logger (handles console, CSV, TensorBoard)
        self.logger = Logger(self.work_dir,
                             save_tb=args.log_save_tb,
                             log_frequency=args.log_frequency_step,
                             agent=args.agent,
                             action_repeat=args.action_repeat)

        utils.set_seed_everywhere(args.seed) # Set random seeds for reproducibility
        self.device = torch.device(args.device) # Set device (cuda/cpu)
        self.env = make_env(mode="train", config=args) # Create the DMC environment

        # Determine camera_id for VideoRecorder based on environment
        camera_id = 2 if args.task.startswith('quadruped') else 0

        # Prepare parameters for DRQ_DIAYNAgent instantiation from args
        agent_params = {
            'obs_shape': self.env.obs_space["image"].shape,
            'action_shape': self.env.act_space["action"].shape,
            'action_range': [float(self.env.act_space["action"].low.min()), float(self.env.act_space["action"].high.max())],
            'device': self.device,
            'feature_dim': args.encoder_feature_dim,
            'critic_hidden_dim': args.critic_hidden_dim,
            'critic_hidden_depth': args.critic_hidden_depth,
            'actor_hidden_dim': args.actor_hidden_dim,
            'actor_hidden_depth': args.actor_hidden_depth,
            'actor_log_std_min': args.actor_log_std_min,
            'actor_log_std_max': args.actor_log_std_max,
            'num_skills': args.num_skills,
            'skill_embedding_dim': args.skill_embedding_dim,
            'diayn_intrinsic_reward_coeff': args.diayn_intrinsic_reward_coeff,
            'discriminator_hidden_dim': args.discriminator_hidden_dim,
            'discriminator_hidden_depth': args.discriminator_hidden_depth,
            'discount': args.discount,
            'init_temperature': args.init_temperature,
            'lr': args.lr,
            'lr_discriminator': args.lr_discriminator if args.lr_discriminator is not None else args.lr,
            'actor_update_frequency': args.actor_update_frequency,
            'critic_tau': args.critic_tau,
            'critic_target_update_frequency': args.critic_target_update_frequency,
            'batch_size': args.batch_size
        }
        # Instantiate DRQ_DIAYNAgent with parameters derived from args
        self.agent = drq.DRQ_DIAYNAgent(**agent_params)

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.env.obs_space['image'].shape,
                                          self.env.act_space['action'].shape,
                                          args.replay_buffer_capacity,
                                          args.image_pad, self.device)

        # Initialize Video Recorder
        self.video_recorder = VideoRecorder(
            self.work_dir if args.save_video else None, # Only enable if save_video is true
            height=args.render_size,
            width=args.render_size,
            camera_id=camera_id,
            fps=25                  # Standardized FPS for recordings
        )
        self.step = 0 # Initialize global training step counter

        # DIAYN: Store num_skills and initialize current skill for training
        self.num_skills = args.num_skills
        if self.num_skills > 0:
            self.current_skill_train = np.random.randint(0, self.num_skills)
        else:
            self.current_skill_train = 0 # Default skill if DIAYN is not active


    def evaluate(self):
        """
        Performs evaluation episodes for the agent.
        If DIAYN is active (num_skills > 0), evaluates each skill separately.
        Otherwise, performs a standard evaluation.
        Logs rewards and videos (if enabled).
        """
        if self.num_skills == 0: # Standard evaluation (no skills)
            total_episode_reward_sum = 0 # Sum of rewards over all evaluation episodes
            for episode_idx in range(self.args.num_eval_episodes):
                timestep = self.env.reset()
                obs = timestep['image']
                current_eval_skill = 0 # Default skill for non-DIAYN agent.act
                self.video_recorder.init(enabled=(episode_idx == 0 and self.args.save_video), skill_id=current_eval_skill)
                done = False
                episode_extrinsic_reward = 0
                while not done:
                    with utils.eval_mode(self.agent): # Set agent to evaluation mode
                        action = self.agent.act(obs, current_eval_skill, sample=False)
                    timestep = self.env.step({'action': action})
                    next_obs, reward, done = timestep['image'], timestep['reward'], timestep['is_terminal']
                    self.video_recorder.record(obs[:, :, :3], intrinsic_reward=None) # No intrinsic reward overlay
                    episode_extrinsic_reward += reward
                    obs = next_obs
                total_episode_reward_sum += episode_extrinsic_reward

                if episode_idx == 0 and self.args.save_video:
                    self.logger.log_video(f'eval/skill_{current_eval_skill}/video', self.video_recorder.frames, self.step, fps=self.video_recorder.fps)
                    self.video_recorder.save(f'{self.step}_skill_{current_eval_skill}.mp4')
                    

                # Log data for this evaluation episode to CSV
                log_data = {
                    'type': 'eval', 'episode': episode_idx, 'step': self.step, 'duration': -1.0,
                    'episode_extrinsic_reward': episode_extrinsic_reward,
                    'skill_id': current_eval_skill,
                    'episode_intrinsic_reward_sum': 0.0
                }
                self.logger.log_episode_to_csv(log_data)

            average_episode_reward = total_episode_reward_sum / self.args.num_eval_episodes
            self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        else: # DIAYN evaluation: iterate over each skill
            for eval_skill_id in tqdm.tqdm(range(self.num_skills)):
                total_extrinsic_reward_for_skill = 0
                total_intrinsic_reward_sum_for_skill = 0
                for episode_idx in range(self.args.num_eval_episodes):
                    timestep = self.env.reset()
                    obs = timestep['image']
                    self.video_recorder.init(enabled=(episode_idx == 0 and self.args.save_video), skill_id=eval_skill_id)
                    done = False
                    current_episode_extrinsic_reward = 0
                    current_episode_intrinsic_reward_sum = 0

                    while not done:
                        with utils.eval_mode(self.agent):
                            action = self.agent.act(obs, eval_skill_id, sample=False) # Act using the current eval_skill_id
                        timestep = self.env.step({'action': action})
                        next_obs, reward, done = timestep['image'], timestep['reward'], timestep['is_terminal']
                        # Compute and accumulate intrinsic reward for this step
                        current_skill_tensor = torch.tensor([eval_skill_id], device=self.device).long()
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        intrinsic_reward = self.agent.compute_intrinsic_reward(obs_tensor, current_skill_tensor).item()
                        current_episode_intrinsic_reward_sum += intrinsic_reward

                        self.video_recorder.record(obs[:, :, :3], intrinsic_reward=intrinsic_reward) # Record frame with IR
                        current_episode_extrinsic_reward += reward
                        obs = next_obs

                    total_extrinsic_reward_for_skill += current_episode_extrinsic_reward
                    total_intrinsic_reward_sum_for_skill += current_episode_intrinsic_reward_sum

                    if episode_idx == 0 and self.args.save_video:
                         self.logger.log_video(f'eval/skill_{eval_skill_id}/video', self.video_recorder.frames, self.step, fps=self.video_recorder.fps)
                         self.video_recorder.save(f'{self.step}_skill_{eval_skill_id}.mp4') # save step will emptify frames
                         

                    # Log data for this evaluation episode to CSV
                    log_data = {
                        'type': 'eval', 'episode': episode_idx, 'step': self.step, 'duration': -1.0,
                        'episode_extrinsic_reward': current_episode_extrinsic_reward,
                        'skill_id': eval_skill_id,
                        'episode_intrinsic_reward_sum': current_episode_intrinsic_reward_sum
                    }
                    self.logger.log_episode_to_csv(log_data)

                # Log average rewards for this skill to TensorBoard
                average_extrinsic_reward_for_skill = total_extrinsic_reward_for_skill / self.args.num_eval_episodes
                average_intrinsic_reward_sum_for_skill = total_intrinsic_reward_sum_for_skill / self.args.num_eval_episodes
                self.logger.log(f'eval/skill_{eval_skill_id}/episode_reward', average_extrinsic_reward_for_skill, self.step)
                self.logger.log(f'eval/skill_{eval_skill_id}/episode_intrinsic_reward_sum', average_intrinsic_reward_sum_for_skill, self.step)

        self.logger.dump(self.step, ty='eval') # Dump aggregated CSV data (train.csv, eval.csv)
        self.agent.save(self.work_dir + f"/models/model_{self.step}.pt")

    def run(self):
        """Main training loop."""
        episode, episode_step, done = 0, 0, True # Initialize episode counters and done flag

        # Initialize accumulators for the current training episode
        current_episode_extrinsic_reward = 0
        current_episode_intrinsic_reward_sum = 0
        episode_start_time = time.time()

        if self.num_skills > 0: # Log initial skill if DIAYN is active
             self.logger.log('train/current_skill', self.current_skill_train, self.step)

        while self.step < self.args.num_train_steps:
            if done: # Marks the end of an episode and start of a new one
                if self.step > 0: # Don't log metrics for the very first "done" state (step 0)
                    episode_duration = time.time() - episode_start_time
                    # Log TensorBoard metrics for the completed episode
                    self.logger.log('train/duration', episode_duration, self.step)
                    self.logger.log('train/episode_reward', current_episode_extrinsic_reward, self.step) # Extrinsic reward for TB
                    # if self.num_skills > 0:
                    #     self.logger.log(f'train/skill_{self.current_skill_train}/episode_intrinsic_reward_sum',
                    #                     current_episode_intrinsic_reward_sum, self.step)

                    # Log detailed episode data to episodes.csv
                    csv_log_data = {
                        'type': 'train', 'episode': episode, 'step': self.step, 'duration': episode_duration,
                        'episode_extrinsic_reward': current_episode_extrinsic_reward,
                        'skill_id': self.current_skill_train if self.num_skills > 0 else -1,
                        'episode_intrinsic_reward_sum': current_episode_intrinsic_reward_sum if self.num_skills > 0 else 0.0
                    }
                    self.logger.log_episode_to_csv(csv_log_data)

                    # Dump aggregated CSVs (train.csv, eval.csv) and console output
                    self.logger.dump(self.step, save=(self.step > self.args.num_seed_steps), ty='train')

                # Perform periodic evaluation
                if self.step % self.args.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step) # Log current training episode for context
                    self.evaluate()

                # Reset environment and episode-specific counters for the new episode
                timestep = self.env.reset()
                obs = timestep['image']
                done = False
                current_episode_extrinsic_reward = 0
                current_episode_intrinsic_reward_sum = 0
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)
                episode_start_time = time.time() # Reset episode timer

                # DIAYN: Sample a new skill for the new episode
                if self.num_skills > 0:
                    self.current_skill_train = np.random.randint(0, self.num_skills)
                    self.logger.log('train/current_skill', self.current_skill_train, self.step)

            # Store current observation; this will be added to replay buffer as 's_t'
            obs_for_replay_buffer = obs

            # Sample action from agent
            if self.step < self.args.num_seed_steps: # Random actions for seed steps
                action = self.env.act_space['action'].sample()
            else: # Policy actions
                with utils.eval_mode(self.agent): # Use agent in eval mode for action selection
                    action = self.agent.act(obs_for_replay_buffer, self.current_skill_train, sample=True)

            # Perform agent update steps
            if self.step >= self.args.num_seed_steps:
                for _ in range(self.args.num_train_iters): # Typically 1 iter for online RL
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            # Take step in the environment
            timestep = self.env.step({'action': action})
            next_obs, reward, done = timestep['image'], timestep['reward'], timestep['is_terminal']

            current_episode_extrinsic_reward += reward # Accumulate extrinsic reward

            # DIAYN: Compute intrinsic reward for the current step (s_t, z_t)
            if self.num_skills > 0:
                obs_tensor = torch.FloatTensor(obs_for_replay_buffer).unsqueeze(0).to(self.device)
                skill_tensor = torch.tensor([self.current_skill_train], device=self.device).long()
                intrinsic_reward_at_step = self.agent.compute_intrinsic_reward(obs_tensor, skill_tensor).item()
                current_episode_intrinsic_reward_sum += intrinsic_reward_at_step

            # Handle 'done' for replay buffer (distinguish timeout from actual done)
            done_float = float(done)
            # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done_float

            # Add transition to replay buffer
            self.replay_buffer.add(obs_for_replay_buffer, action, reward, next_obs, done_float,
                                   done_float, self.current_skill_train)

            obs = next_obs # Update observation for next step
            episode_step += 1
            self.step += 1

        # save final model
        self.agent.save(self.work_dir + '/models/final_model.pt')

def main():
    parser = argparse.ArgumentParser(description="Train DrQ agent with optional DIAYN skill discovery.")

    parser.add_argument('--agent', type=str, default="drq")
    # Environment arguments
    parser.add_argument('--task', type=str, default='dmc_cartpole_swingup', help='Environment name from dm_control suite.')
    parser.add_argument('--action_repeat', type=int, default=4, help='Number of times an action is repeated.')
    parser.add_argument('--time_limit', type=int, default=1, help="the maximum steps of one episode")
    parser.add_argument('--dmc_camera', type=int, default=-1)
    parser.add_argument('--camera', type=str, default='corner')

    # Training lifecycle arguments
    parser.add_argument('--num_train_steps', type=int, default=1000000, help='Total number of environment steps for training.')
    parser.add_argument('--num_train_iters', type=int, default=1, help='Number of agent updates per environment step after seed steps.')
    parser.add_argument('--num_seed_steps', type=int, default=1000, help='Number of steps with random actions at the beginning.')
    parser.add_argument('--replay_buffer_capacity', type=int, default=100000, help='Capacity of the replay buffer.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    parser.add_argument('--save_model', type=lambda x: (str(x).lower() == 'true'), default=True)

    # Evaluation arguments
    parser.add_argument('--eval_frequency', type=int, default=100000, help='Frequency (in steps) of evaluation phases.')
    parser.add_argument('--num_eval_episodes', type=int, default=1, help='Number of episodes per evaluation phase (and per skill if DIAYN).')

    # Logging and miscellaneous arguments
    parser.add_argument('--log_frequency_step', type=int, default=10000, help='Frequency (in steps) for aggregated CSV/console logs.')
    parser.add_argument('--log_save_tb', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable TensorBoard logging (default: True).')
    parser.add_argument('--save_video', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable saving evaluation videos (default: True).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda", "cpu").')

    # Observation arguments
    parser.add_argument('--render_size', type=int, default=64, help='Size of input images (height and width).')
    parser.add_argument('--image_pad', type=int, default=4, help='Padding for image augmentation.')
    parser.add_argument('--framestack', type=int, default=3, help='Number of consecutive frames to stack as observation.')

    # Core RL algorithm arguments (learning rates, batch size, SAC params)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for actor, critic, and SAC alpha.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor (gamma).')
    parser.add_argument('--init_temperature', type=float, default=0.1, help='Initial value for SAC temperature alpha.')
    parser.add_argument('--actor_update_frequency', type=int, default=2, help='Frequency (in steps) of actor and alpha updates.')
    parser.add_argument('--critic_tau', type=float, default=0.01, help='Soft update coefficient (tau) for target critic.')
    parser.add_argument('--critic_target_update_frequency', type=int, default=2, help='Frequency (in steps) of target critic updates.')

    # Network architecture arguments
    parser.add_argument('--encoder_feature_dim', type=int, default=50, help='Output dimension of the convolutional encoder.')
    parser.add_argument('--critic_hidden_dim', type=int, default=1024, help='Hidden dimension for critic MLP.')
    parser.add_argument('--critic_hidden_depth', type=int, default=2, help='Number of hidden layers for critic MLP.')
    parser.add_argument('--actor_hidden_dim', type=int, default=1024, help='Hidden dimension for actor MLP.')
    parser.add_argument('--actor_hidden_depth', type=int, default=2, help='Number of hidden layers for actor MLP.')
    parser.add_argument('--actor_log_std_min', type=float, default=-10, help='Minimum value for actor policy log_std.')
    parser.add_argument('--actor_log_std_max', type=float, default=2, help='Maximum value for actor policy log_std.')

    # DIAYN-specific arguments
    parser.add_argument('--num_skills', type=int, default=10, help='Number of skills for DIAYN. Set to 0 to disable DIAYN.')
    parser.add_argument('--skill_embedding_dim', type=int, default=10, help='Dimension of skill embedding vector for DIAYN.')
    parser.add_argument('--diayn_intrinsic_reward_coeff', type=float, default=1.0, help='Coefficient for DIAYN intrinsic reward.')
    parser.add_argument('--lr_discriminator', type=float, default=None, help='Learning rate for DIAYN discriminator. Defaults to --lr if not set.')
    parser.add_argument('--discriminator_hidden_dim', type=int, default=1024, help='Hidden dimension for DIAYN discriminator MLP.')
    parser.add_argument('--discriminator_hidden_depth', type=int, default=2, help='Number of hidden layers for DIAYN discriminator MLP.')

    args = parser.parse_args()

    # Correcting boolean arg parsing for store_true like behavior with defaults from YAML
    # The type=lambda x: (str(x).lower() == 'true') handles this for default=True cases.
    # If a flag like --no-save-video is preferred for a default True, then action='store_false' and dest='save_video' would be used.

    workspace = Workspace(args)
    workspace.run()


if __name__ == '__main__':
    main()
