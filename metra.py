import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils

class Encoder(nn.Module):
    """
    Convolutional encoder for image-based observations.
    Takes an observation and outputs a flat feature vector.
    """
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 25
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[-1], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * self.output_dim * self.output_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        if obs.shape[-1] == 3 or obs.shape[-1] == 9:
            obs = obs.permute(0, 3, 1, 2)
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.reshape((conv.size(0), -1))
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """
    Actor network (policy) for DrQ. Outputs a distribution over actions.
    Can be skill-conditioned for DIAYN.
    """
    def __init__(self, obs_shape, feature_dim, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, num_skills, skill_embedding_dim):
        """
        Args:
            obs_shape (tuple): Shape of the observation space.
            feature_dim (int): Dimension of the features from the encoder.
            action_shape (tuple): Shape of the action space.
            hidden_dim (int): Dimension of hidden layers in the MLP.
            hidden_depth (int): Number of hidden layers in the MLP.
            log_std_bounds (list/tuple): Min and max values for log_std.
            num_skills (int): Number of skills for DIAYN. If 0, DIAYN is not used.
            skill_embedding_dim (int): Dimension for skill embeddings if num_skills > 0.
        """
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)
        self.num_skills = num_skills
        self.skill_embedding_dim = skill_embedding_dim

        # Determine input dimension for the policy's MLP trunk
        trunk_input_dim = self.encoder.feature_dim
        if self.num_skills > 0:
            self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim)
            trunk_input_dim += skill_embedding_dim

        self.log_std_bounds = log_std_bounds
        # MLP trunk outputs parameters for the action distribution (mean and log_std)
        self.trunk = utils.mlp(trunk_input_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict() # For logging intermediate activations
        self.apply(utils.weight_init) # Apply weight initialization

    def forward(self, obs, skill=None, detach_encoder=False):
        """
        Forward pass of the actor.
        Args:
            obs (torch.Tensor): Batch of observations.
            skill (torch.Tensor, optional): Batch of skills, if DIAYN is used.
            detach_encoder (bool): Whether to detach the encoder features from the graph.
        Returns:
            SquashedNormal distribution over actions.
        """
        obs_features = self.encoder(obs, detach=detach_encoder)

        # If skill-conditioned, concatenate skill embedding with observation features
        if self.num_skills > 0:
            if skill is None:
                raise ValueError("Skill must be provided when num_skills > 0 for DIAYN Actor.")
            # skill shape expected: (batch_size, 1) or (batch_size,)
            skill_emb = self.skill_embedding(skill.long().squeeze(-1)) # Squeeze to handle (B,1) -> (B)
            combined_features = torch.cat([obs_features, skill_emb], dim=-1)
        else:
            combined_features = obs_features

        mu, log_std = self.trunk(combined_features).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """
    Critic network for DrQ. Implements double Q-learning (two Q-functions).
    It takes observations and actions to predict Q-values.
    """
    def __init__(self, obs_shape, feature_dim, action_shape, hidden_dim, hidden_depth):
        """
        Args:
            obs_shape (tuple): Shape of the observation space.
            feature_dim (int): Dimension of the features from the encoder.
            action_shape (tuple): Shape of the action space.
            hidden_dim (int): Dimension of hidden layers in the MLP.
            hidden_depth (int): Number of hidden layers in the MLP.
        """
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        # MLP for the first Q-function
        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0], # Input: encoded_obs + action
                            hidden_dim, 1, hidden_depth) # Output: Q-value
        # MLP for the second Q-function (for double Q-learning)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict() # For logging
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        """
        Forward pass of the critic.
        Args:
            obs (torch.Tensor): Batch of observations.
            action (torch.Tensor): Batch of actions.
            detach_encoder (bool): Whether to detach the encoder features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q1 and Q2 values.
        """
        assert obs.size(0) == action.size(0)
        obs_features = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs_features, action], dim=-1) # Concatenate features and actions
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DRQ_METRAAgent(object):
    """
    DrQ agent with METRA skill discovery integration.
    Manages the actor, critic, discriminator, and their updates.
    Configuration is now passed via explicit arguments (formerly Hydra).
    """
    def __init__(self,
                 obs_shape, action_shape, action_range, device, # Core RL setup
                 feature_dim, # Encoder specific
                 critic_hidden_dim, critic_hidden_depth, # Critic specific
                 actor_hidden_dim, actor_hidden_depth, actor_log_std_min, actor_log_std_max, # Actor specific
                 num_skills, skill_embedding_dim, diayn_intrinsic_reward_coeff, # DIAYN specific
                 discriminator_hidden_dim, discriminator_hidden_depth, # Discriminator specific
                 discount, init_temperature, lr, lr_discriminator, # RL algorithm HPs & optimizers
                 actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size # Update rules
                 ):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau # Soft update coefficient for target critic
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        # DIAYN specific attributes
        self.diayn_intrinsic_reward_coeff = diayn_intrinsic_reward_coeff
        self.num_skills = num_skills

        # Instantiate Actor network
        self.actor = Actor(obs_shape, feature_dim, action_shape, actor_hidden_dim, actor_hidden_depth,
                           [actor_log_std_min, actor_log_std_max], num_skills, skill_embedding_dim).to(device)

        # Instantiate Critic network (and target critic)
        self.critic = Critic(obs_shape, feature_dim, action_shape, critic_hidden_dim, critic_hidden_depth).to(device)
        self.critic_target = Critic(obs_shape, feature_dim, action_shape, critic_hidden_dim, critic_hidden_depth).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Initialize target critic weights

        # Important: Tie convolutional layers between actor's and critic's encoders for shared representation
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # Entropy temperature alpha for SAC
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_shape[0] # Target entropy is usually -|A|

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # DIAYN: Instantiate Discriminator and its optimizer
        # feature_dim here is the output dim of the shared encoder (from critic/actor)
        self.discriminator = Discriminator(feature_dim, num_skills,
                                           discriminator_hidden_dim, discriminator_hidden_depth).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)

        self.train() # Set networks to training mode
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, skill, sample=False):
        """Selects an action based on observation and current skill (if DIAYN active)."""
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0) # Add batch dimension

        # Prepare skill tensor for actor
        # Skill is expected to be a scalar int by this point from Workspace
        skill_tensor = torch.tensor([skill], device=self.device).long()
        if self.num_skills > 0 and skill_tensor.ndim == 1: # Actor expects batched skill input
             skill_tensor = skill_tensor.unsqueeze(0) # Shape: [1,1] or [1] -> [1,1] for consistency with batching

        dist = self.actor(obs, skill_tensor if self.num_skills > 0 else None) # Pass skill to actor
        action = dist.sample() if sample else dist.mean # Sample or take mean
        action = action.clamp(*self.action_range) # Clamp action to valid range
        assert action.ndim == 2 and action.shape[0] == 1, "Action output shape incorrect."
        return utils.to_np(action[0]) # Convert to numpy array for environment

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, skills, logger, step):
        """Updates the critic network(s)."""

        # DIAYN: Compute intrinsic reward using current obs and skills.
        # The choice of (obs, skill) vs (next_obs, skill) for r_i depends on DIAYN variant.
        # Here, r_i = log D(skill | obs) - log p(skill)
        if self.num_skills > 0:
            intrinsic_reward = self.compute_intrinsic_reward(obs, skills) # Use current obs
            logger.log('train_critic/intrinsic_reward', intrinsic_reward.mean(), step)
            # Augment extrinsic reward with intrinsic reward
            reward = reward + self.diayn_intrinsic_reward_coeff * intrinsic_reward
            logger.log('train_critic/total_reward', reward.mean(), step) # Log total reward for critic

        with torch.no_grad(): # Operations inside this block do not track gradients
            # Get next action and its log probability from the actor for the target Q calculation
            dist = self.actor(next_obs, skills if self.num_skills > 0 else None)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True) # Log prob of next action
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action) # Target Q-values from target critic
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob # Target value (SAC style)
            target_Q = reward + (not_done * self.discount * target_V) # Bellman equation

            # Repeat for augmented next observations (DrQ specific)
            dist_aug = self.actor(next_obs_aug, skills if self.num_skills > 0 else None)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
            target_Q1_aug, target_Q2_aug = self.critic_target(next_obs_aug, next_action_aug)
            target_V_aug = torch.min(target_Q1_aug, target_Q2_aug) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V_aug)

            # Average target Q from original and augmented next_obs for data regularization
            target_Q = (target_Q + target_Q_aug) / 2

        # Get current Q estimates from the online critic
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, skills, logger, step): # Added skills
        # Detach encoder for actor update to prevent gradients from flowing back to encoder from actor loss
        dist = self.actor(obs, skills if self.num_skills > 0 else None, detach_encoder=True)
        action = dist.rsample() # Sample action from policy
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) # Log prob of sampled action

        # Detach encoder for critic when used in actor loss calculation
        # Actor_Q values are based on the critic, which is trained on augmented rewards if DIAYN is active.
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2) # Use min Q-value for actor update (SAC style)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug, skills = replay_buffer.sample(
            self.batch_size)

        # Note: extrinsic reward is already augmented with intrinsic_reward in update_critic if DIAYN is active.
        # This step is where intrinsic rewards influence the policy via critic's Q-values.

        # Update critic using observations, actions, and augmented rewards
        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, skills, logger, step)

        # Update actor and temperature alpha (SAC update)
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, skills, logger, step) # Pass skills for skill-conditioned actor

        # Update target critic network (soft update)
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        # DIAYN: Update discriminator
        if self.num_skills > 0: # Only update discriminator if DIAYN is active
            self.update_discriminator(obs, skills, logger, step) # Use current obs and skills from buffer

    def compute_intrinsic_reward(self, obs, skill):
        """
        Computes the DIAYN intrinsic reward: r_i = log D(z|s) - log p(z).
        Args:
            obs (torch.Tensor): Batch of observations (states).
            skill (torch.Tensor): Batch of skills (actions/latents z).
        Returns:
            torch.Tensor: Batch of intrinsic rewards.
        """
        if self.num_skills == 0: # Should not be called if no skills
            return torch.zeros((obs.size(0), 1), device=self.device)

        # Encode observation to get features (state representation)
        # Use no_grad as this is for reward calculation, not for training the encoder here.
        # detach=True on encoder output ensures no gradients flow back to encoder from discriminator.
        with torch.no_grad():
            features = self.critic.encoder(obs, detach=True)

        # Get skill logits from discriminator D(z|s)
        skill_logits = self.discriminator(features)

        # Calculate log p(z|s) using cross-entropy trick or manual log_softmax
        # skill is [batch_size, 1], .long() for gather
        skill = skill.reshape((-1, 1))
        log_p_z_s = F.log_softmax(skill_logits, dim=-1).gather(1, skill.long())

        # Calculate log p(z) - assuming a uniform prior over skills
        # log p(z) = -log(num_skills) since p(z) = 1/num_skills
        log_p_z_uniform = -torch.log(torch.tensor(self.num_skills, dtype=torch.float32, device=skill_logits.device))

        intrinsic_reward = log_p_z_s - log_p_z_uniform

        return intrinsic_reward.reshape((-1, 1)) # Ensure shape [batch_size, 1]

    def update_discriminator(self, obs, skill, logger, step):
        """
        Updates the DIAYN skill discriminator D(z|s).
        Args:
            obs (torch.Tensor): Batch of observations (states).
            skill (torch.Tensor): Batch of skills (true labels for discriminator).
            logger (Logger): Logger object for recording metrics.
            step (int): Current training step.
        """
        # Get features from observation (state representation)
        # Use no_grad as encoder is trained by critic/actor, not directly by discriminator.
        with torch.no_grad():
            features = self.critic.encoder(obs, detach=True)

        # Predict skill logits using the discriminator
        predicted_skill_logits = self.discriminator(features)

        # Calculate discriminator loss (cross-entropy between predicted skill and actual skill)
        # skill is [batch_size, 1], squeeze to [batch_size] for cross_entropy target.
        discriminator_loss = F.cross_entropy(predicted_skill_logits, skill.squeeze(-1).long())

        logger.log('train_discriminator/loss', discriminator_loss, step)

        # Optimize the discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def save(self, save_path):
        params_dict = {}
        params_dict['actor'] = self.actor.state_dict()
        params_dict['critic'] = self.critic.state_dict()
        params_dict['critic_target'] = self.critic_target.state_dict()
        params_dict['discriminator'] = self.discriminator.state_dict()
        params_dict['log_alpha'] = self.log_alpha
        params_dict['actor_optimizer'] = self.actor_optimizer.state_dict()
        params_dict['critic_optimizer'] = self.critic_optimizer.state_dict()
        params_dict['discriminator_optimizer'] = self.discriminator_optimizer.state_dict()
        params_dict['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()
        torch.save(params_dict, save_path)

    def load(self, load_path, device='cpu'):
        params_dict = torch.load(load_path, map_location=device)
        self.actor.load_state_dict(params_dict['actor'])
        self.critic.load_state_dict(params_dict['critic'])
        self.critic_target.load_state_dict(params_dict['critic_target'])
        self.discriminator.load_state_dict(params_dict['discriminator'])
        self.log_alpha = params_dict['log_alpha']
        self.actor_optimizer.load_state_dict(params_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(params_dict['critic_optimizer'])
        self.discriminator_optimizer.load_state_dict(params_dict['discriminator_optimizer'])
        self.log_alpha_optimizer.load_state_dict(params_dict['log_alpha_optimizer'])
        


class Discriminator(nn.Module):
    """
    DIAYN Discriminator network D(z|s).
    Predicts the skill 'z' given the current state 's'.
    """
    def __init__(self, feature_dim, num_skills, hidden_dim=1024, hidden_depth=2):
        """
        Args:
            feature_dim (int): Dimension of the input features (from encoder).
            num_skills (int): Number of skills to classify (output dimension).
            hidden_dim (int): Dimension of hidden layers.
            hidden_depth (int): Number of hidden layers.
        """
        super().__init__()
        # MLP structure for the discriminator
        # Note: utils.mlp typically adds a ReLU after the last linear layer if output_dim > 1.
        # For classification logits, a final ReLU is undesirable.
        # So, constructing layers manually here to ensure no final activation on logits.
        layers = []
        current_dim = feature_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_skills)) # Output layer for skill logits
        self.trunk = nn.Sequential(*layers)

        self.apply(utils.weight_init) # Apply weight initialization

    def forward(self, features):
        """
        Forward pass of the discriminator.
        Args:
            features (torch.Tensor): Input features (encoded state).
        Returns:
            torch.Tensor: Logits for skill classification.
        """
        return self.trunk(features)
