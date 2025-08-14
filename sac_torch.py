import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ReplayBuffer:
    """
    A replay buffer for storing and sampling experiences for reinforcement learning.
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    """
    The Critic Network (Soft Q-value function).
    """
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='model/sac'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        
        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    """
    The Actor Network (Policy).
    """
    def __init__(self, lr, input_dims, n_actions, max_action, fc1_dims=256, fc2_dims=256, name='actor', chkpt_dir='model/sac'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class SACAgent:
    """
    The Soft Actor-Critic Agent with automatic temperature (alpha) tuning.
    """
    def __init__(self, lr_actor=0.0001, lr_critic=0.0001, input_dims=(5,),
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 batch_size=256, target_entropy=None):
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = [1.0] * n_actions

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.actor = ActorNetwork(lr_actor, input_dims, n_actions=n_actions, max_action=self.max_action, name='actor_sac')
        self.critic_1 = CriticNetwork(lr_critic, input_dims, n_actions=n_actions, name='critic_1_sac')
        self.critic_2 = CriticNetwork(lr_critic, input_dims, n_actions=n_actions, name='critic_2_sac')
        self.target_critic_1 = CriticNetwork(lr_critic, input_dims, n_actions=n_actions, name='target_critic_1_sac')
        self.target_critic_2 = CriticNetwork(lr_critic, input_dims, n_actions=n_actions, name='target_critic_2_sac')

        # --- Automatic Temperature Tuning ---
        if target_entropy is None:
            # Heuristic for target entropy
            self.target_entropy = -torch.tensor(n_actions, dtype=torch.float32).to(self.device)
        else:
            self.target_entropy = torch.tensor(target_entropy, dtype=torch.float32).to(self.device)
        
        # log_alpha is the learnable parameter
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        
        self.update_network_parameters(tau=1)

    @property
    def alpha(self):
        """Returns the current alpha value by exponentiating the learnable log_alpha."""
        return self.log_alpha.exp()

    def choose_action(self, observation, evaluate=False):
        """
        Chooses an action based on the policy.

        Args:
            observation: The current state of the environment.
            evaluate (bool): If True, returns the deterministic action (mean).
                             If False (default), returns a stochastic action for exploration.
        """
        state = torch.tensor([observation], dtype=torch.float32).to(self.device)

        # For testing/evaluation, we want the best-known (deterministic) action.
        if evaluate:
            # Get the mean action directly from the actor's forward pass
            mu, _ = self.actor.forward(state)
            # Apply tanh to squash the action to the range [-1, 1]
            actions = torch.tanh(mu)
        # For training, we want to explore by sampling (stochastic action).
        else:
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Soft update for both target critic networks
        for target_params, local_params in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)
        
        for target_params, local_params in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None, None

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done).to(self.device)

        # --- Update Critic Networks ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(new_state, reparameterize=True)
            q1_next_target = self.target_critic_1.forward(new_state, next_actions)
            q2_next_target = self.target_critic_2.forward(new_state, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_probs
            q_target = reward + self.gamma * q_next_target.view(-1)
            q_target[done] = reward[done]

        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)
        
        critic_1_loss = F.mse_loss(q1, q_target)
        critic_2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic_1_loss + critic_2_loss
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # --- Update Actor and Alpha ---
        # Freeze Q-networks to save computation
        for p in self.critic_1.parameters(): p.requires_grad = False
        for p in self.critic_2.parameters(): p.requires_grad = False

        new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_new_policy = self.critic_1.forward(state, new_actions)
        q2_new_policy = self.critic_2.forward(state, new_actions)
        min_q_new_policy = torch.min(q1_new_policy, q2_new_policy).view(-1)
        
        # Actor Loss
        actor_loss = (self.alpha.detach() * log_probs.view(-1) - min_q_new_policy).mean()
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Unfreeze Q-networks
        for p in self.critic_1.parameters(): p.requires_grad = True
        for p in self.critic_2.parameters(): p.requires_grad = True
        
        # Alpha (temperature) Loss
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update target networks ---
        self.update_network_parameters()

        # Return actor_loss, combined critic_loss, and alpha_loss for logging
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()