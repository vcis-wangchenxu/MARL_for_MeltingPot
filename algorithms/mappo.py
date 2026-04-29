import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from networks.MAPPO_Network import MAPPOActor, MAPPOCritic

class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) Algorithm.
    Supports both Shared (MAPPO) and Independent (IPPO)parameter configurations.
    """
    def __init__(self, env_info, args):
        """
        Initialize MAPPO algorithm.

        Args:
            env_info (dict): Environment information.
                - n_agents (int): Number of agents.
                - n_actions (int): Number of actions.
                - obs_rgb_shape (tuple): Shape of RGB observation (C, H, W).
                - obs_vector_dim (int): Dimension of vector observation.
                - state_shape (tuple): Shape of global state (for centralized critic).
                - global_shape (tuple): Shape of global RGB state.
            args (Namespace): Configuration arguments (e.g., hidden_dim, lr, device).
        """
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.obs_shape = env_info['obs_rgb_shape']
        self.vector_dim = env_info['obs_vector_dim']
        
        self.state_shape = env_info.get('state_shape')
        self.global_state_shape = env_info.get('global_shape')
        
        # Critic Input Shape Logic
        if self.state_shape is not None:
            self.critic_input_shape = self.state_shape
            self.use_flatten_state = True
        elif self.global_state_shape is not None:
            self.critic_input_shape = (self.global_state_shape[2], self.global_state_shape[0], self.global_state_shape[1]) 
            self.use_flatten_state = False
        else:
            raise ValueError("[MAPPO] Critic needs state_shape or global_shape!")

        self.args = args
        self.device = torch.device(args.device)
        self.hidden_dim = args.hidden_dim
        self.rnn_layers = args.rnn_layers
        self.share_params = args.share_parameters
        
        # Agent ID Config
        self.use_agent_id = True 

        # --- Initialize Networks ---
        if self.share_params:
            print(f"[MAPPO] Parameter Sharing Enabled (1 Actor, 1 Critic).")
            self.actor = MAPPOActor(
                self.obs_shape, self.vector_dim, self.n_actions, self.n_agents,
                self.hidden_dim, self.rnn_layers, use_agent_id=self.use_agent_id
            ).to(self.device)
            
            self.critic = MAPPOCritic(
                self.critic_input_shape, self.hidden_dim, 
                n_agents=self.n_agents, use_agent_id=self.use_agent_id
            ).to(self.device)

            self.optimizer = optim.Adam([
                {'params': self.actor.parameters(), 'lr': args.lr},
                {'params': self.critic.parameters(), 'lr': args.lr}
            ], eps=args.eps)
            
        else:
            print(f"[MAPPO] Parameter Sharing Disabled ({self.n_agents} Actors, {self.n_agents} Critics).")
            self.actors = nn.ModuleList([
                MAPPOActor(
                    self.obs_shape, self.vector_dim, self.n_actions, self.n_agents,
                    self.hidden_dim, self.rnn_layers, use_agent_id=False
                ) for _ in range(self.n_agents)
            ]).to(self.device)
            
            self.critics = nn.ModuleList([
                MAPPOCritic(
                    self.critic_input_shape, self.hidden_dim, 
                    n_agents=self.n_agents, use_agent_id=False
                ) for _ in range(self.n_agents)
            ]).to(self.device)
            
            params = []
            for alg in self.actors: params.extend(alg.parameters())
            for alg in self.critics: params.extend(alg.parameters())
            self.optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)

    def init_hidden(self, num_envs):
        if self.share_params:
            h = self.actor.init_hidden(num_envs * self.n_agents, self.device)
            h = h.view(self.rnn_layers, num_envs, self.n_agents, self.hidden_dim)
            return h.permute(1, 2, 0, 3).contiguous()
        else:
            h_list = [actor.init_hidden(num_envs, self.device) for actor in self.actors]
            h_stack = torch.stack(h_list, dim=1) 
            return h_stack.permute(2, 1, 0, 3).contiguous()

    @torch.no_grad()
    def take_action(self, obs, state, global_state, hidden_state):
        """
        Select actions using the current policy (Actor).

        Args:
            obs (dict): Dictionary of observations.
                - 'rgb': Tensor of shape (Batch, N_Agents, C, H, W).
                - 'vector': Tensor of shape (Batch, N_Agents, Vector_Dim).
            state (torch.Tensor or None): Flattened global state for Critic.
                Shape: (Batch, N_Agents * C, H, W) or specific to environment.
            global_state (torch.Tensor or None): Global world RGB state for Critic.
                Shape: (Batch, C, H, W).
            hidden_state (torch.Tensor): Current RNN hidden states.
                Shape: (Batch, N_Agents, Layers, Hidden).

        Returns:
            values (np.ndarray): Value estimates from Critic. Shape: (Batch, N_Agents).
            actions (np.ndarray): Selected actions. Shape: (Batch, N_Agents).
            log_probs (np.ndarray): Log probabilities of selected actions. Shape: (Batch, N_Agents).
            next_hidden (torch.Tensor): Updated RNN hidden states. Shape: (Batch, N_Agents, Layers, Hidden).
        """
        B, N = obs['rgb'].shape[:2]
        
        actions_list = []
        log_probs_list = []
        values_list = []
        next_hidden_list = []

        if self.share_params:
            # === Shared Mode ===
            obs_seq = {
                'rgb': obs['rgb'].view(B*N, 1, *self.obs_shape),
                'vector': obs['vector'].view(B*N, 1, -1) if self.vector_dim > 0 else None
            }
            h_in = hidden_state.permute(2, 0, 1, 3).contiguous().view(self.rnn_layers, B*N, self.hidden_dim)
            
            ids = torch.arange(N, device=self.device).repeat(B) 
            ids_onehot = torch.nn.functional.one_hot(ids, num_classes=self.n_agents).float().unsqueeze(1)
            
            logits, h_out = self.actor(obs_seq, h_in, agent_id=ids_onehot)
            dist = Categorical(logits=logits.view(B*N, self.n_actions))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions = action.view(B, N).cpu().numpy()
            log_probs = log_prob.view(B, N).cpu().numpy()
            next_hidden = h_out.view(self.rnn_layers, B, N, self.hidden_dim).permute(1, 2, 0, 3)
            
            critic_in = state.unsqueeze(1) if self.use_flatten_state else global_state.unsqueeze(1)
            critic_in_rep = critic_in.unsqueeze(1).expand(B, N, 1, *self.critic_input_shape).reshape(B*N, 1, *self.critic_input_shape)
            values = self.critic(critic_in_rep, agent_id=ids_onehot) 
            values = values.view(B, N).cpu().numpy()

            return values, actions, log_probs, next_hidden

        else:
            # === Independent Mode ===
            critic_in = state.unsqueeze(1) if self.use_flatten_state else global_state.unsqueeze(1)

            for i in range(self.n_agents):
                obs_i = {
                    'rgb': obs['rgb'][:, i].unsqueeze(1),
                    'vector': obs['vector'][:, i].unsqueeze(1) if self.vector_dim > 0 else None
                }
                h_i = hidden_state[:, i].permute(1, 0, 2).contiguous() 
                
                logits, h_out = self.actors[i](obs_i, h_i) 
                dist = Categorical(logits=logits.squeeze(1))
                act = dist.sample()
                lp = dist.log_prob(act)
                
                actions_list.append(act)
                log_probs_list.append(lp)
                next_hidden_list.append(h_out.permute(1, 0, 2))
                
                val = self.critics[i](critic_in) 
                values_list.append(val.squeeze(1).squeeze(1))

            actions = torch.stack(actions_list, dim=1).cpu().numpy()
            log_probs = torch.stack(log_probs_list, dim=1).cpu().numpy()
            values = torch.stack(values_list, dim=1).cpu().numpy()
            next_hidden = torch.stack(next_hidden_list, dim=1) 
            
            return values, actions, log_probs, next_hidden

    def update(self, buffer):
        """
        Update the policy and value function using the PPO algorithm.

        Args:
            buffer (RolloutBuffer): Buffer containing collected trajectories.

        Returns:
            dict: Training statistics (Actor loss, Critic loss, Entropy).
        """
        data = buffer.get_data()
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']
        
        adv = np.zeros_like(rewards)
        last_gae_lam = 0
        T = self.args.buffer_size
        
        for t in reversed(range(T)):
            # [Fix] Use dones[t] for masking next value
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.args.gamma * values[t+1] * next_non_terminal - values[t]
            last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_gae_lam
            adv[t] = last_gae_lam
            
        returns = adv + values[:T]
        
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        chunk_len = 16 if self.args.buffer_size >= 16 else self.args.buffer_size
        
        for _ in range(self.args.ppo_epoch):
            data_generator = buffer.recurrent_generator(adv, returns, self.args.batch_size, chunk_len)
            
            for sample in data_generator:
                (mb_obs, mb_state, mb_global_state, mb_hidden, mb_actions, 
                 mb_values, mb_returns, mb_log_probs, mb_adv, mb_masks, mb_agent_ids) = sample
                
                B_chunk, T_chunk, N_agents = mb_actions.shape
                
                if self.share_params:
                    # === Shared Update ===
                    
                    # [Helper] Permute (B, T, N) -> (B, N, T) -> Reshape (B*N, T)
                    # This ensures temporal order is preserved for each agent individually
                    def to_bn_t(x): 
                        return x.permute(0, 2, 1, *range(3, x.ndim)).reshape(B_chunk * N_agents, T_chunk, *x.shape[3:])
                    
                    # 1. Inputs
                    obs_in = {}
                    obs_in['rgb'] = to_bn_t(mb_obs['rgb'])
                    if mb_obs.get('vector') is not None:
                        obs_in['vector'] = to_bn_t(mb_obs['vector'])
                    else:
                        obs_in['vector'] = None
                    
                    # 2. Hidden States
                    # mb_hidden: (Layers, Batch, Agents, Hidden) -> (Layers, Batch*Agents, Hidden)
                    # Memory layout of mb_hidden is consistent with to_bn_t (Batch major, then Agent)
                    h_in = mb_hidden.permute(0, 1, 2, 3).reshape(self.rnn_layers, B_chunk * N_agents, self.hidden_dim)
                    
                    # 3. Agent IDs
                    # mb_agent_ids: (B, T, N) -> to_bn_t -> (B*N, T)
                    ids_bn_t = to_bn_t(mb_agent_ids)
                    ids_onehot = torch.nn.functional.one_hot(ids_bn_t, num_classes=self.n_agents).float()
                    
                    # 4. Scalars
                    act_flat = to_bn_t(mb_actions)
                    old_lp_flat = to_bn_t(mb_log_probs)
                    adv_flat = to_bn_t(mb_adv)
                    ret_flat = to_bn_t(mb_returns)
                    val_flat = to_bn_t(mb_values)
                    
                    # --- Actor Forward ---
                    new_logits, _ = self.actor(obs_in, h_in, agent_id=ids_onehot)
                    dist = Categorical(logits=new_logits)
                    new_log_probs = dist.log_prob(act_flat)
                    dist_entropy = dist.entropy().mean()
                    
                    # --- Critic Forward ---
                    if self.use_flatten_state:
                        state_in = to_bn_t(mb_state)
                    else:
                        state_in = to_bn_t(mb_global_state)
                    
                    new_values = self.critic(state_in, agent_id=ids_onehot)
                    new_values = new_values.squeeze(-1)

                    # --- Losses ---
                    ratio = torch.exp(new_log_probs - old_lp_flat)
                    surr1 = ratio * adv_flat
                    surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_flat
                    actor_loss = -torch.min(surr1, surr2).mean()

                    if self.args.use_clipped_value_loss:
                        value_pred_clipped = val_flat + (new_values - val_flat).clamp(-self.args.clip_param, self.args.clip_param)
                        value_losses = (new_values - ret_flat).pow(2)
                        value_losses_clipped = (value_pred_clipped - ret_flat).pow(2)
                        critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        critic_loss = 0.5 * ((new_values - ret_flat).pow(2)).mean()

                    loss = actor_loss + self.args.value_loss_coef * critic_loss - self.args.entropy_coef * dist_entropy
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(dist_entropy.item())

                else:
                    # === Independent Update ===
                    total_a_loss = 0; total_c_loss = 0; total_e_loss = 0
                    
                    for i in range(self.n_agents):
                        # Slice data for agent i
                        # mb_obs['rgb']: (B, T, N, ...) -> slice -> (B, T, ...)
                        obs_i = {
                            'rgb': mb_obs['rgb'][:, :, i],
                            'vector': mb_obs['vector'][:, :, i] if self.vector_dim > 0 else None
                        }
                        h_i = mb_hidden[:, i].permute(1, 0, 2).contiguous() 
                        
                        act_i = mb_actions[:, :, i]
                        old_lp_i = mb_log_probs[:, :, i]
                        adv_i = mb_adv[:, :, i]
                        ret_i = mb_returns[:, :, i]
                        val_i = mb_values[:, :, i]
                        
                        if self.use_flatten_state:
                            state_i = mb_state[:, :, i]
                        else:
                            state_i = mb_global_state[:, :, i]

                        # Forward
                        new_logits, _ = self.actors[i](obs_i, h_i)
                        dist = Categorical(logits=new_logits)
                        new_log_probs = dist.log_prob(act_i)
                        dist_entropy = dist.entropy().mean()
                        
                        new_values = self.critics[i](state_i).squeeze(-1)
                        
                        ratio = torch.exp(new_log_probs - old_lp_i)
                        surr1 = ratio * adv_i
                        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_i
                        a_loss = -torch.min(surr1, surr2).mean()
                        
                        c_loss = 0.5 * ((new_values - ret_i).pow(2)).mean()
                        
                        loss = a_loss + self.args.value_loss_coef * c_loss - self.args.entropy_coef * dist_entropy
                        
                        total_a_loss += a_loss.item()
                        total_c_loss += c_loss.item()
                        total_e_loss += dist_entropy.item()
                        
                        loss.backward()
                    
                    nn.utils.clip_grad_norm_(self.actors.parameters(), self.args.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critics.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    actor_losses.append(total_a_loss / self.n_agents)
                    critic_losses.append(total_c_loss / self.n_agents)
                    entropy_losses.append(total_e_loss / self.n_agents)

        return {
            "loss_actor": np.mean(actor_losses),
            "loss_critic": np.mean(critic_losses),
            "loss_entropy": np.mean(entropy_losses)
        }

    def save(self, path):
        """
        Save the model parameters to a file.
        
        Args:
            path (str): File path to save the model.
        """
        if self.share_params:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
        else:
            torch.save({
                'actors': self.actors.state_dict(),
                'critics': self.critics.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)

    def load(self, path):
        """
        Load model parameters from a file.
        
        Args:
            path (str): File path to load the model from.
        """
        checkpoint = torch.load(path)
        if self.share_params:
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.actors.load_state_dict(checkpoint['actors'])
            self.critics.load_state_dict(checkpoint['critics'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])