from typing import Tuple, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.DRQN import AgentDRQN

class VDN:
    """
    Value Decomposition Networks (VDN) algorithm for Melting Pot.
    Core idea: Q_tot(s, a) = Sum(Q_i(s^i, a^i))
    Function: Implements VDN for multi-modal observations (RGB + Vector).
    """
    def __init__(self, env_info: dict, args: Dict):
        """
        Initialize VDN Algorithm.
        """
        self.env_info = env_info
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']

        self.obs_rgb_shape = env_info['obs_rgb_shape']  # (C, H, W)
        self.vector_dim = env_info['obs_vector_dim']  # int

        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma

        self.hidden_dim = args.rnn_hidden_dim
        self.layers = args.rnn_layers
        self.tau = args.target_tau

        self.share_params = args.share_parameters

        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.epsilon = self.epsilon_start

        self.device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

        if self.share_params:
            print(f"[VDN] Parameter Sharing Enabled. All {self.n_agents} agents share one DRQN (RGB+Vector).")
            self.agent = AgentDRQN(
                obs_rgb_shape=self.obs_rgb_shape,
                vector_dim=self.vector_dim,
                n_actions=self.n_actions,
                n_agents=self.n_agents,
                rnn_hidden_dim=self.hidden_dim,
                rnn_layers=self.layers,
                use_agent_id=True,
            ).to(self.device)

            self.target_agent = AgentDRQN(
                obs_rgb_shape=self.obs_rgb_shape,
                vector_dim=self.vector_dim,
                n_actions=self.n_actions,
                n_agents=self.n_agents,
                rnn_hidden_dim=self.hidden_dim,
                rnn_layers=self.layers,
                use_agent_id=True,
            ).to(self.device)

            self.target_agent.load_state_dict(self.agent.state_dict())
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr)
        
        else:
            print(f"[VDN] Parameter Sharing Disabled. Creating {self.n_agents} independent DRQNs.")
            self.agents = nn.ModuleList([
                AgentDRQN(
                    obs_rgb_shape=self.obs_rgb_shape,
                    vector_dim=self.vector_dim,
                    n_actions=self.n_actions,
                    n_agents=1,
                    rnn_hidden_dim=self.hidden_dim,
                    rnn_layers=self.layers,
                    use_agent_id=False,
                )
                for _ in range(self.n_agents)
            ]).to(self.device)

            self.target_agents = nn.ModuleList([
                AgentDRQN(
                    obs_rgb_shape=self.obs_rgb_shape,
                    vector_dim=self.vector_dim,
                    n_actions=self.n_actions,
                    n_agents=1,
                    rnn_hidden_dim=self.hidden_dim,
                    rnn_layers=self.layers,
                    use_agent_id=False,
                )
                for _ in range(self.n_agents)
            ]).to(self.device)

            self.target_agents.load_state_dict(self.agents.state_dict())
            self.optimizer = optim.Adam(self.agents.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()
        self._train_step_count = 0

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize the hidden states for the agents' RNNs.
        """
        if self.share_params:
            h = self.agent.init_hidden(batch_size * self.n_agents, device=self.device)     # (Layers, Batch * N_Agents, Hidden)
            h = h.view(self.agent.rnn_layers, batch_size, self.n_agents, self.hidden_dim)  # (L, B, N, H)
            h = h.permute(1, 2, 0, 3).contiguous()                                         # (B, N, L, H)
            return h                                     
        else:
            h_list = [agent.init_hidden(batch_size, device=self.device) for agent in self.agents] # (L, B, H) per agent
            h_stack = torch.stack(h_list, dim=0)                                                  # (N, L, B, H)
            h = h_stack.permute(2, 0, 1, 3).contiguous()                                          # (B, N, L, H)
            return h
        
    @torch.no_grad()
    def take_action(self, obs_dict: Dict[str, torch.Tensor], 
                    hidden_state, 
                    current_step, 
                    evaluation=False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Select actions for all agents based on current observations and hidden states.
        """
        # obs_dict['rgb'] shape is (B, N, C, H, W)
        batch_size = obs_dict['rgb'].shape[0]

        # Update Epsilon
        if evaluation:
            self.epsilon = 0.0
            explore_mask = np.zeros((batch_size, self.n_agents), dtype=bool)
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon_start - \
                           (self.epsilon_start - self.epsilon_end) * (current_step / self.epsilon_decay))
            rand_probs = np.random.rand(batch_size, self.n_agents)
            explore_mask = rand_probs < self.epsilon

        if self.share_params:
            rgb = obs_dict['rgb'] 
            B, N, C, H, W = rgb.shape
            rgb_flat = rgb.view(B * N, C, H, W)    # (B, N, C, H, W) -> (B*N, C, H, W)
            rgb_seq = rgb_flat.unsqueeze(1)        # -> (B*N, 1, C, H, W) [Sequence Length = 1]

            vector_seq = None
            if self.vector_dim > 0:
                vector = obs_dict['vector'] 
                vector_flat = vector.view(B * N, -1)    # (B, N, V) -> (B*N, V)
                vector_seq = vector_flat.unsqueeze(1)   # (B*N, 1, V) [Sequence Length = 1]

            # Construct input dict for AgentDRQN
            model_input = {'rgb': rgb_seq, 'vector': vector_seq}
            
            # (B, N, L, H) -> (L, B, N, H) -> (L, B*N, H)
            h_perm = hidden_state.permute(2, 0, 1, 3) 
            h_flat = h_perm.reshape(self.agent.rnn_layers, B * N, self.hidden_dim)

            agent_ids = torch.arange(N, device=self.device).repeat(B)
            agent_ids_seq = agent_ids.unsqueeze(1)  # (B*N, 1)

            q_values_seq, h_out_flat = self.agent(model_input, h_flat, agent_id=agent_ids_seq)
            # q_values_seq: (B*N, 1, n_actions)
            # h_out_flat: (L, B*N, H)

            q_values_flat = q_values_seq.squeeze(1) # (B*N, n_actions)
            q_values = q_values_flat.view(B, N, -1) # (B, N, n_actions)

            # Process Output Hidden: (L, B*N, H) -> (B, N, L, H)
            h_out = h_out_flat.view(self.agent.rnn_layers, B, N, self.hidden_dim)
            next_hidden_state = h_out.permute(1, 2, 0, 3)

            exploit_actions = q_values.argmax(dim=-1).cpu().numpy() # (B, N)
            random_actions = np.random.randint(0, self.n_actions, size=(batch_size, N))
            final_actions = np.where(explore_mask, random_actions, exploit_actions)

            return final_actions, next_hidden_state

        else:
            # Independent Parameters (Loop over agents)
            actions = []
            next_h_list = []
            
            for i in range(self.n_agents):
                # Extract agent specific obs
                agent_rgb = obs_dict['rgb'][:, i]      # (B, C, H, W)
                agent_rgb_seq = agent_rgb.unsqueeze(1) # (B, 1, C, H, W)
                
                agent_vec_seq = None
                if self.vector_dim > 0:
                    agent_vec = obs_dict['vector'][:, i]   # (B, V)
                    agent_vec_seq = agent_vec.unsqueeze(1) # (B, 1, V)
                
                model_input = {'rgb': agent_rgb_seq, 'vector': agent_vec_seq}

                # Hidden: (B, L, H) -> (L, B, H)
                agent_h = hidden_state[:, i].permute(1, 0, 2).contiguous()

                net = self.agents[i]
                q_values_seq, h_out = net(model_input, agent_h) 
                
                q_values = q_values_seq.squeeze(1) # (B, n_actions)
                exploit = q_values.argmax(dim=-1).cpu().numpy()
                random_act = np.random.randint(0, self.n_actions, size=batch_size)
                
                mask_i = explore_mask[:, i]
                chosen = np.where(mask_i, random_act, exploit)
                
                actions.append(chosen)
                next_h_list.append(h_out.permute(1, 0, 2))  # (B, L, H)

            final_actions = np.stack(actions, axis=1)           # (B, N)
            next_hidden_state = torch.stack(next_h_list, dim=1) # (B, N, L, H)
            return final_actions, next_hidden_state

    def update(self, sample):
        """
        Update the network parameters using the VDN logic (Q_tot = sum Q_i).
        Note: VDN does NOT use global state (sample['global']['state']), even if Buffer provides it.
        """
        mask = sample['global']['mask'] # (B, L, 1)
        mask_sum = torch.clamp(mask.sum(), min=1.0)

        init_hidden = sample['all_agents']['init_hidden']
        target_init_hidden = sample['all_agents']['init_target_hidden']
        global_dones = sample['global']['dones']

        if self.share_params:
            all_data = sample['all_agents']
            # Unpack Obs Dicts
            obs_dict = all_data['obs'] # {'rgb': (B, L, N, C, H, W), 'vector': (B, L, N, V)}
            next_obs_dict = all_data['next_obs']
            
            actions = all_data['actions'].long()
            rewards = all_data['rewards']
            
            # --- Flatten Helper ---
            def flatten_batch_agent_dims(data_dict, key_prefix=""):
                """ Flattens (Batch, Seq, N_Agents, ...) -> (Batch*N_Agents, Seq, ...) """
                flat_input = {}
                
                # RGB
                rgb = data_dict['rgb'] # (B, L, N, C, H, W)
                B, L, N, C, H, W = rgb.shape
                # Permute to (B, N, L, ...) then Flatten B*N
                flat_input['rgb'] = rgb.permute(0, 2, 1, 3, 4, 5).reshape(B*N, L, C, H, W)
                
                # Vector
                if 'vector' in data_dict and data_dict['vector'] is not None:
                    vec = data_dict['vector'] # (B, L, N, V)
                    flat_input['vector'] = vec.permute(0, 2, 1, 3).reshape(B*N, L, -1)
                else:
                    flat_input['vector'] = None
                return flat_input, (B, L, N)

            model_input, (B, L, N) = flatten_batch_agent_dims(obs_dict)
            target_model_input, _ = flatten_batch_agent_dims(next_obs_dict)
            
            # Hidden State
            hidden_flat = init_hidden.reshape(self.agent.rnn_layers, B*N, self.hidden_dim)
            target_hidden_flat = target_init_hidden.reshape(self.agent.rnn_layers, B*N, self.hidden_dim)
            
            # Agent IDs
            ids = torch.arange(N, device=self.device).repeat(B) # (B*N)
            ids_seq = ids.unsqueeze(1).expand(-1, L) # (B*N, L)

            # --- Forward ---
            q_vals_flat, _ = self.agent(model_input, hidden_flat, agent_id=ids_seq)
            with torch.no_grad():
                target_q_vals_flat, _ = self.target_agent(target_model_input, target_hidden_flat, agent_id=ids_seq)

            # Reshape back to (B, N, L, n_actions)
            q_vals = q_vals_flat.view(B, N, L, -1)
            target_q_vals = target_q_vals_flat.view(B, N, L, -1)
            
            # Select Actions: actions is (B, L, N) -> need (B, N, L) for gather
            actions_ind = actions.permute(0, 2, 1).unsqueeze(-1)
            
            q_values_selected = q_vals.gather(-1, actions_ind).squeeze(-1) # (B, N, L)
            
            # Target Max
            max_target_q = target_q_vals.max(dim=-1)[0] # (B, N, L)
            
            # VDN Sum: Sum local Qs to get Q_tot
            q_tot = q_values_selected.sum(dim=1).unsqueeze(-1) # (B, L, 1)
            target_q_tot = max_target_q.sum(dim=1).unsqueeze(-1)
            
            total_reward = rewards.sum(dim=2).unsqueeze(-1) # (B, L, 1)

        else:
            # Independent VDN
            all_q_values = []
            all_target_q_values = []
            all_rewards = []
            
            for i in range(self.n_agents):
                # Slicing for Agent i: (B, L, N, ...) -> (B, L, ...)
                obs_i = {
                    'rgb': sample['all_agents']['obs']['rgb'][:, :, i],
                    'vector': sample['all_agents']['obs']['vector'][:, :, i] if self.vector_dim > 0 else None
                }
                next_obs_i = {
                    'rgb': sample['all_agents']['next_obs']['rgb'][:, :, i],
                    'vector': sample['all_agents']['next_obs']['vector'][:, :, i] if self.vector_dim > 0 else None
                }
                
                act_i = sample['all_agents']['actions'][:, :, i].long().unsqueeze(-1)
                rew_i = sample['all_agents']['rewards'][:, :, i].unsqueeze(-1)
                all_rewards.append(rew_i)

                h_i = init_hidden[:, :, i, :].contiguous()
                target_h_i = target_init_hidden[:, :, i, :].contiguous()

                q_vals, _ = self.agents[i](obs_i, h_i)
                q_sel = q_vals.gather(-1, act_i)
                all_q_values.append(q_sel)
                
                with torch.no_grad():
                    t_q_vals, _ = self.target_agents[i](next_obs_i, target_h_i)
                    max_t = t_q_vals.max(dim=-1, keepdim=True)[0]
                    all_target_q_values.append(max_t)

            q_tot = torch.stack(all_q_values).sum(dim=0) # (B, L, 1)
            target_q_tot = torch.stack(all_target_q_values).sum(dim=0)
            total_reward = torch.stack(all_rewards).sum(dim=0)

        # Compute Loss
        target = total_reward + self.gamma * (1 - global_dones) * target_q_tot
        td_error = (q_tot - target.detach()) ** 2
        masked_loss = td_error * mask
        loss = masked_loss.sum() / mask_sum

        self.optimizer.zero_grad()
        loss.backward()
        params = self.agent.parameters() if self.share_params else self.agents.parameters()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        self.optimizer.step()

        self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        """
        Perform soft update of target network parameters towards current network parameters.
        Formula: target_param = tau * local_param + (1 - tau) * target_param
        """
        if self.share_params:
            for target_param, local_param in zip(self.target_agent.parameters(), self.agent.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:
            for i in range(self.n_agents):
                for target_param, local_param in zip(self.target_agents[i].parameters(), self.agents[i].parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        """
        Save the model parameters to a file.

        Args:
            path (str): The file path to save the model weights.
        """
        if self.share_params:
            torch.save(self.agent.state_dict(), path)
        else:
            torch.save(self.agents.state_dict(), path)

    def load(self, path):
        """
        Load the model parameters from a file.

        Args:
            path (str): The file path to load the model weights from.
        """
        if self.share_params:
            self.agent.load_state_dict(torch.load(path))
            self.target_agent.load_state_dict(self.agent.state_dict())
        else:
            self.agents.load_state_dict(torch.load(path))
            self.target_agents.load_state_dict(self.agents.state_dict())

    def train(self):
        if self.share_params:
            self.agent.train()
        else:
            self.agents.train()

    def eval(self):
        if self.share_params:
            self.agent.eval()
        else:
            self.agents.eval()