import numpy as np
import torch

class ParallelReplayBuffer:
    """
    Parallel Replay Buffer for Melting Pot.
    Supports storing BOTH 'state' (flattened obs) AND 'global_state' (world rgb).
    Optimized for Memory: Stores RGB/State as uint8, returns as float32.
    """
    def __init__(self, num_envs, env_info, capacity, batch_size, sequence_length, hidden_dim, rnn_layers=1, device='cpu'):
        """
        Initialize the ParallelReplayBuffer.

        Args:
            num_envs (int): Number of parallel environments.
            env_info (dict): Dictionary containing environment information.
                - n_agents (int): Number of agents.
                - obs_shape (dict): Dictionary of observation shapes (e.g., {'rgb': (..., ...)}).
                - state_shape (tuple, optional): Shape of the flattened global state.
                - global_state_shape (tuple, optional): Shape of the world RGB global state.
            capacity (int): Maximum capacity of the buffer per environment.
            batch_size (int): Number of sequences to sample in a batch.
            sequence_length (int): Length of the sampled sequences (Time steps).
            hidden_dim (int): Dimensionality of the RNN hidden state.
            rnn_layers (int, optional): Number of RNN layers. Defaults to 1.
            device (str, optional): Device to store the sampled tensors (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        """
        self.num_envs = num_envs
        self.n_agents = env_info["n_agents"]
        self.obs_shape_dict = env_info["obs_shape"]
        
        self.state_shape = env_info.get("state_shape")
        self.global_state_shape = env_info.get("global_state_shape")
        
        # print(f"[Buffer] Init. State: {self.state_shape} | Global State: {self.global_state_shape}")

        self.capacity = capacity
        self.batch_size = batch_size
        self.seq_len = sequence_length
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.device = device

        # === Storage ===
        self.obs = {}
        self.next_obs = {}
        
        for key, shape in self.obs_shape_dict.items():
            dim = shape if isinstance(shape, tuple) else (shape,)
            if np.prod(dim) > 0:
                dtype = np.uint8 if key == 'rgb' else np.float32
                self.obs[key] = np.zeros((capacity, num_envs, self.n_agents, *dim), dtype=dtype)
                self.next_obs[key] = np.zeros((capacity, num_envs, self.n_agents, *dim), dtype=dtype)
            else:
                self.obs[key] = None; self.next_obs[key] = None

        # --- State Storage (Flatten) ---
        if self.state_shape:
            self.state = np.zeros((capacity, num_envs, *self.state_shape), dtype=np.uint8)
            self.next_state = np.zeros((capacity, num_envs, *self.state_shape), dtype=np.uint8)
        else:
            self.state = None; self.next_state = None

        # --- Global State Storage (World RGB) ---
        if self.global_state_shape:
            self.global_state = np.zeros((capacity, num_envs, *self.global_state_shape), dtype=np.uint8)
            self.next_global_state = np.zeros((capacity, num_envs, *self.global_state_shape), dtype=np.uint8)
        else:
            self.global_state = None; self.next_global_state = None
        
        self.hidden = np.zeros((capacity, num_envs, self.n_agents, self.rnn_layers, self.hidden_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_envs, self.n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, num_envs, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs, self.n_agents), dtype=np.float32)
        self.global_dones = np.zeros((capacity, num_envs), dtype=bool)

        self.ptr = 0; self.size = 0
        self.agent_ids_template = torch.arange(self.n_agents, device=device).reshape(1, 1, -1)

    def push(self, obs, hidden, state, global_state, actions, rewards, dones, next_obs, next_state, next_global_state):
        """
        Store a new transition layout from parallel environments into the buffer.
        
        Args:
            obs (dict): Dictionary of observations for the current step.
                - Keys define observation types (e.g., 'rgb').
                - Values are numpy arrays of shape (num_envs, n_agents, *obs_shape).
            hidden (np.ndarray): RNN hidden states at the current step.
                - Shape: (num_envs, n_agents, rnn_layers, hidden_dim).
            state (np.ndarray or None): Flattened global state at the current step (Optional).
                - Shape: (num_envs, *state_shape).
            global_state (np.ndarray or None): World RGB global state at the current step (Optional).
                - Shape: (num_envs, *global_state_shape).
            actions (np.ndarray): Actions taken by agents.
                - Shape: (num_envs, n_agents).
            rewards (np.ndarray): Rewards received by agents.
                - Shape: (num_envs, n_agents).
            dones (np.ndarray): Done flags for agents indicating episode termination.
                - Shape: (num_envs, n_agents).
            next_obs (dict): Dictionary of observations for the next step. Same structure as `obs`.
            next_state (np.ndarray or None): Flattened global state at the next step. Same structure as `state`.
            next_global_state (np.ndarray or None): World RGB global state at the next step. Same structure as `global_state`.
        """
        
        # Store Obs
        for key in self.obs:
            if self.obs[key] is not None and key in obs:
                data = obs[key]
                if key == 'rgb': data = (data * 255.0).astype(np.uint8)
                self.obs[key][self.ptr] = data
        
        for key in self.next_obs:
            if self.next_obs[key] is not None and key in next_obs:
                data = next_obs[key]
                if key == 'rgb': data = (data * 255.0).astype(np.uint8)
                self.next_obs[key][self.ptr] = data
        
        # Store State (Flatten)
        if self.state is not None and state is not None: 
            self.state[self.ptr] = (state * 255.0).astype(np.uint8)
        if self.next_state is not None and next_state is not None: 
            self.next_state[self.ptr] = (next_state * 255.0).astype(np.uint8)

        # Store Global State (World RGB)
        if self.global_state is not None and global_state is not None:
            self.global_state[self.ptr] = (global_state * 255.0).astype(np.uint8)
        if self.next_global_state is not None and next_global_state is not None:
            self.next_global_state[self.ptr] = (next_global_state * 255.0).astype(np.uint8)
            
        self.hidden[self.ptr] = hidden
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.global_dones[self.ptr] = np.any(dones, axis=1)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        """
        Sample a batch of sequences from the buffer for training.

        Returns:
            dict: A batch dictionary containing training data on the specified device.
            
            Structure:
            {
                'global': {
                    'mask': (torch.Tensor) Mask for valid time steps. Shape: (Batch_Size, Seq_Len, 1).
                    'dones': (torch.Tensor) Global done flags. Shape: (Batch_Size, Seq_Len, 1).
                    'state': (torch.Tensor, Optional) Flattened Global State. Shape: (Batch_Size, Seq_Len, *state_shape).
                    'next_state': (torch.Tensor, Optional) Next Flattened Global State. Shape: (Batch_Size, Seq_Len, *state_shape).
                    'global_state': (torch.Tensor, Optional) World RGB Global State. Shape: (Batch_Size, Seq_Len, *global_state_shape).
                    'next_global_state': (torch.Tensor, Optional) Next World RGB Global State. Shape: (Batch_Size, Seq_Len, *global_state_shape).
                },
                'all_agents': {
                    'obs': (dict) Dictionary of observation tensors. Shape: (Batch_Size, Seq_Len, n_agents, *obs_shape).
                    'next_obs': (dict) Dictionary of next observation tensors. Shape: (Batch_Size, Seq_Len, n_agents, *obs_shape).
                    'actions': (torch.Tensor) Agent actions. Shape: (Batch_Size, Seq_Len, n_agents).
                    'rewards': (torch.Tensor) Agent rewards. Shape: (Batch_Size, Seq_Len, n_agents).
                    'dones': (torch.Tensor) Agent done flags. Shape: (Batch_Size, Seq_Len, n_agents).
                    'agent_ids': (torch.Tensor) Agent identifiers. Shape: (Batch_Size, Seq_Len, n_agents).
                    'init_hidden': (torch.Tensor) Initial RNN hidden states for the sequence. Shape: (rnn_layers, Batch_Size, n_agents, hidden_dim).
                    'init_target_hidden': (torch.Tensor) Initial target RNN hidden states. Shape: (rnn_layers, Batch_Size, n_agents, hidden_dim).
                }
            }
            Returns None if the buffer contains fewer transitions than `sequence_length`.
        """
        if self.size < self.seq_len: return None 

        # ... (Sampling Index Selection Logic - Same as before) ...
        valid_indices = []; valid_envs = []; needed = self.batch_size
        while len(valid_indices) < needed:
            remaining = needed - len(valid_indices)
            cand_time = np.random.randint(0, self.size, size=remaining)
            cand_env = np.random.randint(0, self.num_envs, size=remaining)
            is_contiguous = (cand_time + self.seq_len <= self.capacity)
            if self.size == self.capacity:
                crosses_ptr = (cand_time <= self.ptr) & (cand_time + self.seq_len > self.ptr)
                is_valid = is_contiguous & (~crosses_ptr)
            else:
                is_valid = (cand_time + self.seq_len <= self.ptr)
            if np.any(is_valid):
                valid_indices.extend(cand_time[is_valid])
                valid_envs.extend(cand_env[is_valid])
        
        idxs = np.array(valid_indices[:self.batch_size])
        next_idxs = (idxs + 1) % self.capacity
        env_idxs = np.array(valid_envs[:self.batch_size])
        seq_time_idxs = idxs[:, None] + np.arange(self.seq_len)[None, :]
        seq_env_idxs = env_idxs[:, None].repeat(self.seq_len, axis=1)
        
        # --- Retrieve Data ---
        batch_obs = {}; batch_next_obs = {}
        for key in self.obs:
            if self.obs[key] is not None:
                raw = self.obs[key][seq_time_idxs, seq_env_idxs]
                raw_next = self.next_obs[key][seq_time_idxs, seq_env_idxs]
                if key == 'rgb':
                    raw = raw.astype(np.float32) / 255.0
                    raw_next = raw_next.astype(np.float32) / 255.0
                batch_obs[key] = raw; batch_next_obs[key] = raw_next
        
        batch_actions = self.actions[seq_time_idxs, seq_env_idxs]
        batch_rewards = self.rewards[seq_time_idxs, seq_env_idxs]
        batch_dones = self.dones[seq_time_idxs, seq_env_idxs]
        
        raw_hidden = self.hidden[idxs, env_idxs]
        batch_hidden = np.transpose(raw_hidden, (2, 0, 1, 3))
        raw_next_hidden = self.hidden[next_idxs, env_idxs]
        batch_next_hidden = np.transpose(raw_next_hidden, (2, 0, 1, 3))

        batch_global_dones = self.global_dones[seq_time_idxs, seq_env_idxs]
        first_done_idx = np.argmax(batch_global_dones, axis=1)
        has_done = np.any(batch_global_dones, axis=1)
        valid_lens = np.where(has_done, first_done_idx + 1, self.seq_len)
        time_steps = np.arange(self.seq_len)[None, :]
        mask = (time_steps < valid_lens[:, None]).astype(np.float32)[:, :, None]
        inv_mask = (1.0 - mask).astype(bool).squeeze(-1)

        # Padding
        for key in batch_obs:
            batch_obs[key][inv_mask] = 0; batch_next_obs[key][inv_mask] = 0
        batch_actions[inv_mask] = 0; batch_rewards[inv_mask] = 0; batch_dones[inv_mask] = 0
        
        batch_agent_ids = self.agent_ids_template.expand(self.batch_size, self.seq_len, -1).to(self.device)

        batch = {
            'global': {
                'mask': torch.from_numpy(mask).to(self.device),
                'dones': torch.from_numpy(batch_global_dones.astype(np.float32)[:, :, None]).to(self.device)
            },
            'all_agents': {
                'obs': {k: torch.from_numpy(v).to(self.device) for k, v in batch_obs.items()},
                'next_obs': {k: torch.from_numpy(v).to(self.device) for k, v in batch_next_obs.items()},
                'actions': torch.from_numpy(batch_actions).to(self.device),
                'rewards': torch.from_numpy(batch_rewards).to(self.device),
                'dones': torch.from_numpy(batch_dones).to(self.device),
                'agent_ids': batch_agent_ids,
                'init_hidden': torch.from_numpy(batch_hidden).to(self.device),
                'init_target_hidden': torch.from_numpy(batch_next_hidden).to(self.device),
            }
        }

        # --- Retrieve State (Flatten) ---
        if self.state is not None:
            b_s = self.state[seq_time_idxs, seq_env_idxs].astype(np.float32) / 255.0
            b_ns = self.next_state[seq_time_idxs, seq_env_idxs].astype(np.float32) / 255.0
            b_s[inv_mask] = 0; b_ns[inv_mask] = 0
            batch['global']['state'] = torch.from_numpy(b_s).to(self.device)
            batch['global']['next_state'] = torch.from_numpy(b_ns).to(self.device)

        # --- Retrieve Global State (World RGB) ---
        if self.global_state is not None:
            b_gs = self.global_state[seq_time_idxs, seq_env_idxs].astype(np.float32) / 255.0
            b_ngs = self.next_global_state[seq_time_idxs, seq_env_idxs].astype(np.float32) / 255.0
            b_gs[inv_mask] = 0; b_ngs[inv_mask] = 0

            batch['global']['global_state'] = torch.from_numpy(b_gs).to(self.device)
            batch['global']['next_global_state'] = torch.from_numpy(b_ngs).to(self.device)

        return batch
    
    def __len__(self): 
        """
        Get the current number of transitions stored in the buffer (per environment).
        
        Returns:
            int: Current size of the buffer.
        """
        return self.size