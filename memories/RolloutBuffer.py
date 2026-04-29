import numpy as np
import torch

class ParallelRolloutBuffer:
    """
    Parallel Rollout Buffer for On-Policy MARL (IPPO/MAPPO).
    
    Features:
    1. Supports Dict Observations (RGB + Vector).
    2. Stores BOTH 'state' (Stacked Obs) and 'global_state' (World RGB).
    3. Memory Optimization: Stores images as uint8.
    4. RNN Support: Yields data chunks for recurrent training.
    """
    def __init__(self, num_envs, env_info, buffer_size, hidden_dim, rnn_layers=1, device='cpu'):
        """
        Initialize the ParallelRolloutBuffer.

        Args:
            num_envs (int): Number of parallel environments.
            env_info (dict): Dictionary containing environment information.
                - n_agents (int): Number of agents per environment.
                - obs_shape (dict): Observation shapes (e.g., {'rgb': (88, 88, 3)}).
                - state_shape (tuple, optional): Shape of the flattened global state.
                - global_state_shape (tuple, optional): Shape of the world RGB global state.
            buffer_size (int): Length of the rollout (number of steps to collect per environment).
            hidden_dim (int): Dimension of the RNN hidden state.
            rnn_layers (int, optional): Number of RNN layers. Defaults to 1.
            device (str, optional): Device to store tensors (e.g., 'cpu', 'cuda').
        """
        self.num_envs = num_envs
        self.n_agents = env_info["n_agents"]
        self.obs_shape_dict = env_info["obs_shape"]
        
        self.state_shape = env_info.get("state_shape")
        self.global_state_shape = env_info.get("global_state_shape")

        print(f"[RolloutBuffer] Init. State: {self.state_shape} | Global State: {self.global_state_shape}")

        self.buffer_size = buffer_size
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.device = device

        # === Storage Initialization ===
        
        # Obs (Dict support)
        self.obs = {}
        for key, shape in self.obs_shape_dict.items():
            dim = shape if isinstance(shape, tuple) else (shape,)
            if np.prod(dim) > 0:
                dtype = np.uint8 if key == 'rgb' else np.float32
                self.obs[key] = np.zeros((buffer_size + 1, num_envs, self.n_agents, *dim), dtype=dtype)
            else:
                self.obs[key] = None

        # State (Stacked Obs)
        if self.state_shape:
            dtype = np.uint8 if len(self.state_shape) >= 3 else np.float32
            self.state = np.zeros((buffer_size + 1, num_envs, *self.state_shape), dtype=dtype)
        else:
            self.state = None

        # Global State (World RGB)
        if self.global_state_shape:
            self.global_state = np.zeros((buffer_size + 1, num_envs, *self.global_state_shape), dtype=np.uint8)
        else:
            self.global_state = None
        
        # RNN Hidden States
        self.hidden_states = np.zeros((buffer_size + 1, num_envs, self.n_agents, self.rnn_layers, self.hidden_dim), dtype=np.float32)

        # Actions, Rewards, Dones, LogProbs, Values
        self.actions = np.zeros((buffer_size, num_envs, self.n_agents), dtype=np.int64) 
        self.rewards = np.zeros((buffer_size, num_envs, self.n_agents), dtype=np.float32)
        
        self.dones = np.zeros((buffer_size + 1, num_envs, self.n_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size + 1, num_envs, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs, self.n_agents), dtype=np.float32)

        self.agent_ids_template = torch.arange(self.n_agents, device=device).reshape(1, 1, -1)
        self.step = 0

    def is_full(self):
        """
        Check if the buffer is full.

        Returns:
            bool: True if the current step equals buffer_size, else False.
        """
        return self.step >= self.buffer_size
        
    def push(self, obs, state, global_state, hidden_states, actions, rewards, dones, log_probs, values):
        """
        Store a transition step into the buffer.

        Args:
            obs (dict): Dictionary of observations. Values shape: (num_envs, n_agents, *obs_shape).
            state (np.ndarray or None): Flattened global state. Shape: (num_envs, *state_shape).
            global_state (np.ndarray or None): World RGB global state. Shape: (num_envs, *global_state_shape).
            hidden_states (np.ndarray): RNN hidden states. Shape: (num_envs, n_agents, rnn_layers, hidden_dim).
            actions (np.ndarray): Actions taken. Shape: (num_envs, n_agents).
            rewards (np.ndarray): Rewards received. Shape: (num_envs, n_agents).
            dones (np.ndarray): Done flags (True if episode finished). Shape: (num_envs, n_agents).
            log_probs (np.ndarray): Log probabilities of actions. Shape: (num_envs, n_agents).
            values (np.ndarray): Critic value estimates. Shape: (num_envs, n_agents).
        """
        if self.step >= self.buffer_size:
            raise IndexError("Rollout Buffer is full!")

        for key in self.obs:
            if self.obs[key] is not None and key in obs:
                data = obs[key]
                if key == 'rgb':
                    data = (data * 255.0).astype(np.uint8)
                self.obs[key][self.step] = data

        if self.state is not None and state is not None:
            data = state
            if self.state.dtype == np.uint8:
                data = (data * 255.0).astype(np.uint8)
            self.state[self.step] = data
        
        if self.global_state is not None and global_state is not None:
            data = global_state
            if self.global_state.dtype == np.uint8:
                data = (data * 255.0).astype(np.uint8)
            self.global_state[self.step] = data
            
        self.hidden_states[self.step] = hidden_states 
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        
        self.step += 1

    def insert_last_step(self, obs, state, global_state, hidden_states, values, dones):
        """ Store the T+1 step data for GAE bootstrap. """
        idx = self.buffer_size
        
        # Obs
        for key in self.obs:
            if self.obs[key] is not None and key in obs:
                data = obs[key]
                if key == 'rgb':
                    data = (data * 255.0).astype(np.uint8)
                self.obs[key][idx] = data

        # State
        if self.state is not None and state is not None:
            data = state
            if self.state.dtype == np.uint8:
                data = (data * 255.0).astype(np.uint8)
            self.state[idx] = data

        # Global State
        if self.global_state is not None and global_state is not None:
            data = global_state
            if self.global_state.dtype == np.uint8:
                data = (data * 255.0).astype(np.uint8)
            self.global_state[idx] = data

        self.hidden_states[idx] = hidden_states
        self.values[idx] = values
        self.dones[idx] = dones

    def get_data(self):
        """
        Retrieve all stored data for GAE calculation or debugging.

        Returns:
            data (dict): Dictionary containing numpy arrays of all stored data.
                - obs: Dict of arrays (T, num_envs, n_agents, ...)
                - state: (T, num_envs, ...)
                - global_state: (T, num_envs, ...)
                - hidden_states: (T, num_envs, n_agents, ...)
                - actions: (T, ...)
                - rewards: (T, ...)
                - dones: (T, ...)
                - log_probs: (T, ...)
                - values: (T+1, ...)  <-- Includes bootstrap value
                - masks: (T, ...)
        """
        T = self.buffer_size
        
        # Handle Dict Obs Slicing
        obs_data = {}
        for k, v in self.obs.items():
            if v is not None:
                obs_data[k] = v[:T]

        data = {
            'obs': obs_data,
            'hidden_states': self.hidden_states[:T],
            'actions': self.actions[:T],
            'rewards': self.rewards[:T],
            'dones': self.dones[:T],
            'log_probs': self.log_probs[:T],
            'values': self.values[:T + 1],  # Return T+1 for GAE bootstrap
            'masks': 1.0 - self.dones[:T],
        }

        if self.state is not None:
            data['state'] = self.state[:T]
            
        if self.global_state is not None:
            data['global_state'] = self.global_state[:T]
            
        return data

    def recurrent_generator(self, advantages, returns, num_mini_batch, data_chunk_length):
        """
        Generator that yields training batches with RNN data chunking.

        Args:
            advantages (np.ndarray): GAE Advantages. Shape: (T, num_envs, n_agents).
            returns (np.ndarray): Calculated Returns (Advantages + Values). Shape: (T, num_envs, n_agents).
            num_mini_batch (int): Number of mini-batches to split the epoch into.
            data_chunk_length (int): Length of the time chunks for RNN training.

        Yields:
            tuple: A tuple containing PyTorch tensors on the specified device:
                - mb_obs (dict): Observations (Batch, Chunk_Len, Agents, *Obs_Shape).
                - mb_state (Tensor or None): Flattened State (Batch, Chunk_Len, Agents, *State_Shape).
                - mb_global_state (Tensor or None): World RGB (Batch, Chunk_Len, Agents, *Global_Shape).
                - mb_hidden (Tensor): RNN Hidden States (Layers, Batch, Agents, Hidden_Dim).
                - mb_actions (Tensor): Actions (Batch, Chunk_Len, Agents).
                - mb_values (Tensor): Value Estimates (Batch, Chunk_Len, Agents).
                - mb_returns (Tensor): Returns (Batch, Chunk_Len, Agents).
                - mb_log_probs (Tensor): Log Probs (Batch, Chunk_Len, Agents).
                - mb_advantages (Tensor): Advantages (Batch, Chunk_Len, Agents).
                - mb_masks (Tensor): Masks (0 if Done, else 1) (Batch, Chunk_Len, Agents).
                - mb_agent_ids (Tensor): Agent Identifiers (Batch, Chunk_Len, Agents).
        """
        T = self.buffer_size
        assert T % data_chunk_length == 0, f"Buffer size {T} must be divisible by chunk length {data_chunk_length}"

        num_chunks_per_env = T // data_chunk_length
        total_chunks = num_chunks_per_env * self.num_envs
        
        # === Helper: Reshape (T, Envs, Agents, ...) -> (Total_Chunks, Chunk_Len, Agents, ...) ===
        def _reshape_to_chunks(x):
            s = x.shape
            x = x[:T] 
            x_reshaped = x.reshape(num_chunks_per_env, data_chunk_length, self.num_envs, self.n_agents, *s[3:])
            dims = list(range(len(x_reshaped.shape)))
            perm = [0, 2, 1, 3] + dims[4:] 
            x_permuted = x_reshaped.transpose(*perm)
            return x_permuted.reshape(total_chunks, data_chunk_length, self.n_agents, *s[3:])

        # === Helper: Reshape Shared State (T, Envs, ...) -> (Total_Chunks, Chunk_Len, Agents, ...) ===
        def _reshape_shared_state_to_chunks(x):
            s = x.shape 
            x = x[:T]
            x_reshaped = x.reshape(num_chunks_per_env, data_chunk_length, self.num_envs, *s[2:])
            dims = list(range(len(x_reshaped.shape)))
            perm = [0, 2, 1] + dims[3:]
            x_permuted = x_reshaped.transpose(*perm)
            x_flat = x_permuted.reshape(total_chunks, data_chunk_length, *s[2:])
            return np.repeat(x_flat[:, :, np.newaxis, ...], self.n_agents, axis=2)

        # --- Reshape All Data ---

        batch_obs = {}
        for k, v in self.obs.items():
            if v is not None:
                batch_obs[k] = _reshape_to_chunks(v)

        batch_state = None
        if self.state is not None:
            s_chunks = _reshape_shared_state_to_chunks(self.state)
            if self.state.dtype == np.uint8:
                batch_state = s_chunks.astype(np.float32) / 255.0
            else:
                batch_state = s_chunks

        batch_global_state = None
        if self.global_state is not None:
            gs_chunks = _reshape_shared_state_to_chunks(self.global_state)
            if self.global_state.dtype == np.uint8:
                batch_global_state = gs_chunks.astype(np.float32) / 255.0
            else:
                batch_global_state = gs_chunks

        batch_actions = _reshape_to_chunks(self.actions)
        batch_log_probs = _reshape_to_chunks(self.log_probs)
        batch_values = _reshape_to_chunks(self.values[:T])
        batch_returns = _reshape_to_chunks(returns)
        batch_advantages = _reshape_to_chunks(advantages)
        batch_masks = _reshape_to_chunks(1.0 - self.dones[:T])

        chunk_indices = np.arange(0, T, data_chunk_length)
        start_hidden = self.hidden_states[chunk_indices] 
        batch_hidden_raw = start_hidden.reshape(total_chunks, self.n_agents, self.rnn_layers, self.hidden_dim)

        # --- Shuffle and Yield ---
        indices = np.arange(total_chunks)
        np.random.shuffle(indices)
        
        mini_batch_size = total_chunks // num_mini_batch

        for i in range(0, total_chunks, mini_batch_size):
            mb_indices = indices[i : i + mini_batch_size]
            
            mb_obs = {}
            for k, v in batch_obs.items():
                data = v[mb_indices] 
                if k == 'rgb': data = data.astype(np.float32) / 255.0
                mb_obs[k] = torch.from_numpy(data).to(self.device)

            mb_state = None
            if batch_state is not None:
                mb_state = torch.from_numpy(batch_state[mb_indices]).to(self.device)

            mb_global_state = None
            if batch_global_state is not None:
                mb_global_state = torch.from_numpy(batch_global_state[mb_indices]).to(self.device)

            mb_hidden = batch_hidden_raw[mb_indices]
            mb_hidden = torch.from_numpy(mb_hidden).to(self.device).permute(2, 0, 1, 3).contiguous() 

            curr_bs = len(mb_indices)
            mb_agent_ids = self.agent_ids_template.expand(curr_bs, data_chunk_length, -1).to(self.device)

            yield (
                mb_obs,
                mb_state,
                mb_global_state,
                mb_hidden,
                torch.from_numpy(batch_actions[mb_indices]).to(self.device),
                torch.from_numpy(batch_values[mb_indices]).to(self.device),
                torch.from_numpy(batch_returns[mb_indices]).to(self.device),
                torch.from_numpy(batch_log_probs[mb_indices]).to(self.device),
                torch.from_numpy(batch_advantages[mb_indices]).to(self.device),
                torch.from_numpy(batch_masks[mb_indices]).to(self.device),
                mb_agent_ids
            )

    def clear(self):
        """ Reset buffer step. """
        self.step = 0