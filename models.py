import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class AgentDRQN(nn.Module):
    """
    Agent network for processing local observations (RGB images) and outputting Q-values.
    Layout: CNN (features) → GRU (memory) → Linear (Q-values).
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, rnn_hidden_dim: int = 64):
        """
        Parameters:
        - obs_shape (Tuple): (C, H, W), e.g. (3, 40, 40)
        - n_actions (int): Action space dimension, e.g. 8
        - rnn_hidden_dim (int): GRU hidden layer size
        """
        super(AgentDRQN, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        # Input: (B * L * N, C, H, W) = (B * L * N, 3, 40, 40)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() 
        )
        
        # Dynamically computing the CNN output shape. 
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)   # Create a dummy input of shape (1, C, H, W)
            dummy_output = self.conv(dummy_input)      # Pass through CNN
            self.cnn_out_dim = dummy_output.shape[1]   # .shape[1] capturing the feature dimension post-flattening

        # 2. Recurrent layer (GRU)
        # Input: (B * L * N, cnn_out_dim)
        # GRU input needs (seq_len, batch, input_size) or (input_size) if batch_first=False
        # We will handle the shape in forward
        self.rnn = nn.GRUCell(self.cnn_out_dim, self.rnn_hidden_dim)

        # 3. Output layer (Q-values)
        # Input: (B * L * N, rnn_hidden_dim)
        self.fc_q = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def init_hidden(self, batch_size=1):
        """Initialize hidden state to zero."""
        return self.rnn.weight_ih.new(batch_size, self.rnn_hidden_dim).zero_() # input_to_hidden

    def forward(self, 
                obs_batch: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Can handle two input shapes:
        1. Sequence sampling: obs_batch (B, L, N, C, H, W), hidden_state (B, N, hidden_dim)
        2. Single-step sampling/execution: obs_batch (B, N, C, H, W), hidden_state (B, N, hidden_dim)

        Returns:
        - q_values (Tensor): (B, L, N, n_actions) or (B, N, n_actions)
        - next_hidden_state (Tensor): (B, N, hidden_dim) (Note: GRU returns the last time step's hidden state)
        """

        is_sequence = (len(obs_batch.shape) == 6)

        if is_sequence:
            B, L, N, C, H, W = obs_batch.shape
            # (B, L, N, C, H, W) -> (B * L * N, C, H, W)
            obs_flat = obs_batch.reshape(-1, C, H, W)
            # (B, N, hidden_dim) -> (B * N, hidden_dim) - Initial hidden state
            # (Note: RNN only needs initial hidden state, subsequent steps are handled internally)
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim) 
        else: # Single-step
            B, N, C, H, W = obs_batch.shape
            L = 1 # Sequence length is 1
            # (B, N, C, H, W) -> (B * N, C, H, W)
            obs_flat = obs_batch.reshape(-1, C, H, W)
            # (B, N, hidden_dim) -> (B * N, hidden_dim)
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
            
        # (B * L * N, C, H, W) -> (B * L * N, cnn_out_dim)
        cnn_features = self.conv(obs_flat)

        # GRUCell needs input (batch, input_size) and hidden (batch, hidden_size)
        # We need to loop over time steps
        # (B * L * N, cnn_out_dim) -> (B, L, N, cnn_out_dim) -> (L, B, N, cnn_out_dim)
        # -> (L, B * N, cnn_out_dim)
        rnn_input = cnn_features.reshape(B, L, N, -1).permute(1, 0, 2, 3).reshape(L, B * N, -1)
        
        gru_hiddens = []
        h_current = h_in
        for t in range(L):
            h_current = self.rnn(rnn_input[t], h_current)   # (B * N, hidden_dim)
            gru_hiddens.append(h_current)                   # Store hidden state at each time step

        # (L, B * N, hidden_dim) -> (B * N * L, hidden_dim)
        rnn_output_flat = torch.stack(gru_hiddens, dim=0).permute(1, 0, 2).reshape(B * L * N, -1)

        # GRUCell returns the last time step's hidden state
        # (B * N, hidden_dim) -> (B, N, hidden_dim)
        next_hidden_state = h_current.reshape(B, N, self.rnn_hidden_dim)

        # 3. Compute Q values
        # (B * L * N, hidden_dim) -> (B * L * N, n_actions)
        q_values_flat = self.fc_q(rnn_output_flat)

        # Restore shape
        if is_sequence:
            # (B * L * N, n_actions) -> (B, L, N, n_actions)
            q_values = q_values_flat.reshape(B, L, N, self.n_actions)
        else:
            # (B * N, n_actions) -> (B, N, n_actions)
            q_values = q_values_flat.reshape(B, N, self.n_actions)
            
        return q_values, next_hidden_state
    
class QMixer(nn.Module):
    """
    Receives all agents' Q-values and the global state, then produces Q_tot.
    """
    def __init__(self, n_agents: int, state_shape: int, mixing_embed_dim: int = 32):
        """
        Args:
        - n_agents (int): Number of agents
        - state_shape (int): Dimension of global state (after flattening)
        - mixing_embed_dim (int): Dimension of hidden layer in mixing network
        """
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.mixing_embed_dim = mixing_embed_dim

        # W1: state -> (n_agents * mixing_embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_shape, mixing_embed_dim * n_agents),
            nn.ReLU()
        )
        # b1: state -> (mixing_embed_dim)
        self.hyper_b1 = nn.Linear(self.state_shape, mixing_embed_dim)

        # W2: state -> (mixing_embed_dim) -> Relu -> (mixing_embed_dim * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim * 1) 
        )
        # b2: state -> (1)
        self.hyper_b2 = nn.Linear(self.state_shape, 1)

    def forward(self, 
                agent_q_values: torch.Tensor, 
                states: torch.Tensor) -> torch.Tensor:
        """
        Can handle two input shapes:
        1. Sequence sampling: agent_q_values (B, L, N), states (B, L, state_shape)
        2. Single-step sampling/execution: agent_q_values (B, N), states (B, state_shape)

        Returns:
        - q_total (Tensor): (B, L) or (B,)
        """

        is_sequence = (len(agent_q_values.shape) == 3)

        if is_sequence:
            B, L, N = agent_q_values.shape
            # (B, L, N) -> (B * L, N)
            q_vals_flat = agent_q_values.reshape(-1, self.n_agents)
            # (B, L, state_shape) -> (B * L, state_shape)
            states_flat = states.reshape(-1, self.state_shape)
        else: # Single-step
            B, N = agent_q_values.shape
            L = 1
            # (B, N)
            q_vals_flat = agent_q_values
            # (B, state_shape)
            states_flat = states

        batch_size_effective = B * L # Effective batch size

        # (B * L, N) -> (B * L, 1, N)
        q_vals_flat = q_vals_flat.view(batch_size_effective, 1, self.n_agents)

        # Generate W1 and b1 (weights must be positive, use abs)
        w1 = torch.abs(self.hyper_w1(states_flat))
        # (B * L, N * embed_dim) -> (B * L, N, embed_dim)
        w1 = w1.view(batch_size_effective, self.n_agents, self.mixing_embed_dim) 
        # (B * L, embed_dim) -> (B * L, 1, embed_dim)
        b1 = self.hyper_b1(states_flat).view(batch_size_effective, 1, self.mixing_embed_dim)

        # (B*L, 1, N) @ (B*L, N, embed_dim) -> (B*L, 1, embed_dim)
        hidden = F.elu(torch.bmm(q_vals_flat, w1) + b1)    # bmm: batch matrix multiplication -> (b, n, m) @ (b, m, p) = (b, n, p)

        # Generate W2 and b2 (weights must be positive, use abs)
        w2 = torch.abs(self.hyper_w2(states_flat))
        # (B * L, embed_dim * 1) -> (B * L, embed_dim, 1)
        w2 = w2.view(batch_size_effective, self.mixing_embed_dim, 1)
        # (B * L, 1) -> (B * L, 1, 1)
        b2 = self.hyper_b2(states_flat).view(batch_size_effective, 1, 1)

        # (B * L, 1, embed_dim) @ (B * L, embed_dim, 1) -> (B * L, 1, 1)
        q_total_flat = torch.bmm(hidden, w2) + b2
        
        # Restore shape
        if is_sequence:
            # (B*L, 1, 1) -> (B, L)
            q_total = q_total_flat.view(B, L)
        else:
            # (B, 1, 1) -> (B,)
            q_total = q_total_flat.view(B)
            
        return q_total
    

if __name__ == "__main__":
    
    print("="*40)
    print("Starting verification of the QMIX network architecture...")
    print("="*40)
    
    try:
        from MeltingPotWrapper import build_meltingpot_env as GeneralMeltingPotWrapper
        print("Successfully imported build_meltingpot_env from MeltingPotWrapper.")
        
        env = GeneralMeltingPotWrapper(env_name="collaborative_cooking__asymmetric")
        
        env_info = env.get_env_info()
        env.close() 
        print("Environment information loaded successfully:")
        print(f"  - n_agents: {env_info['n_agents']}")
        print(f"  - n_actions: {env_info['n_actions']}")
        print(f"  - obs_shape (C,H,W): {env_info['obs_shape']}")
        print(f"  - state_shape (flat): {env_info['state_shape']}")
    
    except ImportError:
        print("\n!!! Error: Unable to import build_meltingpot_env from MeltingPotWrapper.py.")
        print("!!! Please ensure MeltingPotWrapper.py is in the same directory or Python path.")
        env_info = {
            "n_agents": 2, "n_actions": 8,
            "obs_shape": (3, 40, 40), "state_shape": 9600
        }
        print("\nWarning: Using default dimensions to continue verification...")
    except Exception as e:
        print(f"\n!!! Error: Failed to load environment: {e}")
        env_info = {
            "n_agents": 2, "n_actions": 8,
            "obs_shape": (3, 40, 40), "state_shape": 9600
        }
        print("\nWarning: Using default dimensions to continue verification...")

    N_AGENTS = env_info['n_agents']
    N_ACTIONS = env_info['n_actions']
    OBS_SHAPE = env_info['obs_shape'] # (C, H, W)
    STATE_SHAPE = env_info['state_shape'] # int
    RNN_HIDDEN_DIM = 64
    MIXING_EMBED_DIM = 32

    agent_net = AgentDRQN(OBS_SHAPE, N_ACTIONS, RNN_HIDDEN_DIM)
    mixer_net = QMixer(N_AGENTS, STATE_SHAPE, MIXING_EMBED_DIM)

    print("\nNetwork instantiation successful:")

    B = 4 # Batch Size
    L = 8 # Sequence Length

    dummy_obs_seq = torch.rand(B, L, N_AGENTS, *OBS_SHAPE)
    dummy_state_seq = torch.rand(B, L, STATE_SHAPE)
    dummy_hidden_init = agent_net.init_hidden(B * N_AGENTS).reshape(B, N_AGENTS, RNN_HIDDEN_DIM)

    dummy_obs_step = torch.rand(B, N_AGENTS, *OBS_SHAPE)
    dummy_state_step = torch.rand(B, STATE_SHAPE)
    dummy_hidden_step = agent_net.init_hidden(B * N_AGENTS).reshape(B, N_AGENTS, RNN_HIDDEN_DIM)

    print("\nDummy input shapes:")
    print(f"  - dummy_obs_seq: \t\t{dummy_obs_seq.shape}")
    print(f"  - dummy_state_seq: \t\t{dummy_state_seq.shape}")
    print(f"  - dummy_hidden_init: \t\t{dummy_hidden_init.shape}")
    print(f"  - dummy_obs_step: \t\t{dummy_obs_step.shape}")
    print(f"  - dummy_state_step: \t\t{dummy_state_step.shape}")
    print(f"  - dummy_hidden_step: \t\t{dummy_hidden_step.shape}")

    print("\n--- Validate AgentDRQN ---")

    try:
        q_vals_seq, next_hidden_seq = agent_net(dummy_obs_seq, dummy_hidden_init)
        print("Sequence Input:")
        print(f"  - Input obs: {dummy_obs_seq.shape}")
        print(f"  - Input hidden: {dummy_hidden_init.shape}")
        print(f"  - Output q_vals: \t{q_vals_seq.shape} (Expected: {(B, L, N_AGENTS, N_ACTIONS)})")
        print(f"  - Output next_hidden: {next_hidden_seq.shape} (Expected: {(B, N_AGENTS, RNN_HIDDEN_DIM)})")
        assert q_vals_seq.shape == (B, L, N_AGENTS, N_ACTIONS)
        assert next_hidden_seq.shape == (B, N_AGENTS, RNN_HIDDEN_DIM)
    except Exception as e:
        print(f"!!! AgentDRQN Sequence Input failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        q_vals_step, next_hidden_step = agent_net(dummy_obs_step, dummy_hidden_step)
        print("\nStep Input:")
        print(f"  - Input obs: {dummy_obs_step.shape}")
        print(f"  - Input hidden: {dummy_hidden_step.shape}")
        print(f"  - Output q_vals: \t{q_vals_step.shape} (Expected: {(B, N_AGENTS, N_ACTIONS)})")
        print(f"  - Output next_hidden: {next_hidden_step.shape} (Expected: {(B, N_AGENTS, RNN_HIDDEN_DIM)})")
        assert q_vals_step.shape == (B, N_AGENTS, N_ACTIONS)
        assert next_hidden_step.shape == (B, N_AGENTS, RNN_HIDDEN_DIM)
    except Exception as e:
        print(f"!!! AgentDRQN Step Input failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Validate QMixer ---")

    dummy_actions_seq = torch.randint(0, N_ACTIONS, (B, L, N_AGENTS))
    chosen_q_vals_seq = torch.gather(q_vals_seq, dim=3, index=dummy_actions_seq.unsqueeze(-1)).squeeze(-1) 
    
    dummy_actions_step = torch.randint(0, N_ACTIONS, (B, N_AGENTS))
    chosen_q_vals_step = torch.gather(q_vals_step, dim=2, index=dummy_actions_step.unsqueeze(-1)).squeeze(-1)

    try:
        q_tot_seq = mixer_net(chosen_q_vals_seq, dummy_state_seq)
        print("Sequence Input:")
        print(f"  - Input agent_qs: {chosen_q_vals_seq.shape}")
        print(f"  - Input states: {dummy_state_seq.shape}")
        print(f"  - Output q_tot: \t{q_tot_seq.shape} (Expected: {(B, L)})")
        assert q_tot_seq.shape == (B, L)
    except Exception as e:
        print(f"!!! QMixer Sequence Input failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        q_tot_step = mixer_net(chosen_q_vals_step, dummy_state_step)
        print("\nStep Input:")
        print(f"  - Input agent_qs: {chosen_q_vals_step.shape}")
        print(f"  - Input states: {dummy_state_step.shape}")
        print(f"  - Output q_tot: \t{q_tot_step.shape} (Expected: {(B,)})")
        assert q_tot_step.shape == (B,)
    except Exception as e:
        print(f"!!! QMixer Step Input failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*40)
    print("✅ QMIX Network Structure Validation Completed!")
    print("="*40)

    