import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class AgentNetwork(nn.Module):
    """
    General Recurrent Agent Network - R-QMIX
    - Porcessing RGB, Vector, or both combined
    - Using GRUCell to manage hidden state for POMDP
    """
    def __init__(self, rgb_shape: Tuple[int, int, int], vector_dim, n_actions: int, gru_hidden_dim: int = 64):
        super(AgentNetwork, self).__init__()
        self.n_actions = n_actions
        self.use_rgb = rgb_shape is not None and np.prod(rgb_shape) > 0
        self.use_vector = vector_dim > 0
        self.gru_hidden_dim = gru_hidden_dim

        cnn_out_dim = 0
        vector_mlp_out_dim = 0

        if self.use_rgb:
            c, h, w,  = rgb_shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            with torch.no_grad():
                _dummy_input = torch.zeros(1, c, h, w)
                cnn_out_dim = self.cnn(_dummy_input).flatten().shape[0]
            print(f"[AgentNetwork] RGB (CNN) enabled. Output: {cnn_out_dim}")
        
        self.vector_dim = vector_dim
        if self.use_vector:
            self.vector_mlp = nn.Sequential(
                nn.Linear(self.vector_dim, 128), 
                nn.ReLU()
            )
            vector_mlp_out_dim = 128
            print(f"[AgentNetwork] Vector (MLP) enabled. Output: {vector_mlp_out_dim}")

        gru_input_dim = cnn_out_dim + vector_mlp_out_dim
        if gru_input_dim == 0:
            raise ValueError("AgentNetwork has no inputs!")
        
        self.gru = nn.GRUCell(gru_input_dim, self.gru_hidden_dim)
        print(f"[AgentNetwork] GRU enabled. Input: {gru_input_dim}, Hidden: {self.gru_hidden_dim}")

        self.mlp_out = nn.Linear(self.gru_hidden_dim, n_actions)

    def init_hidden(self, batch_size, device):
        """ Return an all-zero initial hidden state """
        return torch.zeros(batch_size, self.gru_hidden_dim).to(device)
    
    def forward(self, rgb_input, vector_input, h_in):
        """
        Processes observations, updates the recurrent state, and computes Q-values.

        Args:
            rgb_input (torch.Tensor): Batch of RGB observations. Shape: (B, C, H, W).
            vector_input (torch.Tensor): Batch of vector observations. Shape: (B, vector_dim).
            h_in (torch.Tensor): Previous hidden state. Shape: (B, gru_hidden_dim).

        Returns:
            q_values (torch.Tensor): Estimated Q-values for each action. Shape: (B, n_actions).
            h_out (torch.Tensor): New hidden state. Shape: (B, gru_hidden_dim).
        """
        b = h_in.size(0)
        input_parts = []

        if self.use_rgb:
            cnn_out = self.cnn(rgb_input)
            cnn_out_flat = cnn_out.view(b, -1)
            input_parts.append(cnn_out_flat)

        if self.use_vector:
            vector_out = self.vector_mlp(vector_input)
            input_parts.append(vector_out)

        x = torch.cat(input_parts, dim=1)
        h_out = self.gru(x, h_in)

        q_values = self.mlp_out(h_out)
        return q_values, h_out
    
class MixerNetwork(nn.Module):
    """
    Gereral Mixer Network for QMIX/R-QMIX
    - Porcessing RGB, Vector, or both combined as global state
    - Using hypernetworks to generate mixing weights and biases
    - Produces joint Q-value Q_tot
    """
    def __init__(self, n_agents, state_shape_dict, embed_dim=64):
        super(MixerNetwork, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        self.state_rgb_shape = state_shape_dict.get('rgb')       # (N, C, H, W)
        self.state_vector_shape = state_shape_dict.get('vector') # (N, V)

        self.use_rgb_state = self.state_rgb_shape is not None and np.prod(self.state_rgb_shape) > 0
        self.use_vector_state = self.state_vector_shape is not None and np.prod(self.state_vector_shape) > 0

        state_processor_out_dim = 0

        if self.use_rgb_state:
            n, c, h, w = self.state_rgb_shape
            cnn_input_channels = n * c
            self.state_cnn = nn.Sequential(
                nn.Conv2d(cnn_input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU()
            )
            with torch.no_grad():
                _dummy_input = torch.zeros(1, cnn_input_channels, h, w)
                cnn_out_dim = self.state_cnn(_dummy_input).flatten().shape[0]
            self.state_rgb_mlp = nn.Sequential(nn.Linear(cnn_out_dim, 128), nn.ReLU())
            state_processor_out_dim += 128
            print(f"[MixerNetwork] State RGB (CNN) enabled. Input: {(n, c, h, w)}, Output: 128")

        if self.use_vector_state:
            n, v = self.state_vector_shape
            vector_input_dim = n * v
            self.state_vector_mlp = nn.Sequential(nn.Linear(vector_input_dim, 128), nn.ReLU())
            state_processor_out_dim += 128
            print(f"[MixerNetwork] State Vector (MLP) enabled. Input: {(n, v)}, Output: 128")

        if state_processor_out_dim == 0:
            raise ValueError("MixerNetwork has no state inputs!")
        print(f"[MixerNetwork] Hypernetwork Input dim: {state_processor_out_dim}")

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_processor_out_dim, 256), nn.ReLU(),
            nn.Linear(256, self.n_agents * self.embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_processor_out_dim, self.embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_processor_out_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_processor_out_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, q_values, state_rgb, state_vector):
        """
        Mix individual agent Q-values into a joint Q_total conditioned on the global state.

        Args:
            q_values (torch.Tensor): Shape (B, N_agents). Per-agent Q estimates to mix.
            state_rgb (torch.Tensor | None): Shape (B, N, C, H, W). Global RGB state; required if RGB state processing is enabled.
            state_vector (torch.Tensor | None): Shape (B, N, V). Global vector state; required if vector state processing is enabled.

        Returns:
            q_total (torch.Tensor): Shape (B, 1). Mixed joint Q-value satisfying monotonicity.
            state_vec (torch.Tensor): Shape (B, state_processor_out_dim). Concatenated state embedding used by the hyper-networks.
        """
        b = q_values.size(0)
        state_parts = []

        if self.use_rgb_state:
            if state_rgb is None: raise ValueError("Mixer expected State RGB")
            n_channels = state_rgb.shape[1] * state_rgb.shape[2]
            rgb_in = state_rgb.view(b, n_channels, state_rgb.shape[3], state_rgb.shape[4])
            cnn_out = self.state_cnn(rgb_in)
            rgb_vec = self.state_rgb_mlp(cnn_out.view(b, -1))
            state_parts.append(rgb_vec)

        if self.use_vector_state:
            if state_vector is None: raise ValueError("Mixer expected State Vector")
            vec_in = state_vector.view(b, -1)
            vec_vec = self.state_vector_mlp(vec_in)
            state_parts.append(vec_vec)

        state_vec = torch.cat(state_parts, dim=1) 
        w1 = torch.abs(self.hyper_w1(state_vec)).view(b, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state_vec).view(b, 1, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(state_vec)).view(b, self.embed_dim, 1)
        b2 = self.hyper_b2(state_vec).view(b, 1, 1)
        q_values = q_values.view(b, 1, self.n_agents)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(b, 1)
        return q_total, state_vec

class VDNMixer(nn.Module):
    """
    A simple summation mixer for ablation studies.
    Sums up individual agent Q-values to produce Q_tot."""
    def __init__(self):
        super(VDNMixer, self).__init__()
    
    def forward(self,
                agent_q_values: torch.Tensor) -> torch.Tensor:
        """
        Can handle two input shapes:
        1. Sequence sampling: agent_q_values (B, L, N)
        2. Single-step sampling/execution: agent_q_values (B, N)
        """
        q_total = torch.sum(agent_q_values, dim=-1)
        return q_total

class ActorCriticDRQN(nn.Module):
    """
    Agent network that processes local observations (RGB images) and outputs
    both policy (actor) and state value (critic).
    Layout: CNN (features) -> GRU (memory) -> Linear (Actor)
                                            -> Linear (Critic)
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, rnn_hidden_dim: int = 64):
        """
        Parameters:
        - obs_shape (Tuple): (C, H, W), e.g. (3, 40, 40)
        - n_actions (int): Action space dimension, e.g. 8
        - rnn_hidden_dim (int): GRU hidden layer size
        """
        super(ActorCriticDRQN, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        # 1. Convolutional layers (Same as AgentDRQN)
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
        
        # Compute CNN output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_output = self.conv(dummy_input)
            self.cnn_out_dim = dummy_output.shape[1]

        # 2. Recurrent layer (GRU)
        self.rnn = nn.GRUCell(self.cnn_out_dim, self.rnn_hidden_dim)

        # 3. Output layers (Split into Actor and Critic)
        self.fc_actor = nn.Linear(self.rnn_hidden_dim, self.n_actions) # Actor head
        self.fc_critic = nn.Linear(self.rnn_hidden_dim, 1)             # Critic head

    def init_hidden(self, batch_size=1):
        """Initialize hidden state to zero."""
        return self.rnn.weight_ih.new(batch_size, self.rnn_hidden_dim).zero_()

    def forward(self, 
                obs_batch: torch.Tensor, 
                hidden_state: torch.Tensor,
                dones_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles three input shapes:
        1. Sequence (training): obs_batch (B, L, N, C, H, W), hidden_state (B, N, hidden_dim)
        2. Step (acting): obs_batch (B, N, C, H, W), hidden_state (B, N, hidden_dim)
        3. Step with done mask (reset hidden state on done): obs_batch (B, N, C, H, W), hidden_state (B, N, hidden_dim), dones_mask (B, N)

        Returns:
        - actor_logits (Tensor): (B, L, N, n_actions) or (B, N, n_actions)
        - state_values (Tensor): (B, L, N, 1) or (B, N, 1)
        - next_hidden_state (Tensor): (B, N, hidden_dim)
        """
        is_sequence = (len(obs_batch.shape) == 6)

        if is_sequence:
            B, L, N, C, H, W = obs_batch.shape
            obs_flat = obs_batch.reshape(-1, C, H, W)
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim) 
        else: # Single-step
            B, N, C, H, W = obs_batch.shape
            L = 1 
            obs_flat = obs_batch.reshape(-1, C, H, W)
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        
        # (B, L, N, 1) -> (L, B * N, 1)
        if dones_mask is None:
            dones_mask = torch.zeros(B, L, N, 1, device=obs_batch.device, dtype=torch.float32)
        # (B, L, N, 1) -> (L, B, N, 1) -> (L, B*N, 1)
        dones_mask_expanded = dones_mask.permute(1, 0, 2, 3).reshape(L, B * N, 1)
            
        # (B * L * N, C, H, W) -> (B * L * N, cnn_out_dim)
        cnn_features = self.conv(obs_flat)

        # (B * L * N, cnn_out_dim) -> (L, B * N, cnn_out_dim)
        rnn_input = cnn_features.reshape(B, L, N, -1).permute(1, 0, 2, 3).reshape(L, B * N, -1)
        
        gru_hiddens = []
        h_current = h_in
        for t in range(L):
            h_current = h_current * (1.0 - dones_mask_expanded[t])  # Reset hidden state where done
            h_current = self.rnn(rnn_input[t], h_current)
            gru_hiddens.append(h_current)

        # (L, B * N, hidden_dim) -> (B * L * N, hidden_dim)
        rnn_output_flat = torch.stack(gru_hiddens, dim=0).permute(1, 0, 2).reshape(B * L * N, -1)
        
        # (B * N, hidden_dim) -> (B, N, hidden_dim)
        next_hidden_state = h_current.reshape(B, N, self.rnn_hidden_dim)

        # 3. Compute Actor Logits and Critic Values
        actor_logits_flat = self.fc_actor(rnn_output_flat)
        state_values_flat = self.fc_critic(rnn_output_flat)

        # Restore shapes
        if is_sequence:
            actor_logits = actor_logits_flat.reshape(B, L, N, self.n_actions)
            state_values = state_values_flat.reshape(B, L, N, 1)
        else:
            actor_logits = actor_logits_flat.reshape(B, N, self.n_actions)
            state_values = state_values_flat.reshape(B, N, 1)
            
        return actor_logits, state_values, next_hidden_state

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
        print(f"  - state_shape (N,C,H,W): {env_info['state_shape']}")
    
    except ImportError:
        print("\n!!! Error: Unable to import build_meltingpot_env from MeltingPotWrapper.py.")
        print("!!! Please ensure MeltingPotWrapper.py is in the same directory or Python path.")
        env_info = {
            "n_agents": 2, "n_actions": 8,
            "obs_shape": (3, 40, 40), 
            "state_shape": (2, 3, 40, 40)
        }
        print("\nWarning: Using default dimensions to continue verification...")
    except Exception as e:
        print(f"\n!!! Error: Failed to load environment: {e}")
        env_info = {
            "n_agents": 2, "n_actions": 8,
            "obs_shape": (3, 40, 40), 
            "state_shape": (2, 3, 40, 40)
        }
        print("\nWarning: Using default dimensions to continue verification...")

    N_AGENTS = env_info['n_agents']
    N_ACTIONS = env_info['n_actions']
    OBS_SHAPE = env_info['obs_shape'] # (C, H, W)
    STATE_SHAPE = env_info['state_shape'] # (N, C, H, W)
    RNN_HIDDEN_DIM = 64
    MIXING_EMBED_DIM = 32

    agent_net = AgentDRQN(OBS_SHAPE, N_ACTIONS, RNN_HIDDEN_DIM)
    mixer_net = QMixer(N_AGENTS, STATE_SHAPE, MIXING_EMBED_DIM)

    print("\nNetwork instantiation successful:")

    B = 4 # Batch Size
    L = 8 # Sequence Length

    dummy_obs_seq = torch.rand(B, L, N_AGENTS, *OBS_SHAPE)
    dummy_state_seq = torch.rand(B, L, *STATE_SHAPE)
    dummy_hidden_init = agent_net.init_hidden(B * N_AGENTS).reshape(B, N_AGENTS, RNN_HIDDEN_DIM)

    dummy_obs_step = torch.rand(B, N_AGENTS, *OBS_SHAPE)
    dummy_state_step = torch.rand(B, *STATE_SHAPE)
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

    