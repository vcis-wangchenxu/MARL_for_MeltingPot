import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class AgentNetwork(nn.Module):
    """
    General Recurrent Agent Network - R-QMIX
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
    
    def forward(self, 
                rgb_input: torch.Tensor,    # (B, L, N, C, H, W) or (B, N, C, H, W)
                vector_input: torch.Tensor, # (B, L, N, V) or (B, N, V)
                h_in: torch.Tensor):        # (B, N, H_dim)
        """
        Processes observations, updates the recurrent state, and computes Q-values.
        """
        is_sequence = (len(rgb_input.shape) == 6)

        if is_sequence:
            B, L, N, C, H, W = rgb_input.shape
            rgb_flat = rgb_input.reshape(-1, C, H, W)
            vector_flat = vector_input.reshape(-1, self.vector_dim) if self.use_vector else None
            h_in_flat = h_in.reshape(-1, self.gru_hidden_dim)
        else: # Single-step (acting)
            B, N, C, H, W = rgb_input.shape
            L = 1
            rgb_flat = rgb_input.reshape(-1, C, H, W)
            vector_flat = vector_input.reshape(-1, self.vector_dim) if self.use_vector else None
            h_in_flat = h_in.reshape(-1, self.gru_hidden_dim)

        input_parts = []
        if self.use_rgb:
            cnn_out = self.cnn(rgb_flat)
            cnn_out_flat = cnn_out.reshape(B * L * N, -1)
            input_parts.append(cnn_out_flat)

        if self.use_vector:
            vector_out = self.vector_mlp(vector_flat)
            input_parts.append(vector_out)

        x_flat = torch.cat(input_parts, dim=1)

        rnn_input = x_flat.reshape(B, L, N, -1).permute(1, 0, 2, 3).reshape(L, B * N, -1)
        
        gru_hiddens = []
        h_current = h_in_flat
        for t in range(L):
            h_current = self.gru(rnn_input[t], h_current)
            gru_hiddens.append(h_current)

        rnn_output_flat = torch.stack(gru_hiddens, dim=0).permute(1, 0, 2).reshape(B * L * N, -1)
        next_hidden_state = h_current.reshape(B, N, self.gru_hidden_dim)

        q_values_flat = self.mlp_out(rnn_output_flat)

        if is_sequence:
            q_values = q_values_flat.reshape(B, L, N, self.n_actions)
        else:
            q_values = q_values_flat.reshape(B, N, self.n_actions)
            
        return q_values, next_hidden_state
    
class MixerNetwork(nn.Module):
    """
    Gereral Mixer Network for QMIX/R-QMIX
    """
    def __init__(self, n_agents, state_shape_dict, embed_dim=64):
        super(MixerNetwork, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        self.state_rgb_shape = state_shape_dict.get('rgb')
        self.state_vector_shape = state_shape_dict.get('vector')

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

    def forward(self, 
                q_values: torch.Tensor,     # (B, L, N) or (B, N)
                state_rgb: torch.Tensor,    # (B, L, N, C, H, W) or (B, N, C, H, W)
                state_vector: torch.Tensor):# (B, L, N, V) or (B, N, V)
        """
        Mix individual agent Q-values into a joint Q_total conditioned on the global state.
        """
        is_sequence = (len(q_values.shape) == 3)

        if is_sequence:
            B, L, N = q_values.shape
            q_values_flat = q_values.reshape(-1, N)
            rgb_flat = state_rgb.reshape(-1, N, self.state_rgb_shape[1], self.state_rgb_shape[2], self.state_rgb_shape[3])
            vector_flat = state_vector.reshape(-1, N, self.state_vector_shape[1]) if self.use_vector_state else None
        else: # Single-step
            B, N = q_values.shape
            L = 1
            q_values_flat = q_values
            rgb_flat = state_rgb
            vector_flat = state_vector
        
        B_flat = q_values_flat.size(0)

        state_parts = []
        if self.use_rgb_state:
            if rgb_flat is None: raise ValueError("Mixer expected State RGB")
            n, c, h, w = self.state_rgb_shape
            
            rgb_in = rgb_flat.reshape(B_flat, n * c, h, w)
            cnn_out = self.state_cnn(rgb_in)

            rgb_vec = self.state_rgb_mlp(cnn_out.reshape(B_flat, -1))
            state_parts.append(rgb_vec)

        if self.use_vector_state:
            if vector_flat is None: raise ValueError("Mixer expected State Vector")
            
            vec_in = vector_flat.reshape(B_flat, -1)
            vec_vec = self.state_vector_mlp(vec_in)
            state_parts.append(vec_vec)

        state_vec = torch.cat(state_parts, dim=1) 
        
        w1 = torch.abs(self.hyper_w1(state_vec)).reshape(B_flat, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state_vec).reshape(B_flat, 1, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(state_vec)).reshape(B_flat, self.embed_dim, 1)
        b2 = self.hyper_b2(state_vec).reshape(B_flat, 1, 1)
        
        q_values_bmm = q_values_flat.reshape(B_flat, 1, self.n_agents)
        
        hidden = F.elu(torch.bmm(q_values_bmm, w1) + b1)
        q_total_flat = torch.bmm(hidden, w2) + b2
        
        if is_sequence:
            q_total = q_total_flat.reshape(B, L, 1)
        else:
            q_total = q_total_flat.reshape(B, 1)
            
        return q_total, state_vec

class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()
    
    def forward(self,
                agent_q_values: torch.Tensor) -> torch.Tensor:
        q_total = torch.sum(agent_q_values, dim=-1)
        return q_total

class ActorCriticDRQN(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, rnn_hidden_dim: int = 64):
        super(ActorCriticDRQN, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() 
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_output = self.conv(dummy_input)
            self.cnn_out_dim = dummy_output.shape[1]

        self.rnn = nn.GRUCell(self.cnn_out_dim, self.rnn_hidden_dim)
        self.fc_actor = nn.Linear(self.rnn_hidden_dim, self.n_actions)
        self.fc_critic = nn.Linear(self.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size=1):
        return self.rnn.weight_ih.new(batch_size, self.rnn_hidden_dim).zero_()

    def forward(self, 
                obs_batch: torch.Tensor, 
                hidden_state: torch.Tensor,
                dones_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        if dones_mask is None:
            dones_mask = torch.zeros(B, L, N, 1, device=obs_batch.device, dtype=torch.float32)
        dones_mask_expanded = dones_mask.permute(1, 0, 2, 3).reshape(L, B * N, 1)
            
        cnn_features = self.conv(obs_flat)
        rnn_input = cnn_features.reshape(B, L, N, -1).permute(1, 0, 2, 3).reshape(L, B * N, -1)
        
        gru_hiddens = []
        h_current = h_in
        for t in range(L):
            h_current = h_current * (1.0 - dones_mask_expanded[t])
            h_current = self.rnn(rnn_input[t], h_current)
            gru_hiddens.append(h_current)

        rnn_output_flat = torch.stack(gru_hiddens, dim=0).permute(1, 0, 2).reshape(B * L * N, -1)
        next_hidden_state = h_current.reshape(B, N, self.rnn_hidden_dim)

        actor_logits_flat = self.fc_actor(rnn_output_flat)
        state_values_flat = self.fc_critic(rnn_output_flat)

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
        print(f"  - state_shape (Dict): {env_info['state_shape']}")
    
    except ImportError:
        print("\n!!! Error: Unable to import build_meltingpot_env from MeltingPotWrapper.py.")
        env_info = {
            "n_agents": 2, "n_actions": 8,
            "obs_shape": (3, 40, 40), 
            "state_shape": {"rgb": (2, 3, 40, 40), "vector": (2, 0)}
        }
        print("\nWarning: Using default dimensions to continue verification...")
    except Exception as e:
        print(f"\n!!! Error: Failed to load environment: {e}")
        env_info = {
            "n_agents": 2, "n_actions": 8,
            "obs_shape": (3, 40, 40), 
            "state_shape": {"rgb": (2, 3, 40, 40), "vector": (2, 0)}
        }
        print("\nWarning: Using default dimensions to continue verification...")

    N_AGENTS = env_info['n_agents']
    N_ACTIONS = env_info['n_actions']
    OBS_SHAPE = env_info['obs_shape']
    OBS_VECTOR_DIM = env_info.get('obs_vector_dim', 0)
    STATE_SHAPE_DICT = env_info['state_shape']
    RNN_HIDDEN_DIM = 64
    MIXING_EMBED_DIM = 32

    agent_net = AgentNetwork(OBS_SHAPE, OBS_VECTOR_DIM, N_ACTIONS, RNN_HIDDEN_DIM)
    mixer_net = MixerNetwork(N_AGENTS, STATE_SHAPE_DICT, MIXING_EMBED_DIM)

    print("\nNetwork instantiation successful:")

    B = 4
    L = 8

    dummy_obs_rgb_seq = torch.rand(B, L, N_AGENTS, *OBS_SHAPE)
    dummy_obs_vec_seq = torch.rand(B, L, N_AGENTS, OBS_VECTOR_DIM)
    
    state_vec_shape_l = STATE_SHAPE_DICT['vector']
    if state_vec_shape_l is None:
        state_vec_shape_l = (N_AGENTS, 0)
        dummy_obs_vec_seq = torch.rand(B, L, N_AGENTS, 0) 
        
    dummy_state_rgb_seq = torch.rand(B, L, *STATE_SHAPE_DICT['rgb'])
    dummy_state_vec_seq = torch.rand(B, L, *state_vec_shape_l)
    dummy_hidden_init = agent_net.init_hidden(B * N_AGENTS, 'cpu').reshape(B, N_AGENTS, RNN_HIDDEN_DIM)

    dummy_obs_rgb_step = torch.rand(B, N_AGENTS, *OBS_SHAPE)
    dummy_obs_vec_step = torch.rand(B, N_AGENTS, OBS_VECTOR_DIM)
    
    state_vec_shape = STATE_SHAPE_DICT['vector']
    if state_vec_shape is None:
        state_vec_shape = (N_AGENTS, 0)
        dummy_obs_vec_step = torch.rand(B, N_AGENTS, 0)

    dummy_state_rgb_step = torch.rand(B, *STATE_SHAPE_DICT['rgb'])
    dummy_state_vec_step = torch.rand(B, *state_vec_shape) 
    dummy_hidden_step = agent_net.init_hidden(B * N_AGENTS, 'cpu').reshape(B, N_AGENTS, RNN_HIDDEN_DIM)

    print("\n--- Validate AgentNetwork ---")
    try:
        q_vals_seq, next_hidden_seq = agent_net(dummy_obs_rgb_seq, dummy_obs_vec_seq, dummy_hidden_init)
        print("Sequence Input:")
        print(f"  - Output q_vals: \t{q_vals_seq.shape} (Expected: {(B, L, N_AGENTS, N_ACTIONS)})")
        print(f"  - Output next_hidden: {next_hidden_seq.shape} (Expected: {(B, N_AGENTS, RNN_HIDDEN_DIM)})")
        assert q_vals_seq.shape == (B, L, N_AGENTS, N_ACTIONS)
        assert next_hidden_seq.shape == (B, N_AGENTS, RNN_HIDDEN_DIM)
    except Exception as e:
        print(f"!!! AgentNetwork Sequence Input failed: {e}")
        import traceback
        traceback.print_exc()


    try:
        q_vals_step, next_hidden_step = agent_net(dummy_obs_rgb_step, dummy_obs_vec_step, dummy_hidden_step)
        print("\nStep Input:")
        print(f"  - Output q_vals: \t{q_vals_step.shape} (Expected: {(B, N_AGENTS, N_ACTIONS)})")
        print(f"  - Output next_hidden: {next_hidden_step.shape} (Expected: {(B, N_AGENTS, RNN_HIDDEN_DIM)})")
        assert q_vals_step.shape == (B, N_AGENTS, N_ACTIONS)
        assert next_hidden_step.shape == (B, N_AGENTS, RNN_HIDDEN_DIM)
    except Exception as e:
        print(f"!!! AgentNetwork Step Input failed: {e}")
        import traceback
        traceback.print_exc()

    
    print("\n--- Validate MixerNetwork ---")
    
    if 'q_vals_seq' in locals():
        dummy_actions_seq = torch.randint(0, N_ACTIONS, (B, L, N_AGENTS))
        chosen_q_vals_seq = torch.gather(q_vals_seq, dim=3, index=dummy_actions_seq.unsqueeze(-1)).squeeze(-1) 
        
        try:
            q_tot_seq, _ = mixer_net(chosen_q_vals_seq, dummy_state_rgb_seq, dummy_state_vec_seq)
            print("Sequence Input:")
            print(f"  - Output q_tot: \t{q_tot_seq.shape} (Expected: {(B, L, 1)})")
            assert q_tot_seq.shape == (B, L, 1)
        except Exception as e:
            print(f"!!! MixerNetwork Sequence Input failed: {e}")
            import traceback
            traceback.print_exc()

    if 'q_vals_step' in locals():
        dummy_actions_step = torch.randint(0, N_ACTIONS, (B, N_AGENTS))
        chosen_q_vals_step = torch.gather(q_vals_step, dim=2, index=dummy_actions_step.unsqueeze(-1)).squeeze(-1)
        
        try:
            q_tot_step, _ = mixer_net(chosen_q_vals_step, dummy_state_rgb_step, dummy_state_vec_step)
            print("\nStep Input:")
            print(f"  - Output q_tot: \t{q_tot_step.shape} (Expected: {(B, 1)})")
            assert q_tot_step.shape == (B, 1)
        except Exception as e:
            print(f"!!! MixerNetwork Step Input failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*40)
    print("✅ QMIX Network Structure Validation Completed!")
    print("="*40)