import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer

class CNNEncoder(nn.Module):
    """
    Standard CNN Encoder.
    """
    def __init__(self, input_channels, hidden_dim):
        super(CNNEncoder, self).__init__()
        self.main = nn.Sequential(
            orthogonal_init(nn.Conv2d(input_channels, 32, 8, stride=4)), nn.ReLU(),
            orthogonal_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            orthogonal_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            orthogonal_init(nn.Linear(64 * 7 * 7, hidden_dim)), nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)

class MAPPOActor(nn.Module):
    def __init__(self, obs_shape, vector_dim, n_actions, n_agents, hidden_dim, rnn_layers=1, use_agent_id=True):
        super(MAPPOActor, self).__init__()
        self.use_vector = (vector_dim > 0)
        self.use_agent_id = use_agent_id
        self.n_agents = n_agents

        self.cnn = CNNEncoder(obs_shape[0], hidden_dim)
        
        input_dim = hidden_dim
        
        if self.use_vector:
            self.vec_mlp = nn.Sequential(
                orthogonal_init(nn.Linear(vector_dim, hidden_dim)), nn.ReLU()
            )
            input_dim += hidden_dim
            
        if self.use_agent_id:
            # Agent ID 是 One-Hot 向量，长度为 n_agents
            input_dim += n_agents

        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.action_head = orthogonal_init(nn.Linear(hidden_dim, n_actions), gain=0.01)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_dim).to(device)

    def forward(self, obs_dict, hidden_state, agent_id=None):
        # obs_dict['rgb']: (Batch, Seq, C, H, W)
        rgb = obs_dict['rgb']
        B, L, C, H, W = rgb.shape
        
        # CNN Feature
        rgb_flat = rgb.view(B * L, C, H, W)
        rgb_feat = self.cnn(rgb_flat) 
        features = [rgb_feat]
        
        # Vector Feature
        if self.use_vector:
            vec = obs_dict['vector'] 
            vec_flat = vec.view(B * L, -1)
            vec_feat = self.vec_mlp(vec_flat)
            features.append(vec_feat)

        # Agent ID Feature
        if self.use_agent_id:
            if agent_id is None:
                raise ValueError("Actor requires agent_id but it is None")
            # agent_id: (Batch, Seq, N_Agents) -> one-hot usually done outside or embedding

            if agent_id.dim() == 3:
                agent_id_flat = agent_id.view(B * L, -1)

            elif agent_id.dim() == 3 and agent_id.shape[1] == L: # Case (Batch, Seq, N_Agents)
                 agent_id_flat = agent_id.view(B * L, -1)
            else:
                agent_id_flat = agent_id

            features.append(agent_id_flat)

        # Concat all features
        combined_feat = torch.cat(features, dim=1)

        # RNN Forward
        rnn_in = combined_feat.view(B, L, -1)
        rnn_out, next_hidden = self.rnn(rnn_in, hidden_state)
        logits = self.action_head(rnn_out)
        
        return logits, next_hidden

class MAPPOCritic(nn.Module):
    """
    Centralized Critic.
    """
    def __init__(self, state_shape, hidden_dim, n_agents=0, use_agent_id=False):
        super(MAPPOCritic, self).__init__()
        self.cnn = CNNEncoder(state_shape[0], hidden_dim)
        
        input_dim = hidden_dim
        self.use_agent_id = use_agent_id
        if use_agent_id:
            input_dim += n_agents

        self.mlp = nn.Sequential(
             orthogonal_init(nn.Linear(input_dim, hidden_dim)), nn.ReLU()
        )
        self.value_head = orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, state, agent_id=None):
        # state: (Batch, Seq, C, H, W)
        B, L, C, H, W = state.shape
        flat_state = state.view(B * L, C, H, W)
        feat = self.cnn(flat_state)

        if self.use_agent_id:
             if agent_id is None: raise ValueError("Critic requires agent_id")
             
             if agent_id.dim() == 3:
                 agent_id_flat = agent_id.view(B * L, -1)
             else:
                 agent_id_flat = agent_id
                 
             feat = torch.cat([feat, agent_id_flat], dim=1)
        
        feat = self.mlp(feat)
        value = self.value_head(feat)
        return value.view(B, L, 1)