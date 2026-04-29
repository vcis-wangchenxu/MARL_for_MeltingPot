import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AgentDRQN(nn.Module):
    def __init__(self, obs_rgb_shape: Tuple[int, int, int],
                 vector_dim: int, 
                 n_actions: int,
                 n_agents: int,
                 rnn_hidden_dim: int,
                 rnn_layers: int =1,
                 use_agent_id: bool = False):
        super(AgentDRQN, self).__init__()
        self.obs_rgb_shape = obs_rgb_shape           
        self.vector_dim = vector_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.use_agent_id = use_agent_id

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(obs_rgb_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_img = torch.zeros(1, *obs_rgb_shape)
            cnn_out = self.conv_backbone(dummy_img)
            self.cnn_flat_dim = cnn_out.reshape(1, -1).shape[1]
        
        if self.vector_dim > 0:    # Projcction for Vector Obs
            self.vec_encoder = nn.Sequential(
                nn.Linear(vector_dim, 64),
                nn.ReLU(),
            )
            vec_out_dim = 64
        else:
            self.vec_encoder = nn.Identity()
            vec_out_dim = 0

        self.rnn_input_dim = self.cnn_flat_dim + vec_out_dim
        if self.use_agent_id:
            self.rnn_input_dim += self.n_agents
        
        self.rnn = nn.GRU(
            input_size = self.rnn_input_dim,
            hidden_size = self.rnn_hidden_dim,
            num_layers = self.rnn_layers,
            batch_first = True,
        )

        self.head = nn.Linear(self.rnn_hidden_dim, self.n_actions)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        return torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
    
    def forward(self, obs_dict, hidden_state, agent_id=None):
        """
        Forward pass for the DRQN agent.

        Args:
            obs_dict (Dict[str, torch.Tensor]): Dictionary containing observations.
                - 'rgb': Image observations with shape (Batch, Seq_Len, C, H, W).
                - 'vector': (Optional) Vector observations with shape (Batch, Seq_Len, Vector_Dim).
            hidden_state (torch.Tensor): The hidden state for the RNN.
                Shape: (num_layers, Batch, rnn_hidden_dim).
            agent_id (torch.Tensor, optional): Agent IDs if use_agent_id is True.
                Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1).

        Returns:
            q_values (torch.Tensor): Q-values for each action.
                Shape: (Batch, Seq_Len, n_actions).
            new_hidden (torch.Tensor): Updated hidden state.
                Shape: (num_layers, Batch, rnn_hidden_dim).
        """
        
        rgb = obs_dict['rgb'].float()
        B, L, C, H, W = rgb.shape
        rgb_flat = rgb.reshape(B * L, C, H, W)   

        features_img = self.conv_backbone(rgb_flat)
        features_img = features_img.reshape(B * L, -1)    # (B*L, D_img)

        features_vec = None
        if self.vector_dim > 0:
            vec = obs_dict['vector'].float()
            vec_flat = vec.reshape(B * L, -1)              # (B*L, D_vec)
            features_vec = self.vec_encoder(vec_flat)      # (B*L, D_vec_out)
        
        components = [features_img]
        if features_vec is not None:
            components.append(features_vec)
        
        if self.use_agent_id:
            if agent_id is None:
                raise ValueError("agent_id must be provided when use_agent_id is True")
            agent_id_flat = agent_id.reshape(-1).long()        # (B*L,)
            agent_id_one_hot = F.one_hot(agent_id_flat, num_classes=self.n_agents).float()    # (B*L, n_agents)
            components.append(agent_id_one_hot)

        rnn_in_flat = torch.cat(components, dim=-1)   # (B*L, D_total)

        rnn_in = rnn_in_flat.reshape(B, L, -1)        # (B, L, D_total)
        self.rnn.flatten_parameters()                 # Ensure efficient processing
        rnn_out, new_hidden = self.rnn(rnn_in, hidden_state)  # rnn_out: (B, L, H), new_hidden: (num_layers, B, H)

        q_values = self.head(rnn_out)                     # (B, L, n_actions)
        return q_values, new_hidden
