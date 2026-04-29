import numpy as np
import torch
from utils.util import set_seed

class Evaluator:
    """
    Evaluator class for Melting Pot (compatible with MeltingPotAsyncVectorEnv).
    Handles dictionary observations and batched actions.
    """
    def __init__(self, env, agent, device, policy_type='off-policy', seed=None):
        """
        Initialize the Evaluator.

        Args:
            env: The environment instance (MeltingPotAsyncVectorEnv with num_envs=1).
            agent: The agent instance.
            device: 'cpu' or 'cuda'.
            policy_type: 'on-policy' or 'off-policy'.
            seed: Evaluation seed.
        """
        self.env = env
        self.agent = agent
        self.device = device
        self.policy_type = policy_type 
        self.seed = seed

        self.env_info = self.env.get_env_info()
        self.n_agents = self.env_info['n_agents']
        self.obs_vector_dim = self.env_info['obs_vector_dim']
        
        print(f"[Evaluator] Initialized | Policy: {self.policy_type} | Seed: {self.seed}")

    def _is_on_policy(self) -> bool:
        policy = (self.policy_type or "").lower().strip()
        if policy.startswith("on"):
            return True
        if "on-policy" in policy:
            return True
        return False

    def _set_agent_mode(self, training: bool) -> None:
        """Best-effort mode switching for agents that may not implement .train()/.eval()."""
        if hasattr(self.agent, "train") and callable(getattr(self.agent, "train")) and \
           hasattr(self.agent, "eval") and callable(getattr(self.agent, "eval")):
            if training:
                self.agent.train()
            else:
                self.agent.eval()
            return

        # Fall back to toggling common module attributes
        def _set(obj):
            if obj is None:
                return
            if hasattr(obj, "train") and callable(getattr(obj, "train")):
                obj.train(training)

        for attr in ("actor", "critic", "actors", "critics", "agent", "agents", "target_agent", "target_agents"):
            _set(getattr(self.agent, attr, None))

    def evaluate(self, n_episodes=5, seed=None):
        """
        Run evaluation episodes.

        Returns:
            float: Mean episode reward.
        """
        self._set_agent_mode(training=False)
        rewards = []

        set_seed(seed if seed is not None else self.seed)
        
        for _ in range(n_episodes):
            obs, infos = self.env.reset()    # obs: {'rgb': (B, N, ...), 'vector': (B, N, ...)}

            # Build critic inputs (MAPPO requires either flattened state or global_state)
            # obs['rgb']: (B, N, C, H, W)
            B, N, C, H, W = obs['rgb'].shape
            state_flatten = obs['rgb'].reshape(B, -1, H, W)

            state_global = None
            if infos and isinstance(infos, list) and 'global_state' in infos[0]:
                s_global = np.stack([info['global_state']['world_rgb'] for info in infos], axis=0)
                state_global = np.transpose(s_global.astype(np.float32) / 255.0, (0, 3, 1, 2))

            # Init Hidden: (Batch=1, N, L, H)
            hidden = self.agent.init_hidden(1)
            
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    # Melting Pot Dict Obs: {'rgb': (B, N, ...), 'vector': ...}
                    # Data is already batched (B=1) from VectorEnv
                    rgb = torch.from_numpy(obs['rgb']).to(self.device)
                    
                    vector = None
                    if self.obs_vector_dim > 0 :
                        vector = torch.from_numpy(obs['vector']).to(self.device)
                    
                    obs_input = {'rgb': rgb, 'vector': vector}

                    if self._is_on_policy():
                        state_tensor = torch.from_numpy(state_flatten).to(self.device)
                        global_state_tensor = None
                        if state_global is not None:
                            global_state_tensor = torch.from_numpy(state_global).to(self.device)

                        # MAPPO.take_action returns: values, actions, log_probs, next_hidden
                        _, actions, _, next_hidden = self.agent.take_action(
                            obs=obs_input,
                            state=state_tensor,
                            global_state=global_state_tensor,
                            hidden_state=hidden,
                        )
                    else:
                        # Off-policy (VDN/QMIX) -> returns (actions, hidden)
                        actions, next_hidden = self.agent.take_action(
                            obs_input, hidden, current_step=0, evaluation=True
                        )
                
                # 4. Process Actions for Env
                # actions tensor: (B, N) -> numpy: (B, N)
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                
                # 5. Step
                # AsyncVectorEnv expects actions as (B, N) or List of length B
                step_ret = self.env.step(actions)
                
                # Handle return values (4 or 5)
                # MP Wrapper: obs, rewards, dones, infos
                if len(step_ret) == 4:
                    next_obs, reward, dones, infos = step_ret
                else:
                    next_obs, reward, dones, truncated, infos = step_ret
                
                # Sum reward (Cooperative: sum over all agents)
                # reward shape: (B, N) -> sum -> scalar (since B=1)
                episode_reward += np.sum(reward)
                
                obs = next_obs
                hidden = next_hidden

                # Update critic inputs for on-policy evaluation
                if self._is_on_policy():
                    B, N, C, H, W = obs['rgb'].shape
                    state_flatten = obs['rgb'].reshape(B, -1, H, W)

                    state_global = None
                    if infos and isinstance(infos, list) and 'global_state' in infos[0]:
                        s_global = np.stack([info['global_state']['world_rgb'] for info in infos], axis=0)
                        state_global = np.transpose(s_global.astype(np.float32) / 255.0, (0, 3, 1, 2))
                
                # Check termination
                # dones shape: (B, N). Check if any agent is done.
                if np.any(dones):
                    done = True
            
            rewards.append(episode_reward)

        self._set_agent_mode(training=True)
        return np.mean(rewards)