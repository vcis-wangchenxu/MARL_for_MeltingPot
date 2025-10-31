from collections import deque
from typing import Dict, List, Any
import numpy as np

class OffPolicyReplayBuffer:
    """
    An episodic experience replay buffer optimized for off-policy multi-agent reinforcement learning.
    """
    def __init__(self, env_info: dict, capacity: int = 5000):
        self.env_info = env_info
        self.n_agents = env_info["n_agents"]                # num of agents
        self.agent_ids = env_info["agent_ids"]              # list of agent IDs
        self.buffer = deque(maxlen=int(capacity))           # episode-level buffer
        self._current_episode = self._init_episode_buffer() # current episode buffer
        self._current_episode_len = 0                       # length of current episode

    def _init_episode_buffer(self) -> Dict[str, List]:
        """ Initialize an empty episode buffer. """
        return {                                            # episode-level storage
            "state": [], "obs": [], "actions": [], "rewards": [],
            "dones": [], "next_state": [], "next_obs": [],
            "collective_rewards": []  # store collective rewards
        }
    
    def push(self, 
             obs_dict: Dict[str, np.ndarray],       # observations for all agents (C, H, W)
             state: np.ndarray,                     # global state (state_dim,)
             actions_dict: Dict[str, int],          # actions for all agents
             rewards_dict: Dict[str, float],        # rewards for all agents
             dones_dict: Dict[str, bool],           # done flags for all agents
             next_obs_dict: Dict[str, np.ndarray],  # next observations for all agents (C, H, W)
             next_state: np.ndarray,                # next global state (state_dim,)
             info_dict: Dict[str, Any] = {}):      # additional info
        """ Add a time step of data to the current episode buffer. """
        obs_array = np.stack([obs_dict[aid]['RGB'] for aid in self.agent_ids], axis=0)      # (n_agents, C, H, W)
        actions_array = np.array([actions_dict[aid] for aid in self.agent_ids])      # (n_agents,)
        next_obs_array = np.stack([next_obs_dict[aid]['RGB'] for aid in self.agent_ids], axis=0) # (n_agents, C, H, W)
        rewards_array = np.array([rewards_dict[aid] for aid in self.agent_ids])      # (n_agents,)

        done_global = any(dones_dict.values())       # global done as any individual done
        
        collective_reward = info_dict.get('collective_reward', 0.0)     # get collective reward
        
        self._current_episode["obs"].append(obs_array)          
        self._current_episode["state"].append(state)
        self._current_episode["actions"].append(actions_array)
        self._current_episode["rewards"].append(rewards_array)     # store individual rewards
        self._current_episode["dones"].append(done_global)
        self._current_episode["next_obs"].append(next_obs_array)
        self._current_episode["next_state"].append(next_state)
        self._current_episode["collective_rewards"].append(collective_reward) # store collective rewards

        self._current_episode_len += 1
        if done_global:
            self.commit_episode()

    def commit_episode(self):
        """ Commit the current episode buffer to the main buffer. """
        if self._current_episode_len == 0:
            return
        episode_batch = {}                            # convert lists to arrays
        for key, data_list in self._current_episode.items():
            episode_batch[key] = np.stack(data_list, axis=0) # (T, ...)
        self.buffer.append(episode_batch)             # add episode to main buffer
        self._current_episode = self._init_episode_buffer() # reset current episode buffer
        self._current_episode_len = 0

    def sample(self, batch_size: int, sequence_length: int) -> Dict[str, np.ndarray]:
        """ Sample a batch of sequences from the buffer. """
        if len(self.buffer) == 0:
            raise ValueError("Not enough episodes in the buffer to sample from (buffer is empty).")
            
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        batch_list = []
        sampled_ep_indices = np.random.choice(            # sample episodes
            len(self.buffer), batch_size, replace=True
        )
        
        for ep_idx in sampled_ep_indices:           # sample sequences from episodes
            ep_data = self.buffer[ep_idx]           # episode data
            ep_len = ep_data["dones"].shape[0]      # episode length

            if ep_len < sequence_length:
                raise ValueError(
                    f"Sampled a too-short episode (length {ep_len}), "
                    f"unable to sample sequence (length {sequence_length}). "
                    f"Please increase the number of 'warm-up' steps before training."
                )
            
            max_start_idx = ep_len - sequence_length          # maximum valid start index
            start_idx = np.random.randint(0, max_start_idx + 1)   # slice the episode data
            
            sliced_ep = {}                        # sliced episode data
            for key, data in ep_data.items():     
                sliced_ep[key] = data[start_idx : start_idx + sequence_length]   # (L, ...)
            batch_list.append(sliced_ep)

        final_batch = {}                          # final batch dictionary
        keys = batch_list[0].keys()               # keys in the episode data
        for key in keys:
            final_batch[key] = np.stack([ep_slice[key] for ep_slice in batch_list], axis=0)     # (B, L, ...)

        return final_batch

    def __len__(self) -> int:
        return sum(ep["dones"].shape[0] for ep in self.buffer)
    
class OnPolicyRolloutBuffer:
    """
    An optimized temporary data buffer (Rollout Storage) for On-Policy MARL.
    """
    def __init__(self, env_info: dict):
        self.env_info = env_info                           # environment information
        self.n_agents = env_info["n_agents"]               # number of agents
        self.agent_ids = env_info["agent_ids"]             # list of agent IDs
        self.storage = self._init_storage()                # initialize storage

    def _init_storage(self) -> Dict[str, Any]:
        """Initialize empty lists to store trajectories."""
        storage = {
            "global_state": [],        # store global states
            "dones": [],               # store done flags
            "next_global_state": [],   # GAE needs s_{t+1}
            "collective_rewards": [],  # store collective rewards  
        }
        for agent_id in self.agent_ids:
            storage[agent_id] = {
                "obs": [],           # observations
                "actions": [],       # actions
                "log_probs": [],     # PPO needs
                "values": [],        # PPO needs
                "next_obs": [],      # GAE needs o_{t+1}
                "rewards": [],       # individual rewards
            }
        return storage

    def push(self, 
             obs_dict: Dict[str, np.ndarray], 
             state: np.ndarray, 
             actions_dict: Dict[str, int], 
             rewards_dict: Dict[str, float], 
             dones_dict: Dict[str, bool],
             log_probs_dict: Dict[str, float],       # Provided by PPO policy
             values_dict: Dict[str, float],          # Provided by PPO policy
             next_obs_dict: Dict[str, np.ndarray],   # From env.step
             next_state: np.ndarray,                  # From env.step
             info_dict: Dict[str, Any] = {}):        # Info dictionary
        """
        Push a single timestep of data into the temporary buffer.
        """
        self.storage["global_state"].append(state)
        self.storage["next_global_state"].append(next_state)
        self.storage["dones"].append(any(dones_dict.values()))
        
        collective_reward = info_dict.get('collective_reward', 0.0)
        self.storage["collective_rewards"].append(collective_reward)  # store collective rewards

        for agent_id in self.agent_ids:
            self.storage[agent_id]["obs"].append(obs_dict[agent_id]['RGB'])
            self.storage[agent_id]["next_obs"].append(next_obs_dict[agent_id]['RGB'])
            self.storage[agent_id]["actions"].append(actions_dict[agent_id])
            self.storage[agent_id]["log_probs"].append(log_probs_dict[agent_id])
            self.storage[agent_id]["values"].append(values_dict[agent_id])
            self.storage[agent_id]["rewards"].append(rewards_dict[agent_id]) 
            

    def get_all_data(self) -> Dict[str, Any]:
        """ Retrieve all stored data as numpy arrays. """
        data = {
            "global_state": np.array(self.storage["global_state"]),
            "next_global_state": np.array(self.storage["next_global_state"]),
            "dones": np.array(self.storage["dones"]),
            "collective_rewards": np.array(self.storage["collective_rewards"]), 
        }
        
        for agent_id in self.agent_ids:
            agent_data = self.storage[agent_id]
            data[agent_id] = {
                "obs": np.array(agent_data["obs"]),
                "next_obs": np.array(agent_data["next_obs"]),
                "actions": np.array(agent_data["actions"]),
                "log_probs": np.array(agent_data["log_probs"]),
                "values": np.array(agent_data["values"]),
                "rewards": np.array(agent_data["rewards"]), # [新增]
            }
            
        return data

    def clear(self) -> None:
        """ Clear all stored data. """
        self.storage = self._init_storage()

    def __len__(self) -> int:
        """ Return the current number of *timesteps* stored. """
        return len(self.storage["global_state"])
    
if __name__ == "__main__":
    
    print("="*40)
    print("Starting validation of the MARL buffer module...")
    print("="*40)

    # --- Initialize environment ---
    try:
        from MeltingPotWrapper import build_meltingpot_env as GeneralMeltingPotWrapper
        print("Successfully imported build_meltingpot_env from MeltingPotWrapper.")
    except ImportError:
        print("Failed to import MeltingPotWrapper. Please ensure the filename is correct and in the Python path.")
        exit()

    env = GeneralMeltingPotWrapper(env_name="collaborative_cooking__asymmetric")
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    agent_ids = env_info["agent_ids"]

    print("Environment Information:")
    print(f"  - n_agents: {n_agents}, agent_ids: {agent_ids}")
    print(f"  - obs_shape: {env_info['obs_shape']}, state_shape: {env_info['state_shape']}")
    print(f"  - episode_limit: {env_info['episode_limit']}")

    print("\n" + "="*20)
    print("Verify: OffPolicyReplayBuffer (QMIX)")
    print("="*20)
    
    off_policy_buffer = OffPolicyReplayBuffer(env_info, capacity=10)

    # Fill the buffer (2 episodes)
    print("Filling Off-Policy buffer (2 episodes)...")
    N_EPISODES = 2
    
    for ep in range(N_EPISODES):
        obs_dict, state = env.reset()
        done = False
        while not done:
            actions_dict = {aid: np.random.randint(0, env.n_actions) for aid in agent_ids}
            next_obs_dict, next_state, rewards_dict, dones_dict, info_dict = env.step(actions_dict)
            done = any(dones_dict.values())
            off_policy_buffer.push(
                obs_dict, state, actions_dict, rewards_dict, dones_dict,
                next_obs_dict, next_state, info_dict
            )
            obs_dict, state = next_obs_dict, next_state

    print(f"  - Episodes in buffer: {len(off_policy_buffer.buffer)} (Expected: {N_EPISODES})")
    print(f"  - Total timesteps in buffer: {len(off_policy_buffer)} (Expected: {N_EPISODES * env_info['episode_limit']})")

    # Sampling
    BATCH_SIZE = 4
    SEQUENCE_LENGTH = 8
    print(f"Sampling (B={BATCH_SIZE}, L={SEQUENCE_LENGTH})...")

    try:
        batch = off_policy_buffer.sample(BATCH_SIZE, SEQUENCE_LENGTH)
        print("  - Sampling successful.")
        print(f"  - batch['state'].shape: {batch['state'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, env_info['state_shape'])})")
        assert batch['state'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, env_info['state_shape'])

        print(f"  - batch['obs'].shape: {batch['obs'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, n_agents, *env_info['obs_shape'])})")
        assert batch['obs'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, n_agents, *env_info['obs_shape'])
        print(f"  - batch['next_obs'].shape: {batch['next_obs'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, n_agents, *env_info['obs_shape'])})")
        assert batch['next_obs'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, n_agents, *env_info['obs_shape'])

        print(f"  - batch['rewards'].shape: {batch['rewards'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, n_agents)})")
        assert batch['rewards'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, n_agents)
        print(f"  - batch['collective_rewards'].shape: {batch['collective_rewards'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH)})")
        assert batch['collective_rewards'].shape == (BATCH_SIZE, SEQUENCE_LENGTH)

        print("✅ Off-Policy buffer verification successful!")

    except Exception as e:
        print(f"\n!!! Off-Policy sampling failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*20)
    print("Verify: OnPolicyRolloutBuffer (MAPPO)")
    print("="*20)
    
    on_policy_buffer = OnPolicyRolloutBuffer(env_info)

    # Fill the buffer (1 episode)
    print("Filling On-Policy buffer (1 episode)...")
    obs_dict, state = env.reset()
    done = False
    
    while not done:
        actions_dict = {aid: np.random.randint(0, env.n_actions) for aid in agent_ids}
        log_probs_dict = {aid: np.random.rand() for aid in agent_ids} # 伪造
        values_dict = {aid: np.random.rand() for aid in agent_ids} # 伪造
        
        next_obs_dict, next_state, rewards_dict, dones_dict, info_dict = env.step(actions_dict)
        done = any(dones_dict.values())

        on_policy_buffer.push(
            obs_dict, state, actions_dict, rewards_dict, dones_dict,
            log_probs_dict, values_dict,
            next_obs_dict, next_state, info_dict
        )
        obs_dict, state = next_obs_dict, next_state

    print(f"  - Timesteps in buffer: {len(on_policy_buffer)} (Expected: {env_info['episode_limit']})")
    assert len(on_policy_buffer) == env_info['episode_limit']

    # Get data (no advantage calculation)
    print(f"Getting all data...")
    data = on_policy_buffer.get_all_data()

    print("  - Getting data successful. Verifying data shapes:")
    T = env_info['episode_limit'] # trajectory length

    print(f"  - data['global_state'].shape: \t{data['global_state'].shape} (Expected: {(T, env_info['state_shape'])})")
    assert data['global_state'].shape == (T, env_info['state_shape'])

    print(f"  - data['next_global_state'].shape: {data['next_global_state'].shape} (Expected: {(T, env_info['state_shape'])})")
    assert data['next_global_state'].shape == (T, env_info['state_shape'])

    assert 'rewards' not in data

    print(f"  - data['collective_rewards'].shape: {data['collective_rewards'].shape} (Expected: {(T,)})")
    assert data['collective_rewards'].shape == (T,)

    print(f"  - data['player_0']['obs'].shape: \t{data['player_0']['obs'].shape} (Expected: {(T, *env_info['obs_shape'])})")
    assert data['player_0']['obs'].shape == (T, *env_info['obs_shape'])

    print(f"  - data['player_0']['next_obs'].shape: \t{data['player_0']['next_obs'].shape} (Expected: {(T, *env_info['obs_shape'])})")
    assert data['player_0']['next_obs'].shape == (T, *env_info['obs_shape'])

    print(f"  - data['player_0']['rewards'].shape: \t{data['player_0']['rewards'].shape} (Expected: {(T,)})")
    assert data['player_0']['rewards'].shape == (T,)
    if 'player_1' in data:
         print(f"  - data['player_1']['rewards'].shape: \t{data['player_1']['rewards'].shape} (Expected: {(T,)})")
         assert data['player_1']['rewards'].shape == (T,)

    on_policy_buffer.clear()
    print("\n  - Buffer cleared.")
    print(f"  - Timesteps in buffer: {len(on_policy_buffer)} (Expected: 0)")
    assert len(on_policy_buffer) == 0

    print("✅ On-Policy buffer validation successful!")

    env.close()