from collections import deque
from typing import Dict, List, Any
import numpy as np
from yaml import warnings

class OffPolicyReplayBuffer:
    """
    An episodic experience replay buffer optimized for off-policy multi-agent reinforcement learning.
    Stores 'rgb' and 'vector' components separately based on models.py and MeltingPotWrapper.py.
    """
    def __init__(self, env_info: dict, capacity: int = 5000):
        self.env_info = env_info                               # environment information
        self.n_agents = env_info["n_agents"]                   # number of agents
        self.agent_ids = env_info["agent_ids"]                 # agent IDs
        self.buffer = deque(maxlen=int(capacity))              # main buffer to store episodes
        self._current_episode = self._init_episode_buffer()    # current episode buffer
        self._current_episode_len = 0                          # current episode length
    
    def _init_episode_buffer(self) -> Dict[str, List]:
        """ Initialize an empty episode buffer with split keys. """
        return {
            "obs_rgb": [],              # list of dicts: each dict maps agent_id to RGB obs
            "obs_vector": [],           # list of dicts: each dict maps agent_id to vector obs
            "state_rgb": [],            # list of global RGB states
            "state_vector": [],         # list of global vector states
            "actions": [],              # list of action arrays (n_agents,)
            "rewards": [],              # list of individual reward arrays (n_agents,)
            "dones": [],                # list of done flags (bool)
            "next_obs_rgb": [],         # list of dicts: each dict maps agent_id to next RGB obs
            "next_obs_vector": [],      # list of dicts: each dict maps agent_id to next vector obs
            "next_state_rgb": [],       # list of next global RGB states
            "next_state_vector": [],    # list of next global vector states
            "collective_rewards": [],   # list of collective rewards (float)
            "next_world_rgb": [],       # list of next world RGB observations
        }

    def push(self, 
             obs_dict: Dict[str, np.ndarray],       
             state: Dict[str, np.ndarray],          
             actions_dict: Dict[str, int],
             rewards_dict: Dict[str, float],
             dones_dict: Dict[str, bool],
             next_obs_dict: Dict[str, np.ndarray],  
             next_state: Dict[str, np.ndarray],     
             info_dict: Dict[str, Any] = {}):
        
        obs_rgb_array = state['rgb']
        obs_vector_array = state['vector']
        state_rgb_array = state['rgb']
        state_vector_array = state['vector']
        
        next_obs_rgb_array = next_state['rgb']
        next_obs_vector_array = next_state['vector']
        next_state_rgb_array = next_state['rgb']
        next_state_vector_array = next_state['vector']

        actions_array = np.array([actions_dict[aid] for aid in self.agent_ids])
        rewards_array = np.array([rewards_dict[aid] for aid in self.agent_ids])
        done_global = any(dones_dict.values())

        collective_reward = info_dict.get('collective_reward', 0.0)
        
        next_global_state_dict = info_dict.get('global_state', {})
        next_world_rgb = next_global_state_dict.get('world_rgb')

        self._current_episode["obs_rgb"].append(obs_rgb_array)
        self._current_episode["obs_vector"].append(obs_vector_array)
        self._current_episode["state_rgb"].append(state_rgb_array)
        self._current_episode["state_vector"].append(state_vector_array)
        
        self._current_episode["actions"].append(actions_array)
        self._current_episode["rewards"].append(rewards_array)
        self._current_episode["dones"].append(done_global)
        
        self._current_episode["next_obs_rgb"].append(next_obs_rgb_array)
        self._current_episode["next_obs_vector"].append(next_obs_vector_array)
        self._current_episode["next_state_rgb"].append(next_state_rgb_array)
        self._current_episode["next_state_vector"].append(next_state_vector_array)
        
        self._current_episode["collective_rewards"].append(collective_reward)
        self._current_episode["next_world_rgb"].append(next_world_rgb) 

        self._current_episode_len += 1
        if done_global:
            self.commit_episode()

    def commit_episode(self):
        """ Commit the current episode to the main buffer. """
        if self._current_episode_len == 0:
            return
        episode_batch = {}
        for key, data_list in self._current_episode.items():
            if data_list:
                episode_batch[key] = np.stack(data_list, axis=0)
            else:
                warnings.warn(f"Committing empty list for key '{key}' in OffPolicyReplayBuffer.")
                
        self.buffer.append(episode_batch)
        self._current_episode = self._init_episode_buffer()
        self._current_episode_len = 0

    def sample(self, batch_size: int, sequence_length: int) -> Dict[str, np.ndarray]:
        """ Sample a batch of sequences from the buffer. """
        if len(self.buffer) == 0:
            raise ValueError("Not enough episodes in the buffer to sample from (buffer is empty).")
            
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        batch_list = []
        sampled_ep_indices = np.random.choice(
            len(self.buffer), batch_size, replace=True
        )
        
        for ep_idx in sampled_ep_indices:
            ep_data = self.buffer[ep_idx]
            ep_len = ep_data["dones"].shape[0]

            if ep_len < sequence_length:
                raise ValueError(
                    f"Sampled a too-short episode (length {ep_len}), "
                    f"unable to sample sequence (length {sequence_length}). "
                    f"Please increase the number of 'warm-up' steps before training."
                )
            
            max_start_idx = ep_len - sequence_length
            start_idx = np.random.randint(0, max_start_idx + 1)
            
            sliced_ep = {}
            for key, data in ep_data.items():
                sliced_ep[key] = data[start_idx : start_idx + sequence_length]
            batch_list.append(sliced_ep)

        final_batch = {}
        keys = batch_list[0].keys()
        for key in keys:
            final_batch[key] = np.stack([ep_slice[key] for ep_slice in batch_list], axis=0)

        return final_batch

    def __len__(self) -> int:
        return sum(ep["dones"].shape[0] for ep in self.buffer)
    
class OnPolicyRolloutBuffer:
    """
    An optimized temporary data buffer (Rollout Storage) for On-Policy MARL.
    Also stores 'next_world_rgb' from info_dict.
    """
    def __init__(self, env_info: dict):
        self.env_info = env_info
        self.n_agents = env_info["n_agents"]
        self.agent_ids = env_info["agent_ids"]
        self.storage = self._init_storage()

        self._global_shape = self.env_info.get('global_shape')

    def _init_storage(self) -> Dict[str, Any]:
        """ Initialize empty lists to store trajectories. """
        storage = {
            "critic_state_rgb": [], 
            "critic_state_vector": [],
            "dones": [],
            "next_critic_state_rgb": [], 
            "next_critic_state_vector": [],
            "collective_rewards": [],
            "next_world_rgb": [] 
        }
        for agent_id in self.agent_ids:
            storage[agent_id] = {
                "obs_rgb": [], 
                "obs_vector": [],
                "actions": [],
                "log_probs": [],
                "values": [],
                "next_obs_rgb": [], 
                "next_obs_vector": [], 
                "rewards": [],
            }
        return storage

    def push(self, 
             obs_dict: Dict[str, np.ndarray],
             state: Dict[str, np.ndarray],
             actions_dict: Dict[str, int], 
             rewards_dict: Dict[str, float], 
             dones_dict: Dict[str, bool],
             log_probs_dict: Dict[str, float],
             values_dict: Dict[str, float],
             next_obs_dict: Dict[str, np.ndarray],
             next_state: Dict[str, np.ndarray],
             info_dict: Dict[str, Any] = {}):
        """
        Push a single timestep of data into the temporary buffer.
        """
        self.storage["critic_state_rgb"].append(state['rgb'])
        self.storage["critic_state_vector"].append(state['vector'])
        self.storage["next_critic_state_rgb"].append(next_state['rgb'])
        self.storage["next_critic_state_vector"].append(next_state['vector'])
        
        self.storage["dones"].append(any(dones_dict.values()))

        collective_reward = info_dict.get('collective_reward', 0.0)
        self.storage["collective_rewards"].append(collective_reward)
        
        next_global_state_dict = info_dict.get('global_state', {})
        next_world_rgb = next_global_state_dict.get('world_rgb')
        if next_world_rgb is None:
            next_world_rgb = np.zeros(self._global_shape, dtype=np.uint8)
        self.storage["next_world_rgb"].append(next_world_rgb)

        for i, agent_id in enumerate(self.agent_ids):
            self.storage[agent_id]["obs_rgb"].append(state['rgb'][i])
            self.storage[agent_id]["obs_vector"].append(state['vector'][i])
            self.storage[agent_id]["next_obs_rgb"].append(next_state['rgb'][i])
            self.storage[agent_id]["next_obs_vector"].append(next_state['vector'][i])

            self.storage[agent_id]["actions"].append(actions_dict[agent_id])
            self.storage[agent_id]["log_probs"].append(log_probs_dict[agent_id])
            self.storage[agent_id]["values"].append(values_dict[agent_id])
            self.storage[agent_id]["rewards"].append(rewards_dict[agent_id]) 
            

    def get_all_data(self) -> Dict[str, Any]:
        """ (已修改) Retrieve all stored data as numpy arrays. """
        data = {
            "critic_state_rgb": np.array(self.storage["critic_state_rgb"]),
            "critic_state_vector": np.array(self.storage["critic_state_vector"]),
            "next_critic_state_rgb": np.array(self.storage["next_critic_state_rgb"]),
            "next_critic_state_vector": np.array(self.storage["next_critic_state_vector"]),
            "dones": np.array(self.storage["dones"]),
            "collective_rewards": np.array(self.storage["collective_rewards"]), 
            "next_world_rgb": np.array(self.storage["next_world_rgb"]), 
        }
        
        for agent_id in self.agent_ids:
            agent_data = self.storage[agent_id]
            data[agent_id] = {
                "obs_rgb": np.array(agent_data["obs_rgb"]),
                "obs_vector": np.array(agent_data["obs_vector"]),
                "next_obs_rgb": np.array(agent_data["next_obs_rgb"]),
                "next_obs_vector": np.array(agent_data["next_obs_vector"]),
                "actions": np.array(agent_data["actions"]),
                "log_probs": np.array(agent_data["log_probs"]),
                "values": np.array(agent_data["values"]),
                "rewards": np.array(agent_data["rewards"]),
            }
            
        return data

    def clear(self) -> None:
        """ Clear all stored data. """
        self.storage = self._init_storage()

    def __len__(self) -> int:
        """ Return the current number of *timesteps* stored. """
        return len(self.storage["dones"])
    
if __name__ == "__main__":
    
    print("="*40)
    print("Starting validation of the MARL buffer module (Vector + Global RGB)...")
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
    
    state_shape_dict = env_info['state_shape']
    obs_vector_dim = env_info['obs_vector_dim']
    obs_shape_rgb = env_info['obs_shape']
    global_shape_rgb = env_info['global_shape'] # (H, W, C)

    print("Environment Information:")
    print(f"  - n_agents: {n_agents}, agent_ids: {agent_ids}")
    print(f"  - obs_shape (RGB): {obs_shape_rgb}")
    print(f"  - obs_vector_dim: {obs_vector_dim}")
    print(f"  - state_shape (Dict): {state_shape_dict}")
    print(f"  - global_shape (WORLD.RGB): {global_shape_rgb}") 
    print(f"  - episode_limit: {env_info['episode_limit']}")
    
    assert global_shape_rgb is not None and np.prod(global_shape_rgb) > 0

    print("\n" + "="*20)
    print("Verify: OffPolicyReplayBuffer (QMIX)")
    print("="*20)
    
    off_policy_buffer = OffPolicyReplayBuffer(env_info, capacity=10)

    print("Filling Off-Policy buffer (2 episodes)...")
    N_EPISODES = 2
    
    for ep in range(N_EPISODES):
        obs_dict, state_dict = env.reset()
        state = state_dict["concat_state"]
        
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

    # Sampling
    BATCH_SIZE = 4
    SEQUENCE_LENGTH = 8
    print(f"Sampling (B={BATCH_SIZE}, L={SEQUENCE_LENGTH})...")

    try:
        batch = off_policy_buffer.sample(BATCH_SIZE, SEQUENCE_LENGTH)
        print("  - Sampling successful.")

        print(f"  - batch['state_rgb'].shape: {batch['state_rgb'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, *state_shape_dict['rgb'])})")
        assert batch['state_rgb'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, *state_shape_dict['rgb'])
        print(f"  - batch['state_vector'].shape: {batch['state_vector'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, *state_shape_dict['vector'])})")
        assert batch['state_vector'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, *state_shape_dict['vector'])

        print(f"  - batch['next_world_rgb'].shape: {batch['next_world_rgb'].shape} (Expected: {(BATCH_SIZE, SEQUENCE_LENGTH, *global_shape_rgb)})")
        assert batch['next_world_rgb'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, *global_shape_rgb)

        print("✅ Off-Policy buffer verification successful!")

    except Exception as e:
        print(f"\n!!! Off-Policy sampling failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*20)
    print("Verify: OnPolicyRolloutBuffer (MAPPO)")
    print("="*20)
    
    on_policy_buffer = OnPolicyRolloutBuffer(env_info)

    print("Filling On-Policy buffer (1 episode)...")
    obs_dict, state_dict = env.reset()
    state = state_dict["concat_state"]
    done = False
    
    while not done:
        actions_dict = {aid: np.random.randint(0, env.n_actions) for aid in agent_ids}
        log_probs_dict = {aid: np.random.rand() for aid in agent_ids}
        values_dict = {aid: np.random.rand() for aid in agent_ids}
        
        next_obs_dict, next_state, rewards_dict, dones_dict, info_dict = env.step(actions_dict)
        
        done = any(dones_dict.values())

        on_policy_buffer.push(
            obs_dict, state, actions_dict, rewards_dict, dones_dict,
            log_probs_dict, values_dict,
            next_obs_dict, next_state, info_dict
        )
        obs_dict, state = next_obs_dict, next_state

    print(f"Getting all data...")
    data = on_policy_buffer.get_all_data()

    print("  - Getting data successful. Verifying data shapes:")
    T = env_info['episode_limit']

    print(f"  - data['critic_state_rgb'].shape: \t{data['critic_state_rgb'].shape} (Expected: {(T, *state_shape_dict['rgb'])})")
    assert data['critic_state_rgb'].shape == (T, *state_shape_dict['rgb'])
    print(f"  - data['critic_state_vector'].shape: {data['critic_state_vector'].shape} (Expected: {(T, *state_shape_dict['vector'])})")
    assert data['critic_state_vector'].shape == (T, *state_shape_dict['vector'])

    print(f"  - data['next_world_rgb'].shape: \t{data['next_world_rgb'].shape} (Expected: {(T, *global_shape_rgb)})")
    assert data['next_world_rgb'].shape == (T, *global_shape_rgb)
    
    agent_id_0 = agent_ids[0]
    print(f"  - data['{agent_id_0}']['obs_rgb'].shape: \t{data[agent_id_0]['obs_rgb'].shape} (Expected: {(T, *obs_shape_rgb)})")
    assert data[agent_id_0]['obs_rgb'].shape == (T, *obs_shape_rgb)
    print(f"  - data['{agent_id_0}']['obs_vector'].shape: {data[agent_id_0]['obs_vector'].shape} (Expected: {(T, obs_vector_dim)})")
    assert data[agent_id_0]['obs_vector'].shape == (T, obs_vector_dim)

    print("✅ On-Policy buffer validation successful!")

    env.close()