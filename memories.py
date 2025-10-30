from collections import deque
from typing import Dict, List, Any
import numpy as np

class OffPolicyReplayBuffer:
    """
    An episodic experience replay buffer optimized for off-policy multi-agent reinforcement learning.
    """
    def __init__(self, env_info: dict, capacity: int = 5000):
        self.env_info = env_info
        self.n_agents = env_info["n_agents"]                   # num of agents
        self.agent_ids = env_info["agent_ids"]                 # list of agent IDs
        self.buffer = deque(maxlen=int(capacity))              # episode-level buffer
        self._current_episode = self._init_episode_buffer()    # current episode buffer
        self._current_episode_len = 0                          # length of current episode

    def _init_episode_buffer(self) -> Dict[str, List]:
        """ Initialize an empty episode buffer. """
        return {                                                   # episode-level storage
            "state": [], "obs": [], "actions": [], "rewards": [],
            "dones": [], "next_state": [], "next_obs": []
        }
    
    def push(self, 
            obs_dict: Dict[str, np.ndarray],         # observations for all agents (C, H, W)
            state: np.ndarray,                       # global state (state_dim,)
            actions_dict: Dict[str, int],            # actions for all agents
            rewards_dict: Dict[str, float],          # rewards for all agents
            dones_dict: Dict[str, bool],             # done flags for all agents
            next_obs_dict: Dict[str, np.ndarray],    # next observations for all agents (C, H, W)
            next_state: np.ndarray):                 # next global state (state_dim,)
        """ Add a time step of data to the current episode buffer. """
        obs_array = np.stack([obs_dict[aid] for aid in self.agent_ids], axis=0)             # (n_agents, C, H, W)
        actions_array = np.array([actions_dict[aid] for aid in self.agent_ids])             # (n_agents,)
        next_obs_array = np.stack([next_obs_dict[aid] for aid in self.agent_ids], axis=0)   # (n_agents, C, H, W)
        
        reward_global = sum(rewards_dict.values())    # global reward as sum of individual rewards
        done_global = any(dones_dict.values())        # global done as any individual done
        
        self._current_episode["obs"].append(obs_array)            
        self._current_episode["state"].append(state)
        self._current_episode["actions"].append(actions_array)
        self._current_episode["rewards"].append(reward_global)
        self._current_episode["dones"].append(done_global)
        self._current_episode["next_obs"].append(next_obs_array)
        self._current_episode["next_state"].append(next_state)
        
        self._current_episode_len += 1
        if done_global:
            self.commit_episode()

    def commit_episode(self):
        """ Commit the current episode buffer to the main buffer. """
        if self._current_episode_len == 0:
            return
        episode_batch = {}                                    # convert lists to arrays
        for key, data_list in self._current_episode.items():
            episode_batch[key] = np.stack(data_list, axis=0)  # (T, ...)
        self.buffer.append(episode_batch)                     # add episode to main buffer
        self._current_episode = self._init_episode_buffer()   # reset current episode buffer
        self._current_episode_len = 0

    def sample(self, batch_size: int, sequence_length: int) -> Dict[str, np.ndarray]:
        """ Sample a batch of sequences from the buffer. """
        if len(self.buffer) == 0:
            raise ValueError("Not enough episodes in the buffer to sample from (buffer is empty).")
            
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        batch_list = []
        sampled_ep_indices = np.random.choice(                # sample episodes
            len(self.buffer), batch_size, replace=True
        )
        
        for ep_idx in sampled_ep_indices:
            ep_data = self.buffer[ep_idx]
            ep_len = ep_data["dones"].shape[0]

            if ep_len < sequence_length:
                # 这个错误不应该发生，如果它发生了，说明 "warm-up" 步数不足
                raise ValueError(
                    f"采样到一个过短的回合 (长度 {ep_len}), "
                    f"无法采样序列 (长度 {sequence_length}). "
                    f"请增加训练前的 'warm-up' 步数。"
                )
            
            max_start_idx = ep_len - sequence_length
            start_idx = np.random.randint(0, max_start_idx + 1)
            
            sliced_ep = {}
            for key, data in ep_data.items():
                sliced_ep[key] = data[start_idx : start_idx + sequence_length]
            batch_list.append(sliced_ep)

        # 将所有采样到的序列 (B 个) 堆叠成一个批次
        final_batch = {}
        keys = batch_list[0].keys() 
        for key in keys:
            final_batch[key] = np.stack([ep_slice[key] for ep_slice in batch_list], axis=0)

        return final_batch

    def __len__(self) -> int:
        return sum(ep["dones"].shape[0] for ep in self.buffer)
    
class OnPolicyRolloutBuffer:
    """
    一个为 On-Policy MARL 优化的临时数据缓冲区 (Rollout Storage)。
    
    它只收集数据，不进行计算。
    它必须在每次训练迭代后被清空 (clear())。
    """
    def __init__(self, env_info: dict):
        self.env_info = env_info
        self.n_agents = env_info["n_agents"]
        self.agent_ids = env_info["agent_ids"]
        # 移除 gamma 和 gae_lambda
        self.storage = self._init_storage()

    def _init_storage(self) -> Dict[str, Any]:
        """初始化空的列表来存储轨迹"""
        storage = {
            "global_state": [],
            "rewards": [],
            "dones": [],
            "next_global_state": [], # GAE 需要 s_{t+1}
        }
        for agent_id in self.agent_ids:
            storage[agent_id] = {
                "obs": [],
                "actions": [],
                "log_probs": [], # PPO 需要
                "values": [],    # PPO 需要
                "next_obs": [],  # GAE 需要 o_{t+1}
            }
        return storage

    def push(self, 
             obs_dict: Dict[str, np.ndarray], 
             state: np.ndarray, 
             actions_dict: Dict[str, int], 
             rewards_dict: Dict[str, float], 
             dones_dict: Dict[str, bool],
             log_probs_dict: Dict[str, float], # (由 PPO 策略提供)
             values_dict: Dict[str, float],    # (由 PPO 策略提供)
             next_obs_dict: Dict[str, np.ndarray], # (来自 env.step)
             next_state: np.ndarray):            # (来自 env.step)
        """
        向临时缓冲区中添加一个时间步的数据。
        """
        self.storage["global_state"].append(state)
        self.storage["next_global_state"].append(next_state)
        self.storage["rewards"].append(sum(rewards_dict.values()))
        self.storage["dones"].append(any(dones_dict.values()))
        
        for agent_id in self.agent_ids:
            self.storage[agent_id]["obs"].append(obs_dict[agent_id])
            self.storage[agent_id]["next_obs"].append(next_obs_dict[agent_id])
            self.storage[agent_id]["actions"].append(actions_dict[agent_id])
            self.storage[agent_id]["log_probs"].append(log_probs_dict[agent_id])
            self.storage[agent_id]["values"].append(values_dict[agent_id])

    def get_all_data(self) -> Dict[str, Any]:
        """
        将所有存储的数据转换为 NumPy 数组并返回。
        GAE/Returns 必须在外部计算。
        """
        data = {
            "global_state": np.array(self.storage["global_state"]),
            "next_global_state": np.array(self.storage["next_global_state"]),
            "rewards": np.array(self.storage["rewards"]),
            "dones": np.array(self.storage["dones"]),
        }
        
        for agent_id in self.agent_ids:
            agent_data = self.storage[agent_id]
            data[agent_id] = {
                "obs": np.array(agent_data["obs"]),
                "next_obs": np.array(agent_data["next_obs"]),
                "actions": np.array(agent_data["actions"]),
                "log_probs": np.array(agent_data["log_probs"]),
                "values": np.array(agent_data["values"]),
            }
            
        return data

    def clear(self) -> None:
        """清空所有临时数据，为下一次 rollout 做准备"""
        self.storage = self._init_storage()

    def __len__(self) -> int:
        """返回当前存储的 *时间步数*"""
        return len(self.storage["global_state"])
    
if __name__ == "__main__":
    
    print("="*40)
    print("开始验证 MARL 缓冲区模块...")
    print("="*40)
    
    # --- 初始化环境 ---
    from MeltingPotWrapper import GeneralMeltingPotWrapper
    env = GeneralMeltingPotWrapper(substrate_name="collaborative_cooking__asymmetric")
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    agent_ids = env_info["agent_ids"]
    
    print("环境信息:")
    print(f"  - n_agents: {n_agents}, agent_ids: {agent_ids}")
    print(f"  - obs_shape: {env_info['obs_shape']}, state_shape: {env_info['state_shape']}")
    print(f"  - episode_limit: {env_info['episode_limit']}")

    # ======================================================
    # 验证 1: OffPolicyReplayBuffer (QMIX)
    # ======================================================
    print("\n" + "="*20)
    print("验证 1: OffPolicyReplayBuffer (用于 QMIX)")
    print("="*20)
    
    off_policy_buffer = OffPolicyReplayBuffer(env_info, capacity=10)
    
    # 填充缓冲区 (2 个回合)
    print("正在填充 Off-Policy 缓冲区 (2 个回合)...")
    N_EPISODES = 2
    # --- [修改点 1 验证] ---
    # 我们需要确保回合长度 > SEQUENCE_LENGTH，否则会触发错误
    # (幸运的是, 我们的 episode_limit=1000 远大于8)
    #
    for ep in range(N_EPISODES):
        obs_dict, state = env.reset()
        done = False
        while not done:
            actions_dict = {aid: np.random.randint(0, env.n_actions) for aid in agent_ids}
            next_obs_dict, next_state, rewards_dict, dones_dict, _ = env.step(actions_dict)
            done = any(dones_dict.values())
            off_policy_buffer.push(
                obs_dict, state, actions_dict, rewards_dict, dones_dict,
                next_obs_dict, next_state
            )
            obs_dict, state = next_obs_dict, next_state
    
    print(f"  - 缓冲区回合数: {len(off_policy_buffer.buffer)} (应为 {N_EPISODES})")
    print(f"  - 缓冲区总步数: {len(off_policy_buffer)} (应为 {N_EPISODES * env_info['episode_limit']})")

    # 采样
    BATCH_SIZE = 4
    SEQUENCE_LENGTH = 8
    print(f"正在采样 (B={BATCH_SIZE}, L={SEQUENCE_LENGTH})...")
    
    try:
        batch = off_policy_buffer.sample(BATCH_SIZE, SEQUENCE_LENGTH)
        print("  - 采样成功。")
        print(f"  - batch['state'].shape: {batch['state'].shape} (预期: {(BATCH_SIZE, SEQUENCE_LENGTH, env_info['state_shape'])})")
        assert batch['state'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, env_info['state_shape'])
        print("✅ Off-Policy 缓冲区验证成功!")
    
    except Exception as e:
        print(f"\n!!! Off-Policy 采样失败: {e}")
        import traceback
        traceback.print_exc()

    
    # ======================================================
    # 验证 2: OnPolicyRolloutBuffer (MAPPO)
    # ======================================================
    print("\n" + "="*20)
    print("验证 2: OnPolicyRolloutBuffer (用于 MAPPO)")
    print("="*20)
    
    # --- [修改点 2 验证] ---
    on_policy_buffer = OnPolicyRolloutBuffer(env_info)
    
    # 填充缓冲区 (1 个回合)
    print("正在填充 On-Policy 缓冲区 (1 个回合)...")
    obs_dict, state = env.reset()
    done = False
    
    while not done:
        actions_dict = {aid: np.random.randint(0, env.n_actions) for aid in agent_ids}
        log_probs_dict = {aid: np.random.rand() for aid in agent_ids} # 伪造
        values_dict = {aid: np.random.rand() for aid in agent_ids} # 伪造
        
        next_obs_dict, next_state, rewards_dict, dones_dict, _ = env.step(actions_dict)
        done = any(dones_dict.values())

        # 推入所有数据
        on_policy_buffer.push(
            obs_dict, state, actions_dict, rewards_dict, dones_dict,
            log_probs_dict, values_dict,
            next_obs_dict, next_state # <-- 新增
        )
        obs_dict, state = next_obs_dict, next_state

    print(f"  - 缓冲区已填充步数: {len(on_policy_buffer)} (应为 {env_info['episode_limit']})")
    assert len(on_policy_buffer) == env_info['episode_limit']

    # 获取数据 (不再计算优势)
    print(f"正在获取所有数据...")
    data = on_policy_buffer.get_all_data()
    
    print("  - 获取数据成功。验证数据形状:")
    T = env_info['episode_limit'] # 轨迹长度
    
    print(f"  - data['global_state'].shape: \t{data['global_state'].shape} (预期: {(T, env_info['state_shape'])})")
    assert data['global_state'].shape == (T, env_info['state_shape'])
    
    print(f"  - data['next_global_state'].shape: {data['next_global_state'].shape} (预期: {(T, env_info['state_shape'])})")
    assert data['next_global_state'].shape == (T, env_info['state_shape'])

    print(f"  - data['rewards'].shape: \t{data['rewards'].shape} (预期: {(T,)})")
    assert data['rewards'].shape == (T,)
    
    print(f"  - data['player_0']['obs'].shape: \t{data['player_0']['obs'].shape} (预期: {(T, *env_info['obs_shape'])})")
    assert data['player_0']['obs'].shape == (T, *env_info['obs_shape'])

    print(f"  - data['player_0']['next_obs'].shape: \t{data['player_0']['next_obs'].shape} (预期: {(T, *env_info['obs_shape'])})")
    assert data['player_0']['next_obs'].shape == (T, *env_info['obs_shape'])
    
    # 清空缓冲区
    on_policy_buffer.clear()
    print("\n  - 缓冲区已清空。")
    print(f"  - 缓冲区当前步数: {len(on_policy_buffer)} (应为 0)")
    assert len(on_policy_buffer) == 0
    
    print("✅ On-Policy 缓冲区验证成功!")
    
    env.close()