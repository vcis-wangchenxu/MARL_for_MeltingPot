"""==================================================================================================================================================
文件名: check_obs.py

功能描述:
该文件定义了一个通用的 MeltingPot 环境封装器 `MeltingPotPettingZooParallelWrapper`，用于将 MeltingPot 环境适配为 PettingZoo 的并行环境接口。主要功能包括：

1. **支持两种模式**:
   - `self_play` 模式：加载 MeltingPot 的 Substrate 环境。
   - `mixed_control` 模式：加载 MeltingPot 的 Scenario 环境。

2. **动态处理观测和信息**:
   - 将非标准的、全局的观测键（如 `WORLD.RGB`）移至 `info` 字典。
   - 仅保留标准的、局部的智能体观测键（如 `RGB`, `HUNGER` 等）。

3. **动态生成空间**:
   - 根据 MeltingPot 环境的 `dm_env` 规范动态生成 `observation_space` 和 `action_space`，避免硬编码。

4. **渲染支持**:
   - 支持 `human` 和 `rgb_array` 两种渲染模式。
   - 使用 Pygame 实现人类可视化渲染。

5. **PettingZoo API**:
   - 实现了 `reset`、`step`、`render` 和 `close` 等标准接口，方便与 PettingZoo 框架集成。

6. **测试功能**:
   - 提供了测试代码，验证封装器在不同场景和模式下的功能，包括环境重置、步进、观测和信息的正确性检查，以及渲染功能。

该封装器代码具有较强的通用性，支持不同的 MeltingPot 场景和模式，适用于强化学习实验中的多智能体环境适配。
=================================================================================================================================================="""

import dm_env
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import time
import pygame

from pettingzoo.utils.env import ParallelEnv
from meltingpot import substrate as mp_substrate
from meltingpot import scenario as mp_scenario
from pettingzoo.utils import parallel_to_aec
import logging
logging.getLogger('absl').setLevel(logging.ERROR) # 抑制 absl 的 INFO 日志

class MeltingPotPettingZooParallelWrapper(ParallelEnv):
    """
    该类是一个通用的封装器，用于将 MeltingPot 环境适配为 PettingZoo 的并行环境接口。
    支持 `self_play` 和 `mixed_control` 两种模式，动态处理观测和动作空间，支持渲染功能，
    并实现了 PettingZoo 的标准 API（如 `reset`、`step` 和 `render`）。适用于多智能体强化学习实验。
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "name": "MeltingPot"}

    def __init__(self, level_name: str, mode: str = "self_play", 
                 render_mode: str = None):
        super().__init__()

        self.mode = mode                      # "self_play" 或 "mixed_control"
        self.level_name = level_name          # MeltingPot 场景名称
        self._mp_env = None                   # 内部 MeltingPot 环境
        self.latest_global_rgb = None         # 最新的全局渲染帧
        
        self.render_mode = render_mode        # None, "human", 或 "rgb_array"
        self.screen = None                    # Pygame 窗口
        self.clock = None                     # Pygame 时钟
        self.screen_size = None               # 渲染屏幕大小
        
        self.global_render_key = "WORLD.RGB"  # 全局渲染观测键
        
        self.PERMITTED_LOCAL_OBSERVATIONS = frozenset({    # 允许的局部观测键
            'RGB', 'HUNGER', 'INVENTORY', 'MY_OFFER', 'OFFERS',
            'READY_TO_SHOOT', 'STAMINA', 'VOTING'
        })
        
        if self.mode == "self_play":
            # 如果模式为 "self_play"（自博弈模式），加载 MeltingPot 的 Substrate 环境。
            # 1. 根据关卡名称获取配置（config），并提取默认玩家角色（roles）。
            # 2. 使用配置和角色构建 Substrate 环境。
            # 3. 设置智能体数量为角色数量。
            print(f"模式: 'self_play'。加载 Substrate: '{self.level_name}'")
            self.config = mp_substrate.get_config(self.level_name)
            self.roles = self.config.default_player_roles
            self._mp_env = mp_substrate.build(self.level_name, roles=self.roles)
            num_agents = len(self.roles)
        
        elif self.mode == "mixed_control":
            # 如果模式为 "mixed_control"（混合控制模式），加载 MeltingPot 的 Scenario 环境。
            # 1. 根据关卡名称获取配置（config）。
            # 2. 应用一个空的 substrate_transform，确保保留全局渲染键（如 "WORLD.RGB"）用于渲染。
            # 3. 使用配置构建 Scenario 环境。
            # 4. 设置智能体数量为可控智能体（Focal agents）的数量。
            print(f"模式: 'mixed_control'。加载 Scenario: '{self.level_name}'")
            self.config = mp_scenario.get_config(self.level_name)
            
            dummy_transform = lambda substrate: substrate
            print("  > 正在应用 'substrate_transform' 以保留 'WORLD.RGB' 用于渲染。")
            self._mp_env = mp_scenario.build(
                self.level_name, 
                substrate_transform=dummy_transform
            )
            
            num_agents = sum(self.config.is_focal)
            print(f"  总角色数: {len(self.config.roles)}")
            print(f"  可控智能体 (Focal): {num_agents}")
        
        else:
            raise ValueError(f"未知的 mode: '{mode}'")

        self.possible_agents = [f"player_{i}" for i in range(num_agents)]       # 智能体名称列表
        self.agents = self.possible_agents[:]                                   # 当前活跃智能体列表

        self._mp_action_specs = self._mp_env.action_spec()                      # MeltingPot 动作规范
        self._mp_obs_specs = self._mp_env.observation_spec()                    # MeltingPot 观测规范
        
        self._set_screen_size()                                                 # 设置渲染屏幕大小
        
        self.action_spaces = {                                                  # 动作空间字典
            agent: self._convert_spec_to_space(self._mp_action_specs[i])
            for i, agent in enumerate(self.possible_agents)
        }
        self.observation_spaces = {                                             # 观测空间字典
            agent: self._convert_obs_spec_to_space(self._mp_obs_specs[i])
            for i, agent in enumerate(self.possible_agents)
        }

    def _set_screen_size(self):
        """
        该方法用于设置渲染屏幕的大小。通过检查第一个智能体的观测规范，提取全局渲染键 (`WORLD.RGB`) 的形状来确定屏幕尺寸。
        如果未找到全局渲染键或发生错误，则屏幕大小设置失败。

        主要步骤:
        1. 检查是否存在观测规范 (`_mp_obs_specs`)。
        2. 提取第一个智能体的观测规范，检查是否包含全局渲染键。
        3. 如果找到全局渲染键，获取其形状并设置屏幕大小。
        4. 如果未找到全局渲染键或发生异常，打印警告或错误信息。
        """
        if not self._mp_obs_specs: return
        try:
            first_agent_spec = self._mp_obs_specs[0]
            if self.global_render_key in first_agent_spec:
                shape = first_agent_spec[self.global_render_key].shape
                self.screen_size = (shape[1], shape[0])
                print(f"  > 渲染屏幕大小设置为: {self.screen_size}")
            else:
                print(f"警告: '{self.global_render_key}' 不在观测规范中。渲染将不可用。")
        except Exception as e:
            print(f"设置屏幕大小时出错: {e}")

    def _convert_spec_to_space(self, spec) -> spaces.Space:
        """
        该方法将 MeltingPot 环境中的 `dm_env` 规范转换为 `gymnasium` 的空间对象 (`spaces.Space`)。
        根据输入的 `spec` 类型，动态生成对应的动作或观测空间。

        主要步骤:
        1. 如果 `spec` 是 `DiscreteArray` 类型，则转换为离散空间 (`spaces.Discrete`)。
        2. 如果 `spec` 是 `BoundedArray` 类型，则转换为有界连续空间 (`spaces.Box`)，并根据最小值和最大值设置边界。
        3. 如果 `spec` 是 `Array` 类型，则根据数据类型生成无界连续空间或其他适配的空间。
        4. 如果 `spec` 是字典类型，则递归调用自身，将每个键值对转换为对应的空间，并生成字典空间 (`spaces.Dict`)。
        5. 如果 `spec` 类型未知，则抛出异常。

        用途:
            该方法用于动态适配 MeltingPot 环境的动作和观测规范，确保与 PettingZoo 的空间定义兼容。
        """
        if isinstance(spec, dm_env.specs.DiscreteArray):                                # 离散空间
            return spaces.Discrete(spec.num_values)                                     
        elif isinstance(spec, dm_env.specs.BoundedArray):                               # 有界连续空间
            low, high = spec.minimum, spec.maximum
            if np.isscalar(low): low = np.full(spec.shape, low)                             # 扩展标量到数组
            if np.isscalar(high): high = np.full(spec.shape, high)                          # 扩展标量到数组
            return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
        elif isinstance(spec, dm_env.specs.Array):                                      # 无界连续空间或其他
            if np.issubdtype(spec.dtype, np.floating):                                      # 浮点类型
                return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
            elif np.issubdtype(spec.dtype, np.uint8):                                       # 无符号整数类型
                return spaces.Box(0, 255, spec.shape, spec.dtype)
            else:                                                                           # 其他整数类型
                 return spaces.Box(np.iinfo(spec.dtype).min, np.iinfo(spec.dtype).max, spec.shape, spec.dtype)
        elif isinstance(spec, dict):                                                    # 字典类型
            return spaces.Dict({k: self._convert_spec_to_space(v) for k, v in spec.items()})
        else:                                                                           # 其他类型
            raise ValueError(f"未知的 dm_env spec 类型: {type(spec)}") 

    def _convert_obs_spec_to_space(self, spec: dict) -> spaces.Dict:
        """
        该方法将 MeltingPot 环境中的观测规范 (`dm_env` 格式) 转换为 `gymnasium` 的字典空间 (`spaces.Dict`)。
        仅为允许的局部观测键（如 `RGB`, `HUNGER` 等）创建空间，忽略全局渲染键和其他非标准键。

        主要步骤:
        1. 遍历输入的观测规范字典 `spec`。
        2. 如果键是全局渲染键（如 `WORLD.RGB`），跳过处理。
        3. 如果键在允许的局部观测键列表中（`PERMITTED_LOCAL_OBSERVATIONS`），将其添加到新的空间字典中。
        4. 返回包含允许键的 `spaces.Dict` 对象。

        用途:
            该方法用于动态生成局部观测空间，确保仅包含标准的局部观测键，同时忽略全局渲染键和其他非标准键。
        """
        new_spec_dict = {}                                                     # 新的观测空间字典
        for key, value_spec in spec.items():                                   # 遍历观测规范
            if key == self.global_render_key:                                  # 如果是全局渲染键，跳过
                continue
            if key in self.PERMITTED_LOCAL_OBSERVATIONS:                       # 仅保留允许的局部观测键
                new_spec_dict[key] = self._convert_spec_to_space(value_spec)   # 转换并添加到新字典
         
        return spaces.Dict(new_spec_dict)                                      # 返回字典空间

    def _process_obs_and_info(self, mp_obs_list: list) -> tuple[dict, dict]:
        """
        该方法用于处理 MeltingPot 环境返回的观测列表，将观测数据拆分为标准的局部观测 (`obs`) 和附加信息 (`info`) 两部分。
        全局渲染键（如 `WORLD.RGB`）和其他非标准键会被移至 `info`，仅保留允许的局部观测键。

        主要步骤:
        1. 提取全局渲染观测（如 `WORLD.RGB`），并将其存储在 `info` 的 `global_obs` 中。
        2. 遍历每个智能体的观测数据：
            - 如果键是全局渲染键，跳过处理（已在全局信息中处理）。
            - 如果键是允许的局部观测键（如 `RGB`, `HUNGER` 等），将其保留在 `obs` 中。
            - 其他键（如 `COLLECTIVE_REWARD` 等）会被移至 `info`。
        3. 返回两个字典：
            - `obs`：包含每个智能体的局部观测。
            - `info`：包含每个智能体的附加信息和全局观测。

        用途:
            该方法确保观测数据的标准化处理，便于强化学习算法使用，同时保留额外信息供调试或分析。
        """
        all_obs = {}                       # 存储所有智能体的观测
        all_infos = {}                     # 存储所有智能体的 info
        
        global_info_payload = {}           # 全局信息载荷
        
        # 1. 提取全局渲染观测
        if mp_obs_list:                    # 确保观测列表非空
            first_obs = mp_obs_list[0]     # 第一个智能体的观测
            if self.global_render_key in first_obs:                        # 如果存在全局渲染键
                global_render_frame = first_obs[self.global_render_key]    # 提取渲染帧
                global_info_payload["global_obs"] = {                      # 存储全局观测
                    self.global_render_key: global_render_frame
                }
                self.latest_global_rgb = global_render_frame               # 更新最新全局渲染帧

        # 2. 为每个智能体处理 obs 和 info
        for i, agent_name in enumerate(self.possible_agents):              # 遍历所有智能体
            agent_obs_full = mp_obs_list[i]                                # 智能体的完整观测
            agent_obs_filtered = {}                                        # 过滤后的观测
            agent_info = global_info_payload.copy()                        # 初始化 info，包含全局信息
            
            for key, value in agent_obs_full.items():                      # 遍历观测键值对
                if key == self.global_render_key:                          # 全局渲染键已处理，跳过
                    pass
                elif key in self.PERMITTED_LOCAL_OBSERVATIONS:             # 允许的局部观测键
                    agent_obs_filtered[key] = value                        # 保留在观测中
                else:                                                      # 其他键，移至 info                            
                    agent_info[key] = value                                # 添加到 info
            
            all_obs[agent_name] = agent_obs_filtered                       # 存储过滤后的观测
            all_infos[agent_name] = agent_info                             # 存储 info
            
        return all_obs, all_infos                                          # 返回 obs 和 info

    def _init_pygame(self):
        """
        该方法用于初始化 Pygame 窗口，以支持 `human` 渲染模式下的可视化显示。

        主要步骤:
        1. 检查是否已初始化 Pygame 窗口 (`self.screen`)。
        2. 如果未安装 Pygame 或屏幕大小未知，禁用 `human` 渲染模式并打印警告信息。
        3. 初始化 Pygame，设置窗口标题和屏幕大小。
        4. 创建 Pygame 时钟对象，用于控制渲染帧率。

        用途:
            该方法确保在 `human` 渲染模式下正确初始化 Pygame 窗口，以便实时显示环境的渲染画面。
        """
        if self.screen is None and self.render_mode == "human":               # 仅在 human 模式下初始化
            if pygame is None: raise ImportError("Pygame 未安装。")            # 检查 Pygame 安装
            if self.screen_size is None:                                      # 检查屏幕大小
                print("警告: 屏幕大小未知。'human' 渲染已禁用。")
                self.render_mode = None                                       # 禁用 human 模式
                return
            pygame.init()                                                     # 初始化 Pygame
            pygame.display.set_caption(f"MeltingPot - {self.level_name}")     # 设置窗口标题
            self.screen = pygame.display.set_mode(self.screen_size)           # 创建窗口
            self.clock = pygame.time.Clock()                                  # 创建时钟对象

    def render(self):
        """
        该方法实现了 PettingZoo 的 `render` 接口，用于渲染环境的当前状态。支持两种渲染模式：
        1. `rgb_array` 模式：返回当前环境的渲染帧（以 NumPy 数组形式表示）。
        2. `human` 模式：通过 Pygame 显示实时渲染画面。

        主要步骤:
        1. 检查渲染模式是否为 `None` 或当前没有全局渲染帧，若是则直接返回 `None`。
        2. 如果渲染模式为 `rgb_array`，返回当前的全局渲染帧。
        3. 如果渲染模式为 `human`：
            - 初始化 Pygame 窗口（若尚未初始化）。
            - 将当前渲染帧转换为 Pygame 表面并显示在窗口中。
            - 处理 Pygame 事件（如窗口关闭事件）。
            - 控制渲染帧率（默认为 60 FPS）。

        用途:
            该方法用于可视化环境的当前状态，便于调试和分析。
        """
        if self.render_mode is None or self.latest_global_rgb is None:    # 如果未设置渲染模式或无渲染帧，返回 None
            return None
            
        if self.render_mode == "rgb_array":                               # 如果是 rgb_array 模式，返回渲染帧
            return self.latest_global_rgb

        if self.render_mode == "human":                                   # 如果是 human 模式，使用 Pygame 显示
            self._init_pygame()
            if self.screen is None: return None                           # 如果 Pygame 未初始化，返回 None

            frame = np.transpose(self.latest_global_rgb, (1, 0, 2))       # 转置为 Pygame 格式
            surface = pygame.surfarray.make_surface(frame)                # 创建 Pygame 表面
            self.screen.blit(surface, (0, 0))                             # 绘制到屏幕
            pygame.display.flip()                                         # 更新显示

            for event in pygame.event.get():                              # 处理 Pygame 事件
                if event.type == pygame.QUIT:                             # 如果窗口关闭事件
                    print("Pygame 窗口已关闭。")
                    self.close()                                          # 关闭环境
            
            self.clock.tick(60)                                           # 控制帧率为 60 FPS
            return None

    def reset(self, seed=None, options=None):
        """
        该方法实现了 PettingZoo 的 `reset` 接口，用于重置环境到初始状态。重置后，所有智能体的观测和附加信息会被返回。

        主要步骤:
        1. 重置环境内部的 MeltingPot 环境 (`self._mp_env`)。
        2. 调用 `_process_obs_and_info` 方法，将 MeltingPot 返回的观测数据拆分为局部观测 (`obs`) 和附加信息 (`info`)。
        3. 更新当前活跃的智能体列表 (`self.agents`)。
        4. 如果设置了渲染模式，调用 `render` 方法进行初始渲染。
        5. 返回包含所有智能体的观测和附加信息的字典。

        返回值:
            - `observations`：包含每个智能体的局部观测数据。
            - `infos`：包含每个智能体的附加信息（如全局观测等）。

        用途:
            该方法用于在每个新实验或新回合开始时重置环境，确保所有状态和数据从初始状态开始。
        """
        self.agents = self.possible_agents[:]                                            # 重置活跃智能体列表
        self._mp_timestep = self._mp_env.reset()                                         # 重置 MeltingPot 环境
        observations, infos = self._process_obs_and_info(self._mp_timestep.observation)  # 处理观测和 info
        self.render()                                                                    # 初始渲染
        return observations, infos                                                       # 返回观测和 info

    def step(self, actions: dict):
        """
        该方法实现了 PettingZoo 的 `step` 接口，用于执行环境中的一步操作。
        根据输入的动作字典，更新环境状态并返回新的观测、奖励、终止标志、截断标志和附加信息。

        主要步骤:
        1. 将输入的动作字典转换为 MeltingPot 环境所需的动作列表。
        2. 调用 MeltingPot 环境的 `step` 方法，执行一步操作。
        3. 调用 `_process_obs_and_info` 方法，将 MeltingPot 返回的观测数据拆分为局部观测 (`obs`) 和附加信息 (`info`)。
        4. 根据 MeltingPot 的 `step_type` 判断是否为最后一步，设置终止标志和截断标志。
        5. 如果环境结束，清空当前活跃的智能体列表。
        6. 如果设置了渲染模式，调用 `render` 方法进行渲染。
        7. 返回包含观测、奖励、终止标志、截断标志和附加信息的字典。

        返回值:
            - `observations`：包含每个智能体的局部观测数据。
            - `rewards`：包含每个智能体的奖励值。
            - `terminations`：包含每个智能体的终止标志。
            - `truncations`：包含每个智能体的截断标志。
            - `infos`：包含每个智能体的附加信息。

        用途:
            该方法用于在环境中执行一步操作，更新状态并返回相关数据，便于强化学习算法进行训练。
        """
        mp_actions = [actions[agent] for agent in self.possible_agents]                   # 转换为动作列表
        self._mp_timestep = self._mp_env.step(mp_actions)                                 # 执行一步操作
        
        observations, infos = self._process_obs_and_info(self._mp_timestep.observation)   # 处理观测和 info
        rewards = {                                                                       # 提取奖励
            agent: self._mp_timestep.reward[i]
            for i, agent in enumerate(self.possible_agents)
        }
        
        is_done = self._mp_timestep.step_type == dm_env.StepType.LAST                      # 检查是否为最后一步
        terminations = {agent: is_done for agent in self.possible_agents}                  # 终止标志
        truncations = {agent: is_done for agent in self.possible_agents}                   # 截断标志
        
        if is_done: self.agents = []                                                       # 如果结束，清空活跃智能体列表
        self.render()                                                                      # 渲染当前状态
        return observations, rewards, terminations, truncations, infos                     # 返回结果

    def close(self):
        """
        方法名: close(self)

        功能描述:
        该方法实现了 PettingZoo 的 `close` 接口，用于关闭环境并释放相关资源，包括 Pygame 窗口和 MeltingPot 环境。

        主要步骤:
        1. 如果 Pygame 窗口已初始化，关闭窗口并退出 Pygame。
        2. 调用 MeltingPot 环境的 `close` 方法，释放环境资源。

        用途:
            该方法用于在实验结束时清理资源，确保不会占用多余的系统内存或窗口资源。
        """
        if self.screen is not None:                          # 如果 Pygame 窗口已初始化
            pygame.display.quit()                            # 关闭 Pygame 窗口
            pygame.quit()                                    # 退出 Pygame
            self.screen = None                               # 重置屏幕引用
            print("Pygame 已关闭。")    
        self._mp_env.close()                                 # 关闭 MeltingPot 环境

    def observation_space(self, agent):
        """
        获取指定智能体的观测空间。
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        获取指定智能体的动作空间。
        """
        return self.action_spaces[agent]

if __name__ == "__main__":
    
    def test_env(env: MeltingPotPettingZooParallelWrapper):
        """用于测试封装器的辅助函数 (已更新为循环)"""
        print("\n" + "="*50)
        print(f"--- 正在测试模式: '{env.mode}' (Render: {env.render_mode}) ---")
        print(f"--- 关卡: '{env.level_name}' ---")
        print(f"--- 智能体: {env.possible_agents} ---")
        print("="*50)
        
        try:
            print("\n[调用 env.reset()]")
            observations, infos = env.reset()
            first_agent = env.possible_agents[0]

            print(f"  > reset() 成功。")
            print("\n  --- 观测空间 (Agent 0) ---")
            print(f"  {env.observation_space(first_agent)}")
            print("\n  --- 实际观测 (Agent 0) ---")
            print(f"  Obs keys: {list(observations[first_agent].keys())}")
            print("\n  --- 实际 Info (Agent 0) ---")
            print(f"  Info keys: {list(infos[first_agent].keys())}")

            # 自动检查
            print("\n  --- 自动检查 ---")
            info_keys = list(infos[first_agent].keys())
            obs_keys = list(observations[first_agent].keys())

            assert "global_obs" in info_keys, "全局观测 'global_obs' 不在 info 中"
            assert "WORLD.RGB" not in obs_keys, "'WORLD.RGB' 不应在 obs 中"
            print("  > [检查通过] 全局观测 (WORLD.RGB) 已成功移至 info['global_obs']")
            
            if "COLLECTIVE_REWARD" in infos[first_agent]:
                assert "COLLECTIVE_REWARD" in info_keys, "团队奖励 'COLLECTIVE_REWARD' 不在 info 中"
                assert "COLLECTIVE_REWARD" not in obs_keys, "团队奖励 'COLLECTIVE_REWARD' 不应在 obs 中"
                print("  > [检查通过] 团队奖励 (COLLECTIVE_REWARD) 已成功移至 info")
            
            if "NUM_OTHERS_WHO_CLEANED_THIS_STEP" in infos[first_agent]:
                 assert "NUM_OTHERS_WHO_CLEANED_THIS_STEP" in info_keys
                 assert "NUM_OTHERS_WHO_CLEANED_THIS_STEP" not in obs_keys
                 print("  > [检查通过] 特殊键 (NUM_OTHERS...) 已成功移至 info")
            
            print("\n[调用 env.step() 10 次 (用于 'human' 渲染测试)]")
            for i in range(10):
                actions = {
                    agent: env.action_space(agent).sample() 
                    for agent in env.possible_agents
                }
                obs_step, rew, term, trunc, info_step = env.step(actions)
                if not env.agents:
                    print(f"  > 环境在第 {i} 步结束。正在重置...")
                    env.reset()

            print("\n  > 10 步循环测试完成。")

        except Exception as e:
            print(f"\n*** 测试失败: {e} ***")
            import traceback
            traceback.print_exc()
        finally:
            env.close()
            print("\n--- 环境已关闭 ---")
            print("="*50)

    # ------------------------------------
    # 测试 1: 'clean_up' (自博弈)
    # ------------------------------------
    env_self_play = MeltingPotPettingZooParallelWrapper(
        level_name="clean_up", 
        mode="self_play",
        render_mode="human"
    )
    test_env(env_self_play)

    # ------------------------------------
    # 测试 2: 'collaborative_cooking__asymmetric' (自博弈)
    # ------------------------------------
    print("\n" * 3)
    try:
        env_cooking = MeltingPotPettingZooParallelWrapper(
            level_name="collaborative_cooking__asymmetric",
            mode="self_play",
            render_mode="human"
        )
        test_env(env_cooking)
    except Exception as e:
        print(f"*** 'collaborative_cooking__asymmetric' 加载失败: {e} ***")
        print("这可能是因为该场景名称不正确或依赖项缺失，但封装器代码本身是通用的。")

    # ------------------------------------
    # 测试 3: 'clean_up_0' (混合博弈)
    # ------------------------------------
    print("\n" * 3)
    try:
        env_cooking = MeltingPotPettingZooParallelWrapper(
            level_name="clean_up_0",
            mode="mixed_control",
            render_mode="human"
        )
        test_env(env_cooking)
    except Exception as e:
        print(f"*** 'collaborative_cooking__asymmetric' 加载失败: {e} ***")
        print("这可能是因为该场景名称不正确或依赖项缺失，但封装器代码本身是通用的。")

        