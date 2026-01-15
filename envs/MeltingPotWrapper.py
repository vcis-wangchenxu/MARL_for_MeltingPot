import logging
import os
# os.environ['SDL_AUDIODRIVER'] = 'dummy'
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
from typing import Dict, List, Any, Tuple, Union
import pygame

from meltingpot import substrate
from meltingpot import scenario

from meltingpot.configs import substrates as substrate_configs
from meltingpot.configs import scenarios as scenario_configs

VECTOR_OBS_EXCLUDE_KEYS = frozenset(['RGB', 'WORLD.RGB', 'COLLECTIVE_REWARD'])    

def _extract_vector_and_stack(obs_list: List[Dict[str, Any]], 
                        vector_keys: List[str]) -> np.ndarray:
    """
    Extract vector features from a list of agent observations and stack them.
    Returns: np.ndarray of shape (N_Agents, Vector_Dim)
    """
    batch_vectors = []

    for obs in obs_list:
        flat_parts = []
        
        for key in vector_keys:
            if key in obs:
                flat_parts.append(np.asarray(obs[key]).flatten())

        if flat_parts:
            agent_vec = np.concatenate(flat_parts, dtype=np.float32)
            batch_vectors.append(agent_vec)
        else:
            batch_vectors.append(np.array([], dtype=np.float32))
    
    return np.stack(batch_vectors, axis=0) # (N_Agents, Vector_Dim)

def _process_rgb_and_stack(obs_list: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract RGB images, transpose to (C, H, W), normalize, and stack.
    Returns: np.ndarray of shape (N_Agents, C, H, W) or empty array.
    """
    processed_rgbs = []
    for obs in obs_list:
        if 'RGB' in obs:
            # (H, W, C) -> (C, H, W) and Normalize 0-1
            obs_rgb = obs['RGB'].astype(np.float32) / 255.0
            obs_rgb = np.transpose(obs_rgb, (2, 0, 1)) 
            processed_rgbs.append(obs_rgb)
    
    if not processed_rgbs:
        return np.array([], dtype=np.float32)
        
    # Stack -> (N, C, H, W)
    return np.stack(processed_rgbs, axis=0)

class _SubstrateWrapper:
    """ Wrapper for Melting Pot Substrate (Controls All Agents) """
    
    def __init__(self, substrate_name: str):
        print(f"[Wrapper] Initializing Substrate: {substrate_name}...")
        self.substrate_name = substrate_name
        self._config = substrate.get_config(self.substrate_name)
        self._roles = self._config.default_player_roles
        self.n_agents = len(self._roles)
        
        self._action_set = self._config.action_set
        self.n_actions = self._config.action_spec.num_values

        self._env = substrate.build_from_config(self._config, roles=self._roles)

        lab2d_settings = self._config.lab2d_settings_builder(roles=self._roles, config=self._config)
        self.episode_limit = lab2d_settings.get("maxEpisodeLengthFrames", 1000)
        self._step_count = 0

        agent_spec = self._config.timestep_spec.observation
        self.vector_keys = []
        self.obs_vector_dim = 0
        
        for key, spec in agent_spec.items():
            if key in VECTOR_OBS_EXCLUDE_KEYS:    # There is no COLLECTIVE_REWARD here
                continue
            self.obs_vector_dim += int(np.prod(spec.shape))
            self.vector_keys.append(key)
        self.vector_keys.sort()
        
        print(f"[Wrapper] Vector keys: {self.vector_keys} | Dim: {self.obs_vector_dim}")

        self.obs_shape = None
        if "RGB" in agent_spec:
            self._obs_spec = agent_spec["RGB"]             # (C, H, W)
            self.obs_shape = (self._obs_spec.shape[2], 
                              self._obs_spec.shape[0], 
                              self._obs_spec.shape[1])

        self.global_shape = None
        if "WORLD.RGB" in agent_spec:
            self.global_shape = agent_spec["WORLD.RGB"].shape

        print(f"[Wrapper] Substrate initialized. Agents: {self.n_agents}. Obs Shape: {self.obs_shape}")

    def get_env_info(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_shape,             # (C, H, W)
            "obs_vector_dim": self.obs_vector_dim,   # Vector Dim
            "global_shape": self.global_shape        # (H, W, C)
        }

    def _format_obs(self, timestep) -> Dict[str, np.ndarray]:
        """ Converts timestep.observation list -> Stacked Dict of Arrays """
        obs_list = timestep.observation
        
        return {
            "rgb": _process_rgb_and_stack(obs_list),                         # (N_Agents, C, H, W)
            "vector": _extract_vector_and_stack(obs_list, self.vector_keys)  # (N_Agents, Vector_Dim)
        }

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        self._step_count = 0
        self._last_timestep = self._env.reset()
        
        obs = self._format_obs(self._last_timestep)
        info = {} 
        
        return obs, info

    def step(self, actions: Union[List[int], np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """
        Args:
            actions: List or Array of ints, shape (N_Agents,)
        Returns:
            obs: {'rgb': (N, C, H, W), 'vector': (N, V)}
            rewards: (N,)
            dones: (N,)
            info: Dict containing global_state, etc.
        """
        # Ensure actions is a list for dm_env
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
            
        self._last_timestep = self._env.step(actions)
        self._step_count += 1

        obs = self._format_obs(self._last_timestep)

        rewards = np.array(self._last_timestep.reward, dtype=np.float32)
        if rewards.shape == (): 
             rewards = np.zeros(self.n_agents, dtype=np.float32)          # Handle scalar case

        done = self._last_timestep.last() or (self._step_count >= self.episode_limit)
        dones = np.full(self.n_agents, done, dtype=bool)                  # All agents share the same done flag

        info = {}
        raw_obs_0 = self._last_timestep.observation[0]
        
        if self.global_shape and 'WORLD.RGB' in raw_obs_0:
            info['global_state'] = {'world_rgb': raw_obs_0['WORLD.RGB']}
        
        if 'COLLECTIVE_REWARD' in raw_obs_0:
            info['collective_reward'] = raw_obs_0['COLLECTIVE_REWARD']

        return obs, rewards, dones, info

    def render(self) -> np.ndarray:
        if self._last_timestep is None:
            raise RuntimeError("Call reset() first")
        return self._last_timestep.observation[0].get('WORLD.RGB', np.zeros(self.global_shape, dtype=np.uint8))

    def close(self):
        self._env.close()

class _ScenarioWrapper:
    """ Wrapper for Melting Pot Scenario (Controls Focal Agents Only) """
    
    def __init__(self, scenario_name: str):
        print(f"[Wrapper] Initializing Scenario: {scenario_name}...")
        self.scenario_name = scenario_name
        self._config = scenario.get_config(scenario_name)
        self._env = scenario.build(scenario_name)
        
        self._is_focal = self._config.is_focal
        self.n_agents = sum(self._is_focal)
        
        self._action_spec = self._env.action_spec()[0]
        self.n_actions = self._action_spec.num_values
        
        agent_spec = self._env.observation_spec()[0]

        self.vector_keys = []
        self.obs_vector_dim = 0
        for key, spec in agent_spec.items():
            if key in VECTOR_OBS_EXCLUDE_KEYS:     # There has COLLECTIVE_REWARD here
                continue
            self.obs_vector_dim += int(np.prod(spec.shape))
            self.vector_keys.append(key)
        self.vector_keys.sort()
        print(f"[Wrapper] Vector keys: {self.vector_keys} | Dim: {self.obs_vector_dim}")

        self.obs_shape = None
        if "RGB" in agent_spec:
            s = agent_spec["RGB"]
            self.obs_shape = (s.shape[2], 
                              s.shape[0], 
                              s.shape[1])

        self._substrate_config = substrate.get_config(self._config.substrate)
        self.global_shape = None
        if "WORLD.RGB" in self._substrate_config.timestep_spec.observation:
            self.global_shape = self._substrate_config.timestep_spec.observation["WORLD.RGB"].shape

        print(f"[Wrapper] Scenario initialized. Focal Agents: {self.n_agents}")

    def get_env_info(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_shape,
            "obs_vector_dim": self.obs_vector_dim,
            "global_shape": self.global_shape
        }

    def _format_obs(self, timestep) -> Dict[str, np.ndarray]:
        # timestep.observation only contains focal agents in Scenario
        obs_list = timestep.observation
        return {
            "rgb": _process_rgb_and_stack(obs_list),                         # (N_Agents, C, H, W)
            "vector": _extract_vector_and_stack(obs_list, self.vector_keys)  # (N_Agents, Vector_Dim)
        }

    def _get_world_rgb(self) -> np.ndarray:
        # Hack to get global state in Scenario
        try:
            return self._env._substrate.observation()[0]['WORLD.RGB']
        except:
            return np.zeros(self.global_shape, dtype=np.uint8) if self.global_shape else None

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        self._step_count = 0
        self._last_timestep = self._env.reset()

        obs = self._format_obs(self._last_timestep)
        info = {}

        return obs, info

    def step(self, actions: Union[List[int], np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
            
        self._last_timestep = self._env.step(actions)
        self._step_count += 1
        
        obs = self._format_obs(self._last_timestep)

        rewards = np.array(self._last_timestep.reward, dtype=np.float32)
        
        lab2d_settings = self._substrate_config.lab2d_settings_builder(
            roles=self._substrate_config.default_player_roles, 
            config=self._substrate_config
        )
        limit = lab2d_settings.get("maxEpisodeLengthFrames", 1000)
        
        done = self._last_timestep.last() or (self._step_count >= limit)
        dones = np.full(self.n_agents, done, dtype=bool)

        info = {}
        if self.global_shape:
            info['global_state'] = {'world_rgb': self._get_world_rgb()}
            
        if self._last_timestep.observation and 'COLLECTIVE_REWARD' in self._last_timestep.observation[0]:
            info['collective_reward'] = self._last_timestep.observation[0]['COLLECTIVE_REWARD']

        return obs, rewards, dones, info

    def render(self) -> np.ndarray:
        return self._get_world_rgb()

    def close(self):
        self._env.close()

def build_meltingpot_env(env_name: str):
    """
    Constructs a Melting Pot environment (either Substrate or Scenario)
    and returns a wrapper providing a unified API interface.
    """

    # Check if env_name is in the imported SUBSTRATES collection
    if env_name in substrate_configs.SUBSTRATES:
        return _SubstrateWrapper(substrate_name=env_name)

    # Check if env_name is in the imported SCENARIO_CONFIGS dictionary
    elif env_name in scenario_configs.SCENARIO_CONFIGS:
        return _ScenarioWrapper(scenario_name=env_name)

    # Otherwise, raise an error
    else:
        raise ValueError(
            f"Unknown Melting Pot environment name: '{env_name}'.\n"
            f"It is neither in meltingpot.configs.substrates.SUBSTRATES nor in meltingpot.configs.scenarios.SCENARIO_CONFIGS."
        )

def run_pygame_test(env, test_name: str, scale_factor: int = 10, fps: int = 10, steps: int = 100):
    """
    Run a brief test loop using Pygame.

    Args:
        env: An initialized _SubstrateWrapper or _ScenarioWrapper instance
        test_name: Title for the display window ("Substrate" or "Scenario")
        scale_factor: Factor to scale up the images (original (40, 72) is too small)
        fps: Rendering frame rate
        steps: Total number of steps to execute
    """
    
    print(f"\n--- Launching Pygame test: {test_name} ---")
    print(f"--- Running {steps} steps... (Press 'Q' or close the window to exit) ---")

    pygame.init()

    # Get the correct render shape (H, W, C) from env_info
    render_shape = env.get_env_info()['global_shape']
    if not render_shape:
        print("Skipping: No global shape found.")
        return
    
    H, W, C = render_shape
    # Pygame uses (Width, Height)
    screen_size = (W * scale_factor, H * scale_factor)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(f"Test: {test_name}")
    clock = pygame.time.Clock()

    # Reset the environment
    env.reset()
    
    running = True
    for i in range(steps):
        if not running:
            break

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Get the render frame
        frame = env.render() # (H, W, C)

        # Prepare Pygame Surface
        # Pygame surface expects (W, H, C), so we swap H and W axes
        frame = frame.swapaxes(0, 1) # (W, H, C)
        surface = pygame.surfarray.make_surface(frame)

        # Scale and draw to the screen
        scaled_surface = pygame.transform.scale(surface, screen_size)
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        # Generate random actions
        # Pygame test does not need to access obs or rewards, we only test rendering
        actions = np.random.randint(0, env.get_env_info()['n_actions'], 
                                    size=env.get_env_info()['n_agents'])

        # Execute one step
        _, _, dones, info = env.step(actions)

        # Print collective reward (if available)
        if 'collective_reward' in info:
            print(f"Step {i} | CR: {info['collective_reward']}", end='\r')

        # Control frame rate
        clock.tick(fps)
        
        if np.any(dones):
            env.reset()

    pygame.quit()
    print(f"\n--- Pygame test: {test_name} completed ---")

if __name__ == "__main__":

    # --- Global Settings ---

    # !!! Set to True to pop up the Pygame window for testing !!!
    SHOW_RENDER_WITH_PYGAME = True
    
    # --- Test 1: Substrate ---
    print("\n" + "="*40 + "\nTesting Substrate (clean_up)\n" + "="*40)
    env = build_meltingpot_env("clean_up")
    obs, info = env.reset()
    
    print(f"Obs Keys: {obs.keys()}")
    print(f"Obs RGB Shape: {obs['rgb'].shape}")     # Should be (7, 3, 88, 88)
    print(f"Obs Vector Shape: {obs['vector'].shape}") # Should be (7, 2)
    
    # Step with array actions
    actions = np.zeros(env.n_agents, dtype=int)
    obs, rewards, dones, info = env.step(actions)
    
    print(f"Rewards Shape: {rewards.shape}") # Should be (7,)
    print(f"Dones Shape: {dones.shape}")     # Should be (7,)
    print(f"Info Keys: {info.keys()}")
    
    if SHOW_RENDER_WITH_PYGAME:
        run_pygame_test(env, "Clean Up", scale_factor=4)
        
    env.close()

    # --- Test 2: Scenario ---
    print("\n" + "="*40 + "\nTesting Scenario (cooking)\n" + "="*40)
    env = build_meltingpot_env("collaborative_cooking__asymmetric_0")
    obs, info = env.reset()
    
    print(f"Obs Keys: {obs.keys()}")
    print(f"Obs RGB Shape: {obs['rgb'].shape}")       # (1, 3, 88, 88)
    print(f"Obs Vector Shape: {obs['vector'].shape}") # (1, 0) -> Cooking has no vector obs usually? Or maybe some.
    
    actions = np.zeros(env.n_agents, dtype=int)
    obs, rewards, dones, info = env.step(actions)
    
    print(f"Rewards Shape: {rewards.shape}")
    print(f"Dones Shape: {dones.shape}")
    
    if SHOW_RENDER_WITH_PYGAME:
        run_pygame_test(env, "Cooking Scenario")
        
    env.close()
    print("\nAll Validations Passed!")