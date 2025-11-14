import logging
import os
# os.environ['SDL_AUDIODRIVER'] = 'dummy'
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import dm_env
from dm_env import specs
from typing import Dict, List, Any, Tuple
import pygame
import time

from meltingpot import substrate
from meltingpot import scenario

from meltingpot.configs import substrates as substrate_configs
from meltingpot.configs import scenarios as scenario_configs

VECTOR_OBS_EXCLUDE_KEYS = frozenset(['RGB', 'WORLD.RGB', 'COLLECTIVE_REWARD'])

def _extract_vector_obs(obs_dict_single_agent: Dict[str, Any], 
                        vector_keys: List[str],
                        discrete_map: Dict[str, int]) -> np.ndarray:
    """ 
    Extract and concatenate vector observations from the observation dictionary.
    - One-hot encodes keys found in discrete_map.
    - Flattens and concatenates other keys.
    """
    vector_parts = []
    for key in vector_keys: # vector_keys is sorted
        try:
            value = obs_dict_single_agent[key]
            
            if key in discrete_map:
                # Key is discrete, one-hot encode it
                num_classes = discrete_map[key]
                vector_parts.append(_one_hot_encode(value, num_classes))
            else:
                # Key is continuous (or just not a discrete scalar), flatten it
                vector_parts.append(np.asarray(value).flatten())

        except KeyError:
            print(f"[Wrapper] Warning: Key '{key}' missing in observation dictionary.- Skipped.")
            pass 
        except ValueError as e:
            print(f"[Wrapper] Warning: Error processing key '{key}' (value: {value}). Skipped. Error: {e}")
            pass
            
    if not vector_parts:
        return np.array([], dtype=np.float32)
        
    return np.concatenate(vector_parts, dtype=np.float32)

def _one_hot_encode(value: int, num_classes: int) -> np.ndarray:
    """Helper to create a one-hot vector."""
    if not 0 <= value < num_classes:
        raise ValueError(f"Value {value} is out of bounds for num_classes {num_classes}")
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[int(value)] = 1.0
    return one_hot

class _SubstrateWrapper:
    """
    Wraps a Melting Pot Substrate environment.

        - Provides access to *all* agents
        - Requires an action dictionary containing actions for *all* agents
    """
    
    def __init__(self, substrate_name: str):
        print(f"[Wrapper] Initializing Substrate: {substrate_name}...")

        # 1. Load substrate config
        self.substrate_name = substrate_name                      # substrate name
        self._config = substrate.get_config(self.substrate_name)  # substrate config

        # 2. Get player roles and agent IDs
        self._roles = self._config.default_player_roles                # player roles
        self.n_agents = len(self._roles)                               # number of agents
        self.agent_ids = [f"player_{i}" for i in range(self.n_agents)] # agent IDs

        # 3. Get action space
        self._action_set = self._config.action_set                # action set
        self.n_actions = len(self._action_set)                    # number of actions (discrete)

        # 4. Build the underlying Melting Pot environment
        self._env = substrate.build_from_config(self._config, roles=self._roles)

        # 5. Get episode length limit
        lab2d_settings = self._config.lab2d_settings_builder(     # get lab2d settings
            roles=self._roles, config=self._config
        )
        self.episode_limit = lab2d_settings.get("maxEpisodeLengthFrames", 1000) # episode limit
        self._step_count = 0

        # 6. Get observation space (RGB, Vector, and Global) from spec
        agent_spec = self._config.timestep_spec.observation
        
        # --- Vector observation spec (processing) ---
        self._discrete_scalar_map = {} # Stores {key: num_classes} for one-hot
        self.obs_vector_dim = 0
        self.vector_keys = [] # Will store all keys for the vector obs
        
        for key, spec in agent_spec.items():
            if key in VECTOR_OBS_EXCLUDE_KEYS:
                continue
            
            # Check if it's a discrete scalar (e.g., DiscreteArray(shape=(), num_values=2))
            if isinstance(spec, specs.DiscreteArray) and spec.shape == ():
                self._discrete_scalar_map[key] = spec.num_values
                self.obs_vector_dim += spec.num_values
                self.vector_keys.append(key)
            
            # Check if it's any other array (continuous vector, bounded array, etc.)
            elif hasattr(spec, 'shape') and hasattr(spec, 'dtype'): 
                # This is a general check for Array-like specs (e.g., shape=(3,))
                self.obs_vector_dim += np.prod(spec.shape) # spec.size is total elements
                self.vector_keys.append(key)
            
            # else: key is some complex/unsupported spec, skip it
        
        self.vector_keys.sort()
        print(f"[Wrapper] Detected vector observation (Vector Obs) dimension: {self.obs_vector_dim}")
        print(f"[Wrapper] Contained discrete (one-hot) keys: {self._discrete_scalar_map}")
        print(f"[Wrapper] All vector keys (sorted): {self.vector_keys}")

        # --- Handle missing RGB ---
        if "RGB" in agent_spec:
            self._obs_spec = agent_spec["RGB"]
            self.obs_shape = (
                self._obs_spec.shape[2], 
                self._obs_spec.shape[0], 
                self._obs_spec.shape[1]
            )
        else:
            print("[Wrapper] Warning: 'RGB' not found in substrate observation spec. 'obs_shape' and 'state_shape' will be None/0.")
            self._obs_spec = None
            self.obs_shape = None
        
        self.state_shape = {
            "rgb": (self.n_agents,) + self.obs_shape if self.obs_shape else None,
            "vector": (self.n_agents, self.obs_vector_dim)
        }
        
        # --- Handle missing WORLD.RGB ---
        if "WORLD.RGB" in agent_spec:
            self._global_spec = agent_spec["WORLD.RGB"]
            self.global_shape = self._global_spec.shape  # (H, W, C)
        else:
            print("[Wrapper] Warning: 'WORLD.RGB' not found in substrate observation spec. 'global_shape' will be None/0.")
            self._global_spec = None
            self.global_shape = None

        # 7. Reset environment properly for the first step
        self._last_timestep = self._env.reset()

        print(f"[Wrapper] Substrate initialization complete. N_Agents = {self.n_agents}")

    def get_env_info(self) -> dict:
        info = {
            "n_agents": self.n_agents,              # number of agents
            "n_actions": self.n_actions,            # number of actions
            "agent_ids": self.agent_ids,            # agent IDs
            "obs_shape": self.obs_shape,            # observation shape (C, H, W) or None
            "obs_vector_dim": self.obs_vector_dim,  # vector observation dimension
            "state_shape": self.state_shape,        # state shape (N, C, H, W) or 0
            "global_shape": self.global_shape,      # render shape (H, W, C)
            "episode_limit": self.episode_limit,    # episode limit
        }
        return info

    # --- General purpose observation preprocessing ---
    def _preprocess_obs(self, obs_dict: dict) -> dict:
        """ Preprocess observation dictionary:
            - Convert 'RGB' to (C, H, W) and normalize
            - Remove 'WORLD.RGB' and 'COLLECTIVE_REWARD'
            - Pass through all other observation keys unmodified
        """
        processed_obs = {}
        for key, value in obs_dict.items():
            if key == 'RGB':
                obs_rgb = value.astype(np.float32)
                obs_rgb /= 255.0
                processed_obs[key] = np.transpose(obs_rgb, (2, 0, 1)) # (C, H, W)
            
            elif key in ['COLLECTIVE_REWARD', 'WORLD.RGB']:
                # Exclude these keys from the agent's observation dict
                pass

            else:
                # Pass through other keys like "READY_TO_SHOOT", "INVENTORY",
                # "NUM_OTHERS_WHO_CLEANED_THIS_STEP", etc.
                processed_obs[key] = value
        
        return processed_obs

    def get_obs(self) -> Dict[str, np.ndarray]:
        """ Get observations for all agents as a dictionary. """
        if self._last_timestep is None:
            raise RuntimeError("Must call reset() before calling get_obs()")
            
        agent_observations_list = self._last_timestep.observation  # list of observations for all agents
        
        obs_dict = {}                                              # dictionary to hold processed observations
        for i, agent_id in enumerate(self.agent_ids):              # iterate over agents
            obs_dict[agent_id] = self._preprocess_obs(
                agent_observations_list[i]
            )
        return obs_dict                                            # return the observation dictionary

    def get_state(self) -> Dict[str, np.ndarray]:
        """ Get global state (stacked 'RGB' observations). """
        obs_dict = self.get_obs()  # get observations dictionary
        
        rgb_list = []
        vector_list = []

        for agent_id in self.agent_ids:
            agent_obs = obs_dict[agent_id]

            if 'RGB' in agent_obs:
                rgb_list.append(agent_obs['RGB'])
            
            # --- MODIFIED CALL ---
            vector_obs = _extract_vector_obs(
                agent_obs, self.vector_keys, self._discrete_scalar_map
            )
            vector_list.append(vector_obs)

        stacked_rgb = np.stack(rgb_list, axis=0) if rgb_list else np.array([], dtype=np.float32)
        stacked_vector = np.stack(vector_list, axis=0) if vector_list else np.array([], dtype=np.float32)

        return {"rgb": stacked_rgb, "vector": stacked_vector}
        
    def get_global(self) -> Dict[str, np.ndarray]:
        """ Get global observation ('WORLD.RGB'). """
        if self._last_timestep is None:
            raise RuntimeError("Must call reset() before calling render()")
        
        agent_observations_list = self._last_timestep.observation
        obs_dict_agent_0 = agent_observations_list[0]

        if 'WORLD.RGB' in obs_dict_agent_0:
            world_rgb = obs_dict_agent_0['WORLD.RGB']
        else:
            world_rgb = np.zeros(self.global_shape, dtype=np.uint8)   # Fallback to zeros if missing                             

        state_dict = self.get_state()
        all_vectors = state_dict["vector"]

        return {"world_rgb": world_rgb, "all_vectors": all_vectors}

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """ Reset the environment and return initial observations and state. """
        self._step_count = 0
        self._last_timestep = self._env.reset()

        return self.get_obs(), {
            "concat_state": self.get_state(),
            "global_state": self.get_global()
        }

    def step(self, actions_dict: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """ Take a step in the environment using the provided actions dictionary. """
        actions_list = [actions_dict[agent_id] for agent_id in self.agent_ids] # convert dict to list
        self._last_timestep = self._env.step(actions_list)                     # take a step
        self._step_count += 1                                                  # increment step count
        
        rewards_dict = {}                                    # dictionary for rewards
        dones_dict = {}                                      # dictionary for done flags

        agent_rewards_list = self._last_timestep.reward      # list of rewards for all agents
        
        global_done = False
        if self._step_count >= self.episode_limit:           # check episode limit
            global_done = True
        elif self._last_timestep.last():                     # check if episode ended
            global_done = True

        for i, agent_id in enumerate(self.agent_ids):
            rewards_dict[agent_id] = agent_rewards_list[i]   # assign rewards
            dones_dict[agent_id] = global_done               # assign done flags
            
        next_obs_dict = self.get_obs()                       # get next observations
        next_state = self.get_state()                        # get next state
        global_state = self.get_global()                     # get global observation
        info_dict = {}                                       # empty info dictionary

        if self._last_timestep.observation:                                          # if observations exist
            raw_obs_agent_0 = self._last_timestep.observation[0]                     # raw obs of agent 0
            if 'COLLECTIVE_REWARD' in raw_obs_agent_0:
                info_dict['collective_reward'] = raw_obs_agent_0['COLLECTIVE_REWARD'] # add collective reward to info
        
        info_dict['global_state'] = global_state
        
        return next_obs_dict, next_state, rewards_dict, dones_dict, info_dict

    def render(self) -> np.ndarray:
        """ Render the current environment state as an RGB array. """
        global_state_dict = self.get_global()
        return global_state_dict["world_rgb"]  # Ensure last timestep is available
    
    def close(self):
        self._env.close()

class _ScenarioWrapper:
    """
    Encapsulates a Melting Pot Scenario environment.

    - Only exposes *focal* agents (the learning agents)
    - Requires an action dictionary containing only *focal* agent actions  
    - Handles bot actions automatically internally
    - Maintains API compatibility with _SubstrateWrapper
    """
    
    def __init__(self, scenario_name: str):
        print(f"[Wrapper] Initializing Scenario: {scenario_name}...")
        self.scenario_name = scenario_name

        # 1. Load scenario config (needed to get is_focal etc.)
        self._config = scenario.get_config(self.scenario_name)

        # 2. Build the underlying Melting Pot *scenario* environment
        self._env = scenario.build(self.scenario_name)

        # 3. Identify *focal* agents
        self._roles = self._config.roles                   # player roles
        self._is_focal = self._config.is_focal             # focal agent flags
        
        self.agent_ids = []                                # list of focal agent IDs
        self._focal_agent_indices = []                     # indices of focal agents in the full env
        for i, is_focal in enumerate(self._is_focal):      # iterate over agents
            if is_focal:
                self.agent_ids.append(f"player_{i}")       # add focal agent ID
                self._focal_agent_indices.append(i)        # add focal agent index
                
        self.n_agents = len(self.agent_ids)                # number of focal agents

        # 4. Get action and observation spaces (only for focal agents)
        self._action_spec_list = self._env.action_spec()   # list of action specs
        self._obs_spec_list = self._env.observation_spec() # list of observation specs
        
        self.n_actions = self._action_spec_list[0].num_values # assume all focal agents have same action space

        # 5. Get episode length limit
        self._substrate_config = substrate.get_config(self._config.substrate) # substrate config
        self._substrate_roles = self._substrate_config.default_player_roles   # substrate roles
        
        lab2d_settings = self._substrate_config.lab2d_settings_builder(       # get lab2d settings
            roles=self._substrate_roles, config=self._substrate_config
        )
        self.episode_limit = lab2d_settings.get("maxEpisodeLengthFrames", 1000)
        self._step_count = 0

        agent_spec = self._obs_spec_list[0] # Spec for focal agents

        # --- Vector observation spec (processing) ---
        self._discrete_scalar_map = {} # Stores {key: num_classes} for one-hot
        self.obs_vector_dim = 0
        self.vector_keys = [] # Will store all keys for the vector obs
        
        for key, spec in agent_spec.items():
            if key in VECTOR_OBS_EXCLUDE_KEYS:
                continue
            
            # Check if it's a discrete scalar (e.g., DiscreteArray(shape=(), num_values=2))
            if isinstance(spec, specs.DiscreteArray) and spec.shape == ():
                self._discrete_scalar_map[key] = spec.num_values
                self.obs_vector_dim += spec.num_values
                self.vector_keys.append(key)
            
            # Check if it's any other array (continuous vector, bounded array, etc.)
            elif hasattr(spec, 'shape') and hasattr(spec, 'dtype'): 
                # This is a general check for Array-like specs (e.g., shape=(3,))
                self.obs_vector_dim += np.prod(spec.shape) # spec.size is total elements
                self.vector_keys.append(key)
            
            # else: key is some complex/unsupported spec, skip it
        
        self.vector_keys.sort()
        print(f"[Wrapper] Detected vector observation (Vector Obs) dimension: {self.obs_vector_dim}")
        print(f"[Wrapper] Contained discrete (one-hot) keys: {self._discrete_scalar_map}")
        print(f"[Wrapper] All vector keys (sorted): {self.vector_keys}")

        # --- Handle missing RGB ---
        if "RGB" in agent_spec:
            obs_spec_rgb = agent_spec["RGB"]
            self.obs_shape = (
                obs_spec_rgb.shape[2], 
                obs_spec_rgb.shape[0], 
                obs_spec_rgb.shape[1]
            )
        else:
            print("[Wrapper] Warning: 'RGB' not found in scenario observation spec. 'obs_shape' will be None.")
            self.obs_shape = None
        
        self.state_shape = {
            "rgb": (self.n_agents,) + self.obs_shape if self.obs_shape else None,
            "vector": (self.n_agents, self.obs_vector_dim)
        }

        # --- Global shape (remains the same, from substrate config) ---
        if "WORLD.RGB" in self._substrate_config.timestep_spec.observation:
            self._global_spec = self._substrate_config.timestep_spec.observation["WORLD.RGB"]
            self.global_shape = self._global_spec.shape    # (H, W, C)
        else:
            print("[Wrapper] Warning: 'WORLD.RGB' not found in substrate observation spec. 'global_shape' will be None/0.")
            self._global_spec = None
            self.global_shape = None

        self._last_timestep = self._env.reset()

        print(f"[Wrapper] Scenario initialization finished. N_Agents = {self.n_agents} (focal agents only)")

    def get_env_info(self) -> dict:
        info = {
            "n_agents": self.n_agents,              # number of agents
            "n_actions": self.n_actions,            # number of actions
            "agent_ids": self.agent_ids,            # agent IDs
            "obs_shape": self.obs_shape,            # observation shape (C, H, W) or None
            "obs_vector_dim": self.obs_vector_dim,  # vector observation dimension
            "state_shape": self.state_shape,        # state shape (Dict) or 0
            "global_shape": self.global_shape,      # render shape (H, W, C)
            "episode_limit": self.episode_limit     # episode limit
        }
        return info

    def _preprocess_obs(self, obs_dict: dict) -> dict:
        """
        Preprocessing the observation dictionary:
        - Transform 'RGB' to (C, H, W) format and normalize
        - Remove 'WORLD.RGB' and 'COLLECTIVE_REWARD'
        - Retain all other keys unchanged
        """
        processed_obs = {}

        for key, value in obs_dict.items():
            if key == 'RGB':
                obs_rgb = value.astype(np.float32)
                obs_rgb /= 255.0
                processed_obs[key] = np.transpose(obs_rgb, (2, 0, 1)) # (C, H, W)
            
            elif key in ['COLLECTIVE_REWARD', 'WORLD.RGB']:
                # Exclude these keys from the agent's observation dict
                pass

            else:
                # Pass through other keys
                processed_obs[key] = value
                
        return processed_obs

    def get_obs(self) -> Dict[str, np.ndarray]:
        if self._last_timestep is None:
            raise RuntimeError("Must call reset() before calling get_obs()")
            
        focal_obs_list = self._last_timestep.observation  # list of observations for all agents
        
        obs_dict = {}                                     # dictionary to hold processed observations
        for i, agent_id in enumerate(self.agent_ids):     # iterate over focal agents
            obs_dict[agent_id] = self._preprocess_obs(    # preprocess observation
                focal_obs_list[i]
            )
        return obs_dict

    def get_state(self) -> Dict[str, np.ndarray]:
        """ Get concat state (concatenation of all 'RGB' observations of focal agents). """
        obs_dict = self.get_obs()    # get observations dictionary
        
        rgb_list = []
        vector_list = []

        for agent_id in self.agent_ids:
            agent_obs = obs_dict[agent_id]

            if 'RGB' in agent_obs:
                rgb_list.append(agent_obs['RGB'])
            
            # --- MODIFIED CALL ---
            vector_obs = _extract_vector_obs(
                agent_obs, self.vector_keys, self._discrete_scalar_map
            )
            vector_list.append(vector_obs)
        
        stacked_rgb = np.stack(rgb_list, axis=0) if rgb_list else np.array([], dtype=np.float32)
        stacked_vector = np.stack(vector_list, axis=0) if vector_list else np.array([], dtype=np.float32)
        
        return {
            "rgb": stacked_rgb,
            "vector": stacked_vector
        }
    
    def get_global(self) -> np.ndarray:
        """ Get global observation ('WORLD.RGB'). """
        if self._last_timestep is None:
            raise RuntimeError("Must call reset() before calling render()")
        
        substrate_obs_list = self._env._substrate.observation()  # full substrate observations
        if not substrate_obs_list:
            raise ValueError("self._env._substrate.observation() returned an empty list.")
        
        obs_dict_agent_0 = substrate_obs_list[0]  # observation of agent 0
        if 'WORLD.RGB' in obs_dict_agent_0:
            world_rgb = obs_dict_agent_0['WORLD.RGB']
        else:
            world_rgb = np.zeros(self.global_shape, dtype=np.uint8) # Fallback
        
        state_dict = self.get_state()
        all_vectors = state_dict["vector"] # This is (N_focal, V)

        return {"world_rgb": world_rgb, "all_vectors": all_vectors}

    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """ Reset the environment and return initial observations and state. """
        self._step_count = 0
        self._last_timestep = self._env.reset()
        
        return self.get_obs(), {
                "concat_state": self.get_state(),
                "global_state": self.get_global()
            }

    def step(self, actions_dict: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """ Take a step in the environment using the provided actions dictionary for focal agents. """
        actions_list = [actions_dict[agent_id] for agent_id in self.agent_ids] # convert dict to list for focal agents
        self._last_timestep = self._env.step(actions_list)                     # take a step
        self._step_count += 1
        
        rewards_dict = {}       # dictionary for rewards
        dones_dict = {}         # dictionary for done flags

        focal_rewards_list = self._last_timestep.reward  # list of rewards for focal agents
        
        global_done = False
        if self._step_count >= self.episode_limit:       # check episode limit
            global_done = True
        elif self._last_timestep.last():                 # check if episode ended
            global_done = True

        for i, agent_id in enumerate(self.agent_ids):    # iterate over focal agents
            rewards_dict[agent_id] = focal_rewards_list[i]
            dones_dict[agent_id] = global_done
            
        next_obs_dict = self.get_obs()                   # get next observations
        next_state = self.get_state()                    # get next state
        info_dict = {}                                   # empty info dictionary

        if self._last_timestep.observation:
            raw_obs_agent_0 = self._last_timestep.observation[0]  # raw obs of focal agent 0
            if 'COLLECTIVE_REWARD' in raw_obs_agent_0:
                info_dict['collective_reward'] = raw_obs_agent_0['COLLECTIVE_REWARD']
        
        info_dict['global_state'] = self.get_global() # Add global_state to info
        
        return next_obs_dict, next_state, rewards_dict, dones_dict, info_dict
    
    def render(self) -> np.ndarray:
        """ Render the current environment state as an RGB array. """
        global_state_dict = self.get_global()
        return global_state_dict["world_rgb"]  # Ensure last timestep is available

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
    H, W, C = render_shape

    # Pygame uses (Width, Height)
    screen_size = (W * scale_factor, H * scale_factor)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(f"Test: {test_name}")
    clock = pygame.time.Clock()

    # Reset the environment
    obs_dict, state = env.reset()
    
    running = True
    for i in range(steps):
        if not running:
            break

        # 1. Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        # 2. Get the render frame
        frame = env.render() # (H, W, C)

        # 3. Prepare Pygame Surface
        # Pygame surface expects (W, H, C), so we swap H and W axes
        frame_transposed = frame.swapaxes(0, 1) # (W, H, C)
        surface = pygame.surfarray.make_surface(frame_transposed)

        # 4. Scale and draw to the screen
        scaled_surface = pygame.transform.scale(surface, screen_size)
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        # 5. Generate random actions
        # Pygame test does not need to access obs or rewards, we only test rendering
        actions_dict = {
            agent_id: np.random.randint(0, env.n_actions) 
            for agent_id in env.agent_ids
        }

        # 6. Execute one step
        next_obs_dict, next_state, rewards_dict, dones_dict, info = env.step(actions_dict)

        # 7. Print collective reward (if available)
        if 'collective_reward' in info:
            coll_reward = info['collective_reward']
            print(f"  [Step {i+1}/{steps}] {test_name} - Collective Reward: {coll_reward}", end="\r")

        # 8. Control frame rate
        clock.tick(fps)
        
        if dones_dict[env.agent_ids[0]]:
            print(f"\n  [Step {i+1}] Environment terminated, resetting...")
            obs_dict, state = env.reset()

    pygame.quit()
    print(f"\n--- Pygame test: {test_name} completed ---")

if __name__ == "__main__":

    # --- Global Settings ---

    # !!! Set to True to pop up the Pygame window for testing !!!
    SHOW_RENDER_WITH_PYGAME = False

    # Shapes from config files
    CORRECT_RENDER_SHAPE_CLEANUP = (168, 240, 3) # From clean_up.py (H, W, C)
    CORRECT_RENDER_SHAPE_COOKING = (40, 72, 3)  # From collaborative_cooking.py

    PYGAME_SCALE_CLEANUP = 4  # Scale 168x240 to 672x960
    PYGAME_SCALE_COOKING = 10 # Scale 40x72 to 400x720
    PYGAME_FPS = 10           # Simulate running speed
    PYGAME_STEPS = 200        # Simulate running steps

    # --- Test 1: Load Substrate (clean_up) ---
    print("="*50)
    print("Test 1: Load Substrate (clean_up)")
    print("="*50)
    
    env_substrate = build_meltingpot_env("clean_up")
    
    info_sub = env_substrate.get_env_info()
    print("Substrate environment info (get_env_info()):")
    print(f"  - n_agents: {info_sub['n_agents']}")
    print(f"  - agent_ids: {info_sub['agent_ids']}")
    print(f"  - global_shape: {info_sub['global_shape']}")
    print(f"  - obs_shape (C,H,W): {info_sub['obs_shape']}")
    print(f"  - obs_vector_dim: {info_sub['obs_vector_dim']}")
    print(f"  - state_shape (Critic): {info_sub['state_shape']}")


    # Validate metadata
    assert info_sub["n_agents"] == 7
    assert info_sub["agent_ids"] == [f"player_{i}" for i in range(7)]
    assert info_sub["global_shape"] == CORRECT_RENDER_SHAPE_CLEANUP 

    # Run Pygame test (if enabled)
    if SHOW_RENDER_WITH_PYGAME:
        run_pygame_test(
            env_substrate, 
            test_name="Substrate (All Agents - clean_up)", 
            scale_factor=PYGAME_SCALE_CLEANUP, 
            fps=PYGAME_FPS,
            steps=PYGAME_STEPS
        )
    
    # Run headless test (check API)
    obs_dict, state = env_substrate.reset()
    
    # Test that obs_dict contains clean_up keys
    print(f"\nSubstrate obs keys (player_0): {obs_dict['player_0'].keys()}")
    assert 'RGB' in obs_dict['player_0']
    assert 'READY_TO_SHOOT' in obs_dict['player_0']
    assert 'NUM_OTHERS_WHO_CLEANED_THIS_STEP' in obs_dict['player_0']
    
    frame = env_substrate.render()
    assert frame.shape == CORRECT_RENDER_SHAPE_CLEANUP
    
    # Create actions for all 7 agents
    actions_dict_sub = {
        agent_id: np.random.randint(0, info_sub["n_actions"])
        for agent_id in info_sub["agent_ids"]
    }
    
    next_obs_dict, next_state, rewards_sub, dones_sub, info_sub_step = env_substrate.step(actions_dict_sub)
    
    # Assert that clean_up keys are in the *next* obs dict
    assert "RGB" in next_obs_dict["player_0"]
    assert "READY_TO_SHOOT" in next_obs_dict["player_0"]
    assert "NUM_OTHERS_WHO_CLEANED_THIS_STEP" in next_obs_dict["player_0"]
    
    env_substrate.close()
    print("\n✅ Substrate validation successful!")


    # --- Test 2: Load Scenario (Control Focal Agent Only) ---
    print("\n" + "="*50)
    print("Test 2: Load Scenario (collaborative_cooking__asymmetric_0)")
    print("="*50)

    env_scenario = build_meltingpot_env("collaborative_cooking__asymmetric_0")

    info_scn = env_scenario.get_env_info()
    print("Scenario environment info (get_env_info()):")
    print(f"  - n_agents: {info_scn['n_agents']}")
    print(f"  - agent_ids: {info_scn['agent_ids']}")
    print(f"  - global_shape: {info_scn['global_shape']}")
    print(f"  - obs_shape (C,H,W): {info_scn['obs_shape']}")
    print(f"  - obs_vector_dim: {info_scn['obs_vector_dim']}")
    print(f"  - state_shape (Critic): {info_scn['state_shape']}")

    # Validate metadata
    assert info_scn["n_agents"] == 1
    assert info_scn["agent_ids"] == ["player_0"]
    assert info_scn["global_shape"] == CORRECT_RENDER_SHAPE_COOKING

    # Run Pygame test (if enabled)
    if SHOW_RENDER_WITH_PYGAME:
        run_pygame_test(
            env_scenario, 
            test_name="Scenario (Focal Agents + Bots)", 
            scale_factor=PYGAME_SCALE_COOKING, 
            fps=PYGAME_FPS,
            steps=PYGAME_STEPS
        )

    # Run headless test (check API)
    obs_dict, state = env_scenario.reset()
    
    # Test that obs_dict contains more than just RGB
    assert 'RGB' in obs_dict['player_0']
    print(f"\nScenario obs keys (player_0): {obs_dict['player_0'].keys()}")
    
    frame_scn = env_scenario.render()
    assert frame_scn.shape == CORRECT_RENDER_SHAPE_COOKING
    actions_dict_scn = {
        "player_0": np.random.randint(0, info_scn["n_actions"])
    }
    next_obs_dict_scn, next_state, rewards_scn, dones_scn, info_scn_step = env_scenario.step(actions_dict_scn)
    assert "player_0" in rewards_scn
    assert "player_1" not in rewards_scn
    assert "collective_reward" in info_scn_step

    env_scenario.close()
    print("\n✅ Scenario validation successful!")


    # --- Test 3: API Consistency Check (Substrate vs Scenario) ---
    print("\n" + "="*50)
    print("Test 3: Verify API Consistency (Substrate vs Scenario)")
    print("="*50)

    try:
        # --- 1. Substrate (clean_up) ---
        print("Checking Substrate (clean_up)...")
        env_sub = build_meltingpot_env("clean_up")
        info_sub = env_sub.get_env_info()
        obs_sub, state_sub = env_sub.reset()
        state_sub = state_sub["concat_state"]
        
        # 1a. Check reset() state (should be concat_state)
        print(f"[Substrate] reset() state keys: {state_sub.keys()}")
        assert "rgb" in state_sub and "vector" in state_sub
        print(f"[Substrate] state['vector'].shape: {state_sub['vector'].shape} (Expected: {(info_sub['n_agents'], info_sub['obs_vector_dim'])})")
        assert state_sub['vector'].shape == (info_sub['n_agents'], info_sub['obs_vector_dim'])

        actions_sub = {aid: np.random.randint(0, info_sub["n_actions"]) for aid in info_sub["agent_ids"]}
        next_obs_sub, next_state_sub, _, _, info_sub_step = env_sub.step(actions_sub)

        # 1b. Check step() next_state (should be concat_state)
        print(f"[Substrate] step() next_state keys: {next_state_sub.keys()}")
        assert "rgb" in next_state_sub and "vector" in next_state_sub
        
        # 1c. Check info_dict (should contain global_state)
        print(f"[Substrate] info_dict keys: {info_sub_step.keys()}")
        assert "global_state" in info_sub_step
        assert "world_rgb" in info_sub_step["global_state"]
        assert "all_vectors" in info_sub_step["global_state"]
        
        print("✅ Substrate API check passed.")
        env_sub.close()

        # --- 2. Scenario (collaborative_cooking__asymmetric_0) ---
        print("\nChecking Scenario (collaborative_cooking__asymmetric_0)...")
        env_scn = build_meltingpot_env("collaborative_cooking__asymmetric_0")
        info_scn = env_scn.get_env_info()
        obs_scn, state_scn = env_scn.reset()
        state_scn = state_scn["concat_state"]

        # 2a. Check reset() state (should be concat_state)
        print(f"[Scenario] reset() state keys: {state_scn.keys()}")
        assert "rgb" in state_scn and "vector" in state_scn
        print(f"[Scenario] state['vector'].shape: {state_scn['vector'].shape} (Expected: {(info_scn['n_agents'], info_scn['obs_vector_dim'])})")
        assert state_scn['vector'].shape == (info_scn['n_agents'], info_scn['obs_vector_dim'])

        actions_scn = {aid: np.random.randint(0, info_scn["n_actions"]) for aid in info_scn["agent_ids"]}
        next_obs_scn, next_state_scn, _, _, info_scn_step = env_scn.step(actions_scn)

        # 2b. Check step() next_state (should be concat_state)
        print(f"[Scenario] step() next_state keys: {next_state_scn.keys()}")
        assert "rgb" in next_state_scn and "vector" in next_state_scn

        # 2c. Check info_dict (should contain global_state)
        print(f"[Scenario] info_dict keys: {info_scn_step.keys()}")
        assert "global_state" in info_scn_step
        assert "world_rgb" in info_scn_step["global_state"]
        assert "all_vectors" in info_scn_step["global_state"]

        print("✅ Scenario API check passed.")
        env_scn.close()

        print("\n✅ API Consistency Validation Successful!")

    except Exception as e:
        print(f"\n❌ API Consistency Validation FAILED: {e}")
        import traceback
        traceback.print_exc()

    
    print("\n" + "="*50)
    print("🎉 Factory function (build_meltingpot_env) compatibility validation passed! 🎉")
    print("="*50)