import multiprocessing as mp
import numpy as np
from typing import List, Any, Callable, Union
from MeltingPotWrapper import build_meltingpot_env

def recursive_stack(list_of_structs: List[Any]) -> Any:
    """
    Recursively stack data structures.
    
    Example input: 
        [ {'rgb': (7,3,88,88)}, {'rgb': (7,3,88,88)} ] (Batch=2)
    Example output: 
        {'rgb': (2, 7, 3, 88, 88)}
    """
    if not list_of_structs:
        return []
        
    first_elem = list_of_structs[0]
    
    if isinstance(first_elem, dict):
        return {
            key: recursive_stack([d[key] for d in list_of_structs])
            for key in first_elem
        }
    elif isinstance(first_elem, (list, tuple)):
        return type(first_elem)(recursive_stack([s[i] for s in list_of_structs]) for i in range(len(first_elem)))
    elif isinstance(first_elem, np.ndarray):
        return np.stack(list_of_structs, axis=0)
    elif isinstance(first_elem, (int, float, bool, np.number)):
        return np.array(list_of_structs)
    else:
        # For other types, return the list directly
        return list_of_structs

def _worker(parent_conn, child_conn, env_fn):
    """
    Worker process: handles the new dictionary-free interface (returning 4-tuple)
    """
    parent_conn.close()
    
    try:
        env = env_fn()
        while True:
            cmd, data = child_conn.recv()
            
            if cmd == 'step':
                actions = data    # data is action array: (N_Agents,)
                
                obs, rewards, dones, info = env.step(actions) # dones is now a bool array (N_Agents,)

                if np.any(dones):
                    info["final_observation"] = obs

                    reset_obs, reset_info = env.reset()
                    obs = reset_obs
                    
                    info.update(reset_info)
                
                child_conn.send((obs, rewards, dones, info))
                
            elif cmd == 'reset':
                obs, info = env.reset()
                child_conn.send((obs, info))
                
            elif cmd == 'close':
                env.close()
                child_conn.close()
                break
                
            elif cmd == 'get_env_info':
                if hasattr(env, 'get_env_info'):
                    child_conn.send(env.get_env_info())
                else:
                    child_conn.send(None)
            else:
                child_conn.send(('ERROR', f"Unknown command: {cmd}"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            child_conn.send(('ERROR', str(e)))
        except:
            pass
    finally:
        if 'env' in locals():
            env.close()

class MeltingPotAsyncVectorEnv:
    """
    Melting Pot Asynchronous Vector Environment (optimized for dictionary-free interface)
    """
    def __init__(self, env_fns: List[Callable]):
        self.num_envs = len(env_fns)
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        self.processes = []
        for parent_conn, child_conn, env_fn in zip(self.parent_conns, self.child_conns, env_fns):
            p = mp.Process(target=_worker, args=(parent_conn, child_conn, env_fn))
            p.daemon = True
            p.start()
            child_conn.close()
            self.processes.append(p)
            
    def reset(self):
        """
        Returns:
            stacked_obs: Dict {'rgb': (B, N, C, H, W), 'vector': (B, N, V)}
            infos: List[Dict]
        """
        for conn in self.parent_conns:
            conn.send(('reset', None))
            
        results = [conn.recv() for conn in self.parent_conns]
        self._check_errors(results)
        
        # results: [(obs0, info0), (obs1, info1), ...]
        obs_list, info_list = zip(*results)
        
        stacked_obs = recursive_stack(obs_list)
        return stacked_obs, list(info_list)

    def step(self, actions: Union[np.ndarray, List]):
        """
        Args:
            actions: shape (Num_Envs, N_Agents)
                     or list of arrays
        Returns:
            obs: Stacked Dict {'rgb': (B, N, ...), 'vector': (B, N, ...)}
            rewards: (B, N)
            dones: (B, N)
            infos: List[Dict] (length B)
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Actions length ({len(actions)}) != Num Envs ({self.num_envs})")

        for conn, act in zip(self.parent_conns, actions):
            conn.send(('step', act))

        results = [conn.recv() for conn in self.parent_conns]
        self._check_errors(results)
        
        # Unpack: results[i] = (obs, rewards, dones, info)
        obs_list, rewards_list, dones_list, infos_list = zip(*results)

        # Obs (Dict of Arrays -> Stacked Dict of Arrays)
        stacked_obs = recursive_stack(obs_list)
        
        # Rewards (Array (N,) -> Stacked Array (B, N))
        stacked_rewards = np.stack(rewards_list, axis=0)
        
        # Dones (Array (N,) -> Stacked Array (B, N))
        stacked_dones = np.stack(dones_list, axis=0)
        
        # Infos (List of Dicts)
        infos = list(infos_list)
        
        return stacked_obs, stacked_rewards, stacked_dones, infos

    def get_env_info(self):
        self.parent_conns[0].send(('get_env_info', None))
        result = self.parent_conns[0].recv()
        if isinstance(result, tuple) and result[0] == 'ERROR':
            raise RuntimeError(result[1])
        return result

    def _check_errors(self, results):
        for item in results:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and item[0] == 'ERROR':
                raise RuntimeError(f"Worker Error: {item[1]}")

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except:
                pass
        for p in self.processes:
            p.join()

if __name__ == "__main__":
    def make_env_fn(env_name):
        def _thunk():
            return build_meltingpot_env(env_name) 
        return _thunk

    try:
        NUM_ENVS = 4
        ENV_NAME = "clean_up"
        
        print(f"Init {NUM_ENVS} envs: {ENV_NAME}...")
        venv = MeltingPotAsyncVectorEnv([make_env_fn(ENV_NAME) for _ in range(NUM_ENVS)])
        
        # 1. Info
        env_info = venv.get_env_info()
        n_agents = env_info['n_agents']
        n_actions = env_info['n_actions']
        print(f"Info: {n_agents} agents, {n_actions} actions")
        
        # 2. Reset
        print("\n--- Testing Reset ---")
        obs, infos = venv.reset()
        print(f"Obs RGB: {obs['rgb'].shape} (Exp: {NUM_ENVS}, {n_agents}, 3, 88, 88)")
        print(f"Obs Vector: {obs['vector'].shape} (Exp: {NUM_ENVS}, {n_agents}, V)")
        
        # 3. Step
        print("\n--- Testing Step ---")
        # Construct actions: numpy array of (Num_Envs, N_Agents)
        actions = np.random.randint(0, n_actions, size=(NUM_ENVS, n_agents))
        
        next_obs, rewards, dones, next_infos = venv.step(actions)
        
        print(f"Rewards shape: {rewards.shape} (Exp: {NUM_ENVS}, {n_agents})")
        print(f"Dones shape: {dones.shape} (Exp: {NUM_ENVS}, {n_agents})")
        print(f"Dones sample: {dones[0]}") # Should be all False or all True (MeltingPot characteristic)
        
        # 4. Verify Auto-Reset (Simulation)
        print("\n--- Simulating Auto-Reset ---")
        # Force running for many steps until completion
        total_steps = 0
        while True:
            actions = np.random.randint(0, n_actions, size=(NUM_ENVS, n_agents))
            _, _, batch_dones, batch_infos = venv.step(actions)
            total_steps += 1
            
            # Check if any environment has reset
            if np.any(batch_dones):
                print(f"Environment finished at step {total_steps}!")
                # Find the index of the finished environment
                env_idx = np.where(batch_dones[:, 0])[0][0] # Just check the done of the first agent
                print(f"Env {env_idx} is done: {batch_dones[env_idx]}")
                
                # Check if info contains final_observation
                info = batch_infos[env_idx]
                if "final_observation" in info:
                    print("✅ 'final_observation' found in info (Auto-Reset successful).")
                    f_obs = info["final_observation"]
                    print(f"Final Obs RGB shape: {f_obs['rgb'].shape}")
                else:
                    print("❌ 'final_observation' MISSING!")
                break
                
            if total_steps > 5005: # Prevent infinite loop
                print("Force break (episode too long)")
                break

        venv.close()
        print("\nDone!")

    except NameError:
        print("❌ Error: Please ensure the Wrapper class definition code (build_meltingpot_env) is run before this snippet.")
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()