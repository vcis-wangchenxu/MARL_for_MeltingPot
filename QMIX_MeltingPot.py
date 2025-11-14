import numpy as np
from typing import List, Tuple, Dict
import swanlab
import random
import time
import copy
import os 

import torch
import torch.optim as optim
import torch.nn as nn

from utils import set_seed, plot_learning_curve
from models import AgentNetwork, MixerNetwork
from memories import OffPolicyReplayBuffer
from MeltingPotWrapper import build_meltingpot_env as GeneralMeltingPotWrapper
from MeltingPotWrapper import _extract_vector_obs 

# -------------------------------------------------------------------
class QMIXAgent:
    """ QMIX Agent Class - Adapted for RGB + Vector """
    def __init__(self, env_info: dict, args: Dict):
        """ Initialize QMIX agent/learner. """
        self.env_info = env_info
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.agent_ids = env_info["agent_ids"]
        
        self.obs_rgb_shape = env_info["obs_shape"]
        self.obs_vector_dim = env_info["obs_vector_dim"]
        self.state_shape_dict = env_info["state_shape"]
        
        self.vector_keys = env_info.get("vector_keys")
        self.discrete_scalar_map = env_info.get("discrete_scalar_map")
        if self.vector_keys is None or self.discrete_scalar_map is None:
            raise ValueError("env_info must contain 'vector_keys' and 'discrete_scalar_map'. \
                             Please modify MeltingPotWrapper.py accordingly.")
        
        self.args = args
        self.lr = args["lr"]
        self.gamma = args["gamma"]
        self.rnn_hidden_dim = args["rnn_hidden_dim"]
        self.mixing_embed_dim = args["mixing_embed_dim"]
        self.grad_norm_clip = args["grad_norm_clip"]
        self.target_update_frequency = args["target_update_frequency"]

        self.device = torch.device(args["device"])

        self.agent_network = AgentNetwork(
            self.obs_rgb_shape, self.obs_vector_dim, self.n_actions, self.rnn_hidden_dim
        ).to(self.device)
        self.target_agent_network = copy.deepcopy(self.agent_network)

        self.mixer_network = MixerNetwork(
            self.n_agents, self.state_shape_dict, self.mixing_embed_dim
        ).to(self.device)
        self.target_mixer_network = copy.deepcopy(self.mixer_network)

        self.params = list(self.agent_network.parameters()) + list(self.mixer_network.parameters())
        self.optimizer = optim.Adam(params=self.params, lr=self.lr)
        self.criterion = nn.MSELoss()

        self._train_step_count = 0 

    @torch.no_grad()
    def take_action(self, 
                    obs_dict: Dict[str, np.ndarray], # Take action receives raw obs_dict.
                    hidden_states_dict: Dict[str, np.ndarray], 
                    epsilon: float) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
        """ Select Action according to the current policy (epsilon-greedy). """
        
        obs_rgb_list = []
        obs_vector_list = []
        hidden_list = []
        
        for aid in self.agent_ids:
            agent_obs = obs_dict[aid]
            
            obs_rgb_list.append(agent_obs['RGB'])
            
            agent_vector_obs = _extract_vector_obs(
                agent_obs, self.vector_keys, self.discrete_scalar_map
            )
            obs_vector_list.append(agent_vector_obs)
            
            hidden_list.append(hidden_states_dict[aid])

        obs_rgb_batch_np = np.stack(obs_rgb_list, axis=0)        # (N, C, H, W)
        obs_vector_batch_np = np.stack(obs_vector_list, axis=0)  # (N, V)
        hidden_batch_np = np.stack(hidden_list, axis=0)          # (N, H_dim)

        # (1, N, C, H, W)
        obs_rgb_tensor = torch.tensor(obs_rgb_batch_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        # (1, N, V)
        obs_vector_tensor = torch.tensor(obs_vector_batch_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        # (1, N, H_dim)
        hidden_tensor = torch.tensor(hidden_batch_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.agent_network.eval() 
        
        q_values, next_hidden_tensor = self.agent_network(
            obs_rgb_tensor, obs_vector_tensor, hidden_tensor
        )
        # q_values: (1, N, n_actions), next_hidden_tensor: (1, N, hidden_dim)

        actions_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            agent_q_values = q_values[0, i, :]
            if random.random() < epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = agent_q_values.argmax(dim=0).item()
            actions_dict[agent_id] = action
        
        next_hidden_np = next_hidden_tensor.squeeze(0).cpu().numpy()
        next_hidden_states_dict = {
            agent_id: next_hidden_np[i] for i, agent_id in enumerate(self.agent_ids)
        }
        
        return actions_dict, next_hidden_states_dict

    def init_hidden(self) -> Dict[str, np.ndarray]:
        return {
            agent_id: np.zeros(self.rnn_hidden_dim, dtype=np.float32)
            for agent_id in self.agent_ids
        }
        
    def _get_agent_q_values(self, 
                            obs_rgb_batch: torch.Tensor,   
                            obs_vector_batch: torch.Tensor,
                            hidden_states: torch.Tensor, 
                            network_type: str = "main") -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get the Q-values of all agents. """
        agent_net = self.agent_network if network_type == "main" else self.target_agent_network
        return agent_net(obs_rgb_batch, obs_vector_batch, hidden_states)

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        """ Update the QMIX agent/learner networks. """
        self.agent_network.train() 
        self.mixer_network.train()

        obs_rgb = torch.tensor(batch["obs_rgb"], dtype=torch.float32).to(self.device)
        obs_vector = torch.tensor(batch["obs_vector"], dtype=torch.float32).to(self.device)
        state_rgb = torch.tensor(batch["state_rgb"], dtype=torch.float32).to(self.device)
        state_vector = torch.tensor(batch["state_vector"], dtype=torch.float32).to(self.device)
        
        actions = torch.tensor(batch["actions"], dtype=torch.int64).to(self.device)
        rewards_batch = torch.tensor(batch["rewards"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)
        
        next_obs_rgb = torch.tensor(batch["next_obs_rgb"], dtype=torch.float32).to(self.device)
        next_obs_vector = torch.tensor(batch["next_obs_vector"], dtype=torch.float32).to(self.device)
        next_state_rgb = torch.tensor(batch["next_state_rgb"], dtype=torch.float32).to(self.device)
        next_state_vector = torch.tensor(batch["next_state_vector"], dtype=torch.float32).to(self.device)
        
        B, L, N, C, H, W = obs_rgb.shape 

        if hasattr(self.agent_network, 'init_hidden'):
             initial_hidden = self.agent_network.init_hidden(B * N, self.device).reshape(B, N, -1)
        else:
             initial_hidden = torch.zeros(B, N, self.rnn_hidden_dim).to(self.device)

        with torch.no_grad():    # Target Q calculation
            target_q_values, _ = self._get_agent_q_values(
                next_obs_rgb, next_obs_vector, initial_hidden, "target"
            )
            main_q_values_next, _ = self._get_agent_q_values(
                next_obs_rgb, next_obs_vector, initial_hidden, "main"
            )
            
            best_next_actions = main_q_values_next.argmax(dim=3)
            target_max_q_values = torch.gather(target_q_values, dim=3, index=best_next_actions.unsqueeze(3)).squeeze(3)
            
            target_q_total, _ = self.target_mixer_network( 
                target_max_q_values, next_state_rgb, next_state_vector
            )
            target_q_total = target_q_total.squeeze(-1) 
            
            rewards_sum = rewards_batch.sum(dim=2) 
            td_target = rewards_sum + self.gamma * (1 - dones) * target_q_total    

        # Current Q calculation
        main_q_values, _ = self._get_agent_q_values(
            obs_rgb, obs_vector, initial_hidden, "main"
        )
        chosen_action_q_values = torch.gather(main_q_values, dim=3, index=actions.unsqueeze(3)).squeeze(3)
        
        current_q_total, _ = self.mixer_network( 
            chosen_action_q_values, state_rgb, state_vector
        )
        current_q_total = current_q_total.squeeze(-1) 
        
        # Loss
        loss = self.criterion(current_q_total, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        self.optimizer.step()
        
        self._train_step_count += 1
        if self._train_step_count % self.target_update_frequency == 0:
            self._update_target_networks()
        
        return loss.item()

    def _update_target_networks(self):
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixer_network.load_state_dict(self.mixer_network.state_dict())
        
    def get_networks_state_dict(self):
        return {
            'agent_network': self.agent_network.state_dict(),
            'mixer_network': self.mixer_network.state_dict()
        }

    def load_networks_state_dict(self, state_dict):
        self.agent_network.load_state_dict(state_dict['agent_network'])
        self.mixer_network.load_state_dict(state_dict['mixer_network'])
        self._update_target_networks()

def epsilon_decay(step: int, start_epsilon: float, end_epsilon: float, decay_steps: int) -> float:
    """ Epsilon decay schedule. """
    if step < decay_steps: return start_epsilon - (start_epsilon - end_epsilon) * (step / decay_steps)
    else: return end_epsilon

def evaluate(qmix_agent: QMIXAgent, 
                  env: GeneralMeltingPotWrapper, 
                  n_episodes: int) -> float:
    """ Evaluate the QMIX agent over n_episodes and return average reward. """
    total_reward = 0.0
    qmix_agent.agent_network.eval() 

    for _ in range(n_episodes):
        obs_dict, state_dict = env.reset()
        state = state_dict["concat_state"] 
        
        hidden_states_dict = qmix_agent.init_hidden()
        done = False
        
        while not done:
            actions_dict, next_hidden_states_dict = qmix_agent.take_action(
                obs_dict, hidden_states_dict, epsilon=0.0 # Epsilon=0
            )
            
            next_obs_dict, next_state, rewards_dict, dones_dict, _ = env.step(actions_dict)
            done = any(dones_dict.values())
            total_reward += sum(rewards_dict.values())
            
            obs_dict = next_obs_dict
            state = next_state 
            hidden_states_dict = next_hidden_states_dict
            
    avg_reward = total_reward / n_episodes
    return avg_reward

def train_QMIX(qmix_agent: QMIXAgent,
               env: GeneralMeltingPotWrapper, 
               config: dict,
               replay_buffer: OffPolicyReplayBuffer) -> Tuple[List[Dict], List[Dict]]:
    print("--- Start QMIX Training ---")
    return_list: List[Dict] = []      
    return_list_eval: List[Dict] = [] 
    
    best_eval_reward = -float('inf')
    save_dir = config['models_dir']
    os.makedirs(save_dir, exist_ok=True)
    run_name = config.get('run_name', f"QMIX_{config['env_name']}_{config['seed']}")
    save_path = os.path.join(save_dir, f"{run_name}.pth")
    
    total_steps = 0
    i_episode = 0

    if config['learning_starts'] > 0:
        print(f"Start warming up for {config['learning_starts']} steps...")

        obs_dict, state_dict = env.reset()
        state = state_dict["concat_state"]     # {"rgb": stack..., "vector": stack...}
        # if want use global as state, use: state = state_dict["global_state"]
        # return {"world_rgb": ..., "all_vector": stack...}
        
        for step in range(config['learning_starts']):
            actions_dict = {aid: random.randint(0, env.n_actions - 1) for aid in env.agent_ids}
            next_obs_dict, next_state, rewards_dict, dones_dict, info_dict = env.step(actions_dict)
            
            replay_buffer.push(obs_dict, state, actions_dict, rewards_dict, dones_dict, next_obs_dict, next_state, info_dict)
            
            obs_dict, state = next_obs_dict, next_state 
            if any(dones_dict.values()): 
                obs_dict, state_dict = env.reset()
                state = state_dict["concat_state"]
            if (step + 1) % 1000 == 0: 
                print(f"  Warming up {step+1}/{config['learning_starts']}...")
        print("Warming up completed.")
        total_steps = config['learning_starts']
    
    else:
        obs_dict, state_dict = env.reset()
        state = state_dict["concat_state"]
        hidden_states_dict = qmix_agent.init_hidden()

    while total_steps < config['total_timesteps']:
        """ Main training loop for QMIX agent. """
        obs_dict, state_dict = env.reset()             # original value (not one-hot)
        state = state_dict["concat_state"]
        hidden_states_dict = qmix_agent.init_hidden()
        done = False
        episode_return = 0.0
        episode_steps = 0
        i_episode += 1
          
        qmix_agent.agent_network.train()
        qmix_agent.mixer_network.train()

        while not done:
            epsilon = epsilon_decay(
                total_steps - config['learning_starts'],
                config['epsilon_start'], config['epsilon_end'], config['epsilon_decay_steps']
            )
            
            actions_dict, next_hidden_states_dict = qmix_agent.take_action(
                obs_dict, hidden_states_dict, epsilon
            )
            
            next_obs_dict, next_state, rewards_dict, dones_dict, info_dict = env.step(actions_dict)
            done = any(dones_dict.values()) or episode_steps + 1 >= env.episode_limit

            replay_buffer.push(
                obs_dict, state, actions_dict, rewards_dict, dones_dict,
                next_obs_dict, next_state, info_dict
            )
            
            obs_dict = next_obs_dict
            state = next_state
            hidden_states_dict = next_hidden_states_dict
            
            current_step_reward = sum(rewards_dict.values())
            episode_return += current_step_reward
            total_steps += 1
            episode_steps += 1

            if total_steps > config['learning_starts']:
                try:
                    batch = replay_buffer.sample(config['batch_size'], config['sequence_length'])
                    loss = qmix_agent.update(batch) 
                    
                    if total_steps % 100 == 0:
                        swanlab.log({"Train/Loss": loss}, step=total_steps)
                except ValueError as e:
                    if total_steps % 100 == 0:
                        print(f"Step {total_steps}: not able to sample... {e}")
            
            if done:
                break

        return_list.append({'steps': total_steps, 'reward': episode_return, 'seed': config['seed']})
        swanlab.log({"Return/by_Episode": episode_return}, step=i_episode)
        swanlab.log({
            "Return/by_Step": episode_return,
            "Train/Epsilon_by_Step": epsilon,
            "Episode/length_by_Step": episode_steps
        }, step=total_steps)
        
        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(qmix_agent, env, n_episodes=3) 
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"Episode: {i_episode}, Steps: {total_steps}/{config['total_timesteps']}, "
                  f"Eval Avg Return: {eval_reward:.2f}, Epsilon: {epsilon:.3f}")
            return_list_eval.append({'steps': total_steps, 'reward': eval_reward, 'seed': config['seed']})

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"  [New Best Model] Saving model with eval reward: {best_eval_reward:.2f} to {save_path}")
                torch.save(qmix_agent.get_networks_state_dict(), save_path)

    env.close()
    print("--- QMIX Training Finished ---")
    return return_list, return_list_eval

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    seeds = [1, 42, 100, 2024]
    all_seeds_data_list = []
    all_seeds_data_eval_list = []
    multi_algo_data = {}
    multi_algo_data_eval = {}

    for seed in seeds:
        print(f"\n===== Starting execution with Seed: {seed} =====\n")
        run_name = f"QMIX-MeltingPot-{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
        run = swanlab.init(
            project="QMIX-MeltingPot-MultiSeed-v1",
            experiment_name=run_name,
            config = {
                "lr": 0.0005, 
                "gamma": 0.99, 
                "grad_norm_clip": 10.0,
                "rnn_hidden_dim": 64, 
                "mixing_embed_dim": 32,
                "total_timesteps": 2_000_000, 
                "target_update_frequency": 200,
                "buffer_capacity": 5000, 
                "sequence_length": 10,
                "batch_size": 32,
                "learning_starts": 10000,   
                "epsilon_start": 1.0, 
                "epsilon_end": 0.05, 
                "epsilon_decay_steps": 100_000,
                "env_name": 'collaborative_cooking__asymmetric',
                "models_dir": 'models',
                "eval_freq": 10, 
                "seed": seed, 
                "device": device_str
            }
        )

        config = swanlab.config
        config["run_name"] = run_name
        
        set_seed(config['seed'])
        
        env = GeneralMeltingPotWrapper(env_name=config['env_name'])
        env_info = env.get_env_info()
        
        replay_buffer = OffPolicyReplayBuffer(env_info, config['buffer_capacity'])
        
        multi_agent = QMIXAgent(env_info, config)
        
        print(f"Using device: {config['device']}")

        train_QMIX(multi_agent, env, config, replay_buffer)

        single_run_data, single_run_data_eval = train_QMIX(multi_agent, env, config, replay_buffer)
        
        all_seeds_data_list.extend(single_run_data)
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()


    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['QMIX'] = all_seeds_data_list
    multi_algo_data_eval['QMIX'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data, 
                        title="QMIX Training Performance on MeltingPot", 
                        output_filename="qmix_meltingpot_training_curve.png")
    plot_learning_curve(experiments_data=multi_algo_data_eval, 
                        title="QMIX Evaluation Performance on MeltingPot", 
                        output_filename="qmix_meltingpot_evaluation_curve.png")