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
from models import AgentDRQN, QMixer
from memories import OffPolicyReplayBuffer
from MeltingPotWrapper import build_meltingpot_env as GeneralMeltingPotWrapper
# -------------------------------------------------------------------
class QMIXAgent:
    """ QMIX Agent Class """
    def __init__(self, env_info: dict, args: Dict):
        """ Initialize QMIX agent/learner. """
        self.env_info = env_info                      # environment information dictionary
        self.n_agents = env_info["n_agents"]          # number of agents
        self.n_actions = env_info["n_actions"]        # number of actions
        self.obs_shape = env_info["obs_shape"]        # observation shape (for each agent)
        self.state_shape = env_info["state_shape"]    # global state shape
        self.agent_ids = env_info["agent_ids"]        # list of agent IDs
        
        self.args = args                               # training arguments
        self.lr = args["lr"]                           # learning rate
        self.gamma = args["gamma"]                     # discount factor
        self.rnn_hidden_dim = args["rnn_hidden_dim"]   # RNN hidden dimension
        self.mixing_embed_dim = args["mixing_embed_dim"]    # mixing network embedding dimension
        self.grad_norm_clip = args["grad_norm_clip"]        # gradient norm clipping value
        self.target_update_frequency = args["target_update_frequency"]     # target network update frequency

        self.device = torch.device(args["device"])

        self.agent_network = AgentDRQN(
            self.obs_shape, self.n_actions, self.rnn_hidden_dim
        ).to(self.device)
        self.target_agent_network = copy.deepcopy(self.agent_network)

        self.mixer_network = QMixer(
            self.n_agents, self.state_shape, self.mixing_embed_dim
        ).to(self.device)
        self.target_mixer_network = copy.deepcopy(self.mixer_network)

        self.params = list(self.agent_network.parameters()) + list(self.mixer_network.parameters())
        self.optimizer = optim.Adam(params=self.params, lr=self.lr)
        self.criterion = nn.MSELoss()

        self._train_step_count = 0 

    @torch.no_grad()
    def take_action(self, 
                    obs_dict: Dict[str, np.ndarray], 
                    hidden_states_dict: Dict[str, np.ndarray], 
                    epsilon: float) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
        """ Select Action according to the current policy (epsilon-greedy)."""
        obs_list = [obs_dict[aid]['RGB'] for aid in self.agent_ids]        # (C, H, W)
        hidden_list = [hidden_states_dict[aid] for aid in self.agent_ids]  # (hidden_dim,)
        
        obs_batch_np = np.stack(obs_list, axis=0)           # (N, C, H, W)
        hidden_batch_np = np.stack(hidden_list, axis=0)     # (N, hidden_dim)

        obs_tensor = torch.tensor(obs_batch_np, dtype=torch.float32).unsqueeze(0).to(self.device)       # (1, N, C, H, W)
        hidden_tensor = torch.tensor(hidden_batch_np, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, N, hidden_dim)

        self.agent_network.eval() 
        
        q_values, next_hidden_tensor = self.agent_network(obs_tensor, hidden_tensor)  # q_values: (1, N, n_actions), next_hidden_tensor: (1, N, hidden_dim)

        actions_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            agent_q_values = q_values[0, i, :]          # Extract the Q-value of the i-th agent -> (n_actions,)
            
            if random.random() < epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = agent_q_values.argmax(dim=0).item()
                
            actions_dict[agent_id] = action             # Store action in the dictionary
        
        # Unpack next_hidden_tensor for storage in the environment
        next_hidden_np = next_hidden_tensor.squeeze(0).cpu().numpy() # (N, hidden_dim)
        next_hidden_states_dict = {
            agent_id: next_hidden_np[i] for i, agent_id in enumerate(self.agent_ids)
        }
        
        return actions_dict, next_hidden_states_dict

    def init_hidden(self) -> Dict[str, np.ndarray]:
        """ Return a dictionary of zero-initialized hidden states for all agents (used for environment loops) """
        return {
            agent_id: np.zeros(self.rnn_hidden_dim, dtype=np.float32)
            for agent_id in self.agent_ids
        }
        
    def _get_agent_q_values(self, 
                            obs_batch: torch.Tensor, 
                            hidden_states: torch.Tensor, 
                            network_type: str = "main") -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get the Q-values of all agents. """
        agent_net = self.agent_network if network_type == "main" else self.target_agent_network
        return agent_net(obs_batch, hidden_states)

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        """ Update the QMIX agent/learner networks. """
        self.agent_network.train() 
        self.mixer_network.train()
        
        states = torch.tensor(batch["state"], dtype=torch.float32).to(self.device)            # (B, L, state_dim)
        obs = torch.tensor(batch["obs"], dtype=torch.float32).to(self.device)                 # (B, L, N, C, H, W)
        actions = torch.tensor(batch["actions"], dtype=torch.int64).to(self.device)           # (B, L, N)
        rewards_batch = torch.tensor(batch["rewards"], dtype=torch.float32).to(self.device)   # (B, L, N)
        dones = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)             # (B, L)
        next_states = torch.tensor(batch["next_state"], dtype=torch.float32).to(self.device)  # (B, L, state_dim)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32).to(self.device)       # (B, L, N, C, H, W)
        
        B, L, N, C, H, W = obs.shape

        initial_hidden = self.agent_network.init_hidden(B * N).reshape(B, N, -1).to(self.device)  # (B, N, hidden_dim)

        with torch.no_grad():    # Target Q calculation
            target_q_values, _ = self._get_agent_q_values(next_obs, initial_hidden, "target")    # (B, L, N, n_actions)
            main_q_values_next, _ = self._get_agent_q_values(next_obs, initial_hidden, "main")   # (B, L, N, n_actions)
            
            best_next_actions = main_q_values_next.argmax(dim=3)         # (B, L, N)
            target_max_q_values = torch.gather(target_q_values, dim=3, index=best_next_actions.unsqueeze(3)).squeeze(3) # (B, L, N)
            
            target_q_total = self.target_mixer_network(target_max_q_values, next_states)    # (B, L)
            
            rewards_sum = rewards_batch.sum(dim=2) # (B, L, N) -> (B, L)
            td_target = rewards_sum + self.gamma * (1 - dones) * target_q_total    # (B, L)

        # Current Q calculation
        main_q_values, _ = self._get_agent_q_values(obs, initial_hidden, "main")    # (B, L, N, n_actions)
        chosen_action_q_values = torch.gather(main_q_values, dim=3, index=actions.unsqueeze(3)).squeeze(3)    # (B, L, N)
        current_q_total = self.mixer_network(chosen_action_q_values, states)        # (B, L)
        
        # Loss
        loss = self.criterion(current_q_total, td_target)

        # Optimization
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
        """ Get the state_dict of agent and mixer networks. """
        return {
            'agent_network': self.agent_network.state_dict(),
            'mixer_network': self.mixer_network.state_dict()
        }

    def load_networks_state_dict(self, state_dict):
        """ Load the state_dict into agent and mixer networks. """
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
        obs_dict, state = env.reset()
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
            hidden_states_dict = next_hidden_states_dict
            
    avg_reward = total_reward / n_episodes
    return avg_reward

def train_QMIX(qmix_agent: QMIXAgent,
               env: GeneralMeltingPotWrapper, 
               config: dict,
               replay_buffer: OffPolicyReplayBuffer) -> Tuple[List[Dict], List[Dict]]:
    """ Train the QMIX agent. """
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
        obs_dict, state = env.reset()
        for step in range(config['learning_starts']):
            actions_dict = {aid: random.randint(0, env.n_actions - 1) for aid in env.agent_ids}
            next_obs_dict, next_state, rewards_dict, dones_dict, info_dict = env.step(actions_dict)
            
            replay_buffer.push(obs_dict, state, actions_dict, rewards_dict, dones_dict, next_obs_dict, next_state, info_dict)
            
            obs_dict, state = next_obs_dict, next_state
            if any(dones_dict.values()): 
                obs_dict, state = env.reset()
            if (step + 1) % 1000 == 0: 
                print(f"  Warming up {step+1}/{config['learning_starts']}...")
        print("Warming up completed.")
        total_steps = config['learning_starts']
    
    else:
        obs_dict, state = env.reset()
        hidden_states_dict = qmix_agent.init_hidden()

    while total_steps < config['total_timesteps']:
        """ Main training loop for QMIX agent. """
        obs_dict, state = env.reset()
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