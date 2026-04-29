import numpy as np
import swanlab
import torch
import os
from tqdm import tqdm

from utils.evaluator import Evaluator 

def train(args, env, eval_env, agent, buffer):
    """
    Main training loop for On-Policy MARL algorithms (IPPO/MAPPO) on Melting Pot.
    
    Assumptions about Agent API:
    1. agent.take_action(...) -> values, actions, log_probs, next_hidden_state
       (All outputs except hidden_state should be numpy arrays)
    2. agent.update(buffer) -> train_info (dict)
       (Agent handles GAE calculation internally using the buffer)
    3. agent.init_hidden(num_envs) -> hidden_state (Tensor)
    """
    # Create models directory
    model_dir = os.path.join(args.run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"--> Start On-Policy Training | Seed: {args.seed} | Log: {args.run_dir}")

    # Initialize Evaluator
    evaluator = Evaluator(eval_env, agent, args.device, policy_type='on-policy', seed=getattr(args, 'eval_seed', None))

    train_stats = []
    eval_stats = []

    total_steps = 0
    i_episode = 0
    best_return = -float('inf')

    # === 1. Environment Reset & Initial State ===
    obs_dict, infos = env.reset()
    
    # Calculate Initial State (Flatten & Global)
    B, N, C, H, W = obs_dict['rgb'].shape
    state_flatten = obs_dict['rgb'].reshape(B, -1, H, W)
    
    state_global = None
    if 'global_state' in infos[0]:
        s_global = np.stack([info['global_state']['world_rgb'] for info in infos], axis=0)
        # Normalize & Transpose -> (B, C, H, W)
        state_global = np.transpose(s_global.astype(np.float32) / 255.0, (0, 3, 1, 2))

    # agent.init_hidden returns shape (Batch, N_Agents, Layers, Hidden)
    hidden_state = agent.init_hidden(args.num_envs)
    
    # Metrics
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    
    pbar = tqdm(total=args.max_steps, desc=f"Seed {args.seed}")
    
    # On-Policy Loop: Run until max_steps
    while total_steps < args.max_steps:
        
        # Collect 'buffer_size' steps
        for step in range(args.buffer_size):
            
            obs_tensor_dict = {
                'rgb': torch.from_numpy(obs_dict['rgb']).to(args.device),
                'vector': torch.from_numpy(obs_dict['vector']).to(args.device),
            }
            
            state_tensor = None
            if state_flatten is not None:
                state_tensor = torch.from_numpy(state_flatten).to(args.device)
            
            global_state_tensor = None
            if state_global is not None:
                global_state_tensor = torch.from_numpy(state_global).to(args.device)

            # Agent should return numpy arrays for values, actions, log_probs
            values, actions, log_probs, next_hidden_state = agent.take_action(
                obs=obs_tensor_dict,
                state=state_tensor,
                global_state=global_state_tensor,
                hidden_state=hidden_state
            )

            # Environment Step
            next_obs_dict, rewards, dones, infos = env.step(actions)
            
            hidden_np = hidden_state.cpu().numpy()
            
            # Calculate Next State
            B, N, C, H, W = next_obs_dict['rgb'].shape
            next_state_flatten = next_obs_dict['rgb'].reshape(B, -1, H, W)
            next_state_global = None
            if 'global_state' in infos[0]:
                ns_global = np.stack([info['global_state']['world_rgb'] for info in infos], axis=0)
                next_state_global = np.transpose(ns_global.astype(np.float32) / 255.0, (0, 3, 1, 2))

            # Store in Buffer
            buffer.push(
                obs=obs_dict,
                state=state_flatten,
                global_state=state_global,
                hidden_states=hidden_np,
                actions=actions,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs,
                values=values
            )
            
            # --- Stats & Logging ---
            for i in range(args.num_envs):
                episode_returns[i] += np.sum(rewards[i])
                episode_lengths[i] += 1
                
                done_bool = np.any(dones[i])
                if done_bool:
                    i_episode += 1
                    final_reward = episode_returns[i]
                    if 'collective_reward' in infos[i]:
                        final_reward = infos[i]['collective_reward']

                    if i_episode % getattr(args, 'log_freq', 100) == 0:
                        swanlab.log({
                            "Train/Reward": final_reward,
                            "Train/Episode_Length": episode_lengths[i],
                        }, step=total_steps)
                        pbar.write(f" [Train] Steps {total_steps} | Ep {i_episode}: Reward = {final_reward:.2f}")
                        
                        train_stats.append({
                            'steps': total_steps,
                            'reward': final_reward,
                            'len': episode_lengths[i],
                            'seed': args.seed
                        })

                    # Evaluation
                    if i_episode % getattr(args, 'eval_freq', 1000) == 0:
                        eval_seed_to_use = getattr(args, 'eval_seed', args.seed)
                        avg_reward = evaluator.evaluate(n_episodes=args.eval_episodes, seed=eval_seed_to_use)
                        
                        swanlab.log({"Eval/Return": avg_reward}, step=total_steps)
                        pbar.write(f" [Eval] Steps {total_steps}: Mean Reward = {avg_reward:.2f}")
                        
                        eval_stats.append({
                            'steps': total_steps, 'reward': avg_reward, 'seed': args.seed
                        })

                        agent.save(os.path.join(model_dir, "model_latest.pth"))
                        if avg_reward > best_return:
                            best_return = avg_reward
                            agent.save(os.path.join(model_dir, "model_best.pth"))
                            pbar.write(f"   >>> [New Best] Saved model_best.pth (Return: {best_return:.2f})")

                    episode_returns[i] = 0
                    episode_lengths[i] = 0

            # Update Loop Variables
            obs_dict = next_obs_dict
            state_flatten = next_state_flatten
            state_global = next_state_global
            hidden_state = next_hidden_state.detach()

            # RNN State Reset for Done Envs
            if np.any(dones):
                env_dones = np.any(dones, axis=1)
                done_indices = np.where(env_dones)[0]
                if len(done_indices) > 0:
                    new_hidden = agent.init_hidden(len(done_indices))
                    hidden_state[done_indices] = new_hidden

            total_steps += args.num_envs
            pbar.update(args.num_envs)

        # Use take_action to get values for T+1 (reusing Agent logic, ignoring actions)
        with torch.no_grad():
            obs_tensor_dict = {
                'rgb': torch.from_numpy(obs_dict['rgb']).to(args.device),
                'vector': torch.from_numpy(obs_dict['vector']).to(args.device),
            }
            state_tensor = torch.from_numpy(state_flatten).to(args.device) if state_flatten is not None else None
            global_state_tensor = torch.from_numpy(state_global).to(args.device) if state_global is not None else None
            
            # We only need 'values' for bootstrapping
            next_values, _, _, _ = agent.take_action(
                obs=obs_tensor_dict, 
                state=state_tensor, 
                global_state=global_state_tensor, 
                hidden_state=hidden_state
            )

        # Insert the final step data (Bootstrap Value & Last Dones)
        buffer.insert_last_step(
            obs=obs_dict,
            state=state_flatten,
            global_state=state_global,
            hidden_states=hidden_state.cpu().numpy(),
            values=next_values,
            dones=dones 
        )

        train_info = agent.update(buffer)
        
        if train_info:
            swanlab.log(train_info, step=total_steps)

        # Clear buffer for next rollout
        buffer.clear()

    pbar.close()
    return train_stats, eval_stats