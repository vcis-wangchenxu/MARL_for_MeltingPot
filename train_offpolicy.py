import numpy as np
import swanlab
import torch
import os
from tqdm import tqdm

from utils.evaluator import Evaluator 

def train(args, env, eval_env, agent, buffer):
    """
    Main training loop for Off-Policy MARL algorithms (VDN) on Melting Pot.
    
    Args:
        args (Namespace): Configuration parameters.
        env (MeltingPotAsyncVectorEnv): Vectorized training environment.
        eval_env (MeltingPotAsyncVectorEnv): Evaluation environment (Single process).
        agent (VDN_MeltingPot): The agent instance.
        buffer (ParallelReplayBuffer_MP): Experience replay buffer.
    """
    # Create models directory
    model_dir = os.path.join(args.run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"--> Start Off-Policy Training | Seed: {args.seed} | Log: {args.run_dir}")

    # Initialize Evaluator
    evaluator = Evaluator(eval_env, agent, args.device, policy_type='off-policy', seed=getattr(args, 'eval_seed', None))

    train_stats = []
    eval_stats = []

    total_steps = 0
    i_episode = 0
    best_return = -float('inf')

    obs_dict, infos = env.reset()
    # obs_dict: {'rgb': (B, N, C, H, W), 'vector': (B, N, V)}

    B, N, C, H, W = obs_dict['rgb'].shape
    state_flatten = obs_dict['rgb'].reshape(B, -1, H, W)

    state_global = None
    if 'global_state' in infos[0]:
        s_global = np.stack([info['global_state']['world_rgb'] for info in infos], axis=0) # (B, H, W, C)
        # Normalize & Transpose -> (B, C, H, W)
        state_global = np.transpose(s_global.astype(np.float32) / 255.0, (0, 3, 1, 2))
    
    # agent.init_hidden returns shape (Batch, N_Agents, Layers, Hidden)
    hidden_state = agent.init_hidden(args.num_envs)
    
    # Metrics
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    
    pbar = tqdm(total=args.max_steps, desc=f"Seed {args.seed}")
    current_loss = 0.0

    while total_steps < args.max_steps:
        
        obs_tensor_dict = {
            'rgb': torch.from_numpy(obs_dict['rgb']).to(args.device),
            'vector': torch.from_numpy(obs_dict['vector']).to(args.device),
        }

        actions, next_hidden_state = agent.take_action(
            obs_tensor_dict, 
            hidden_state, 
            current_step=total_steps
        )

        # MP Wrapper returns: next_obs (dict), rewards (arr), dones (arr), infos (list)
        next_obs_dict, rewards, dones, infos = env.step(actions)
        
        hidden_np = hidden_state.cpu().numpy()
        
        B, N, C, H, W = next_obs_dict['rgb'].shape
        next_state_flatten = next_obs_dict['rgb'].reshape(B, -1, H, W)

        next_state_global = None
        if 'global_state' in infos[0]:
            ns_global = np.stack([info['global_state']['world_rgb'] for info in infos], axis=0)
            next_state_global = np.transpose(ns_global.astype(np.float32) / 255.0, (0, 3, 1, 2))
            
        buffer.push(
            obs=obs_dict,
            hidden=hidden_np,
            state=state_flatten,
            global_state=state_global,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs_dict,
            next_state=next_state_flatten,
            next_global_state=next_state_global
        )
        
        for i in range(args.num_envs):
            # Sum rewards across agents (Cooperative)
            episode_returns[i] += np.sum(rewards[i])
            episode_lengths[i] += 1
            
            done_bool = np.any(dones[i])  # Melting Pot agents share done status
            # Check Episode Done (Melting Pot agents share done status)
            if done_bool: 
                i_episode += 1
                
                # Retrieve actual collective reward from info if available (more accurate)
                final_reward = episode_returns[i]
                if 'collective_reward' in infos[i]:
                    final_reward = infos[i]['collective_reward']

                # Log Training Stats
                if i_episode % getattr(args, 'log_freq', 100) == 0:
                    swanlab.log({
                        "Train/Loss": current_loss,
                        "Train/Epsilon": agent.epsilon,
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
                    eval_seed_to_use = getattr(args, 'eval_seed', None)
                    avg_reward = evaluator.evaluate(n_episodes=args.eval_episodes, seed=eval_seed_to_use)
                    
                    swanlab.log({"Eval/Return": avg_reward}, step=total_steps)
                    pbar.write(f" [Eval] Steps {total_steps}: Mean Reward = {avg_reward:.2f}")
                    
                    eval_stats.append({
                        'steps': total_steps, 
                        'reward': avg_reward, 
                        'seed': args.seed
                    })

                    # Save Checkpoints
                    agent.save(os.path.join(model_dir, "model_latest.pth"))
                    if avg_reward > best_return:
                        best_return = avg_reward
                        agent.save(os.path.join(model_dir, "model_best.pth"))
                        pbar.write(f"   >>> [New Best] Saved model_best.pth (Return: {best_return:.2f})")

                # Reset counters for this env
                episode_returns[i] = 0
                episode_lengths[i] = 0
        
        obs_dict = next_obs_dict
        state_flatten = next_state_flatten
        state_global = next_state_global
        hidden_state = next_hidden_state.detach()

        # Handle RNN Hidden State Reset for Done Envs
        if np.any(dones):
            env_dones = np.any(dones, axis=1) # (B,)
            done_indices = np.where(env_dones)[0]
            
            if len(done_indices) > 0:
                # Generate fresh zero hidden states for reset environments
                # Shape: (N_Done, N_Agents, Layers, Hidden)
                new_hidden = agent.init_hidden(len(done_indices))
                hidden_state[done_indices] = new_hidden

        total_steps += args.num_envs
        pbar.update(args.num_envs)
        
        if len(buffer) >= args.batch_size and total_steps > args.warmup_steps:
            if total_steps % getattr(args, 'train_freq', 1) == 0:
                batch = buffer.sample()
                if batch is not None:
                    current_loss = agent.update(batch)

    pbar.close()
    return train_stats, eval_stats