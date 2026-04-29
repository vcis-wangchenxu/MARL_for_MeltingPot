import argparse
import os
import torch
import swanlab
import glob
import pickle
import copy
import sys

from utils.util import set_seed, save_results
from envs.multi_envs import make_env_fn, MeltingPotAsyncVectorEnv
from utils.config import load_algorithm, get_config
from utils.evaluator import Evaluator
from memories.ReplayBuffer import ParallelReplayBuffer
from memories.RolloutBuffer import ParallelRolloutBuffer

def run_train_single_seed(cfg):

    algo_path = cfg.algo
    algo_name = algo_path.split('.')[-1]

    policy_type = cfg.policy_type

    if "on-policy" in policy_type.lower():
        from train_onpolicy import train
        print(f"[{algo_name}] Policy Type: {policy_type} -> Loading train_onpolicy")
    else:
        from train_offpolicy import train
        print(f"[{algo_name}] Policy Type: {policy_type} -> Loading train_offpolicy")

    device_str = "cuda" if torch.cuda.is_available() and getattr(cfg, "device", "cpu") == "cuda" else "cpu"
    cfg.device = device_str
    print(f"[{algo_name}] Seed {cfg.seed} | Device: {cfg.device}")

    set_seed(getattr(cfg, 'seed', 0))

    swanlab.init(
        project=getattr(cfg, "project_name", "MeltingPot-MARL"),
        experiment_name=f"{algo_name}_{cfg.env}_seed{cfg.seed}",
        config=vars(cfg),
        logdir=os.path.join(cfg.run_dir, "logs"), 
        mode="cloud" if getattr(cfg, "use_swanlab", True) else "disabled",
    )

    dummy_env = make_env_fn(cfg.env)()
    env_info = dummy_env.get_env_info()
    dummy_env.close()
    print(f"Env Info: {env_info}")

    num_envs = getattr(cfg, "num_envs", 8)
    train_env_fns = [make_env_fn(cfg.env) for _ in range(num_envs)]
    train_env = MeltingPotAsyncVectorEnv(train_env_fns)

    # Determine Evaluation Environment and Seed
    # If eval_env is not specified, use training env.
    eval_env_id = getattr(cfg, "eval_env", cfg.env)

    # If eval_seed is not specified, use training seed + 10000
    eval_seed = getattr(cfg, "eval_seed", cfg.seed + 10000)
    
    # Update cfg with these values so they are passed to train() and Evaluator
    cfg.eval_env = eval_env_id
    cfg.eval_seed = eval_seed

    print(f"[{algo_name}] Eval Env: {eval_env_id} | Eval Seed: {eval_seed}")

    eval_env_fn = make_env_fn(eval_env_id)
    eval_env = MeltingPotAsyncVectorEnv([eval_env_fn])

    hidden_dim = getattr(cfg, "hidden_dim", 64)
    rnn_layers = getattr(cfg, "rnn_layers", 1)

    buffer_env_info = env_info.copy()
    buffer_env_info['obs_shape'] = {
        'rgb': env_info['obs_rgb_shape'],
        'vector': env_info['obs_vector_dim']    
    }
    buffer_env_info['state_shape'] = env_info['state_shape']
    if 'global_shape' in env_info and env_info['global_shape'] is not None:
        H, W, C = env_info['global_shape']
        buffer_env_info['global_state_shape'] = (C, H, W)
    else:
        buffer_env_info['global_state_shape'] = None

    if "on-policy" in policy_type.lower():
        replay_buffer = ParallelRolloutBuffer(
            num_envs=num_envs,
            env_info=buffer_env_info,
            buffer_size=getattr(cfg, "buffer_size", 128), # Rollout length
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            device=cfg.device
        )
        print(f"[{algo_name}] Initialized ParallelRolloutBuffer (On-Policy)")
    else:
        replay_buffer = ParallelReplayBuffer(
            num_envs=num_envs,
            env_info=buffer_env_info,   
            capacity=getattr(cfg, "buffer_capacity", 50000),
            batch_size=getattr(cfg, "batch_size", 32),
            sequence_length=getattr(cfg, "sequence_length", 8),
            hidden_dim=hidden_dim, 
            rnn_layers=rnn_layers,
            device=cfg.device
        )
        print(f"[{algo_name}] Initialized ParallelReplayBuffer (Off-Policy)")

    AlgoClass = load_algorithm(algo_path)
    agent = AlgoClass(env_info, cfg) 

    try:
        train_stats, eval_stats = train(
            args=cfg,
            env=train_env,
            eval_env=eval_env,
            agent=agent,
            buffer=replay_buffer,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        train_stats, eval_stats = [], []
    finally:
        swanlab.finish()
        train_env.close()
        eval_env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return algo_name, train_stats, eval_stats

def run_train_multi_seeds(base_cfg):
    """
    Run training for multiple seeds sequentially and save aggregated results.

    Args:
        base_cfg (argparse.Namespace): Base configuration object.
        seeds (list): List of random seeds to run.
    """
    seeds = base_cfg.seeds
    algo_name = base_cfg.algo.split(".")[-1]
    env_name = base_cfg.env

    # Define Base Results Directory: results/{env_name}/{algo_name}/
    base_results_dir = os.path.join(base_cfg.run_dir, env_name, algo_name)

    print(f"\n{'='*60}")
    print(f"Algorithm: {algo_name} | Environment: {env_name}")
    print(f"Seeds: {seeds}")
    print(f"Results Dir: {base_results_dir}")
    print(f"{'='*60}")

    for seed in seeds:
        print(f"\n>>> Running Seed: {seed}")
        cfg = argparse.Namespace(**vars(base_cfg))
        cfg.seed = seed

        # Set specific run directory for this seed: results/{env}/{algo}/seed_{seed}
        cfg.run_dir = os.path.join(base_results_dir, f"seed_{seed}")
        os.makedirs(cfg.run_dir, exist_ok=True)

        _, train_stats, eval_stats = run_train_single_seed(cfg)

        # Save individual seed results locally in its folder
        save_results(train_stats, os.path.join(cfg.run_dir, "train_data.pkl"))
        save_results(eval_stats, os.path.join(cfg.run_dir, "eval_data.pkl"))
        
        print(f"[Seed {seed}] Data saved to {cfg.run_dir}")

    print(f"\nAll seeds finished. To visualize, run: python run.py --mode plot --env {env_name} --run_dir {base_cfg.run_dir}")
    
def run_plot(cfg):
    """
    Plot learning curves for BOTH training and evaluation data.
    Structure: run_dir/{env_name}/{algo_name}/seed_{seed}/*.pkl
    Generates two plots: one for eval, one for train.
    """
    from utils.util import plot_learning_curve

    env_name = cfg.env
    root_dir = os.path.join(cfg.run_dir, env_name)

    if not os.path.exists(root_dir):
        print(f"[Error] No results found for environment '{env_name}' in '{cfg.run_dir}'")
        return

    print(f"Scanning {root_dir} for algorithms...")

    algo_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    if not algo_dirs:
        print("No algorithm directories found.")
        return

    # Define the tasks: (File Suffix, Plot Title Suffix, Output Suffix, Bin Size)
    # Bin Size Recommendation:
    # - Eval: Usually sparse (e.g. every 1000 steps), bin_size=1000 or 1 is fine.
    # - Train: Very dense (every episode), bin_size needs to be larger (e.g. 1000-5000) to smooth out noise.
    plot_tasks = [
        {
            "type": "Evaluation",
            "filename": "eval_data.pkl",
            "title": f"Evaluation Curve: {env_name}",
            "output_suffix": "_eval.png",
            "bin_size": 1000 
        },
        {
            "type": "Training",
            "filename": "train_data.pkl",
            "title": f"Training Curve: {env_name}",
            "output_suffix": "_train.png",
            "bin_size": 2000 # Larger bin size for training data to reduce jitter
        }
    ]

    for task in plot_tasks:
        print(f"\n>>> Processing {task['type']} Data...")
        experiments_data = {}

        for algo in algo_dirs:
            algo_path = os.path.join(root_dir, algo)
            seed_dirs = glob.glob(os.path.join(algo_path, "seed_*"))
            
            algo_all_records = []
            for s_dir in seed_dirs:
                # Parse seed
                try:
                    seed_val = int(os.path.basename(s_dir).split("_")[-1])
                except ValueError:
                    seed_val = 0
                
                target_file = os.path.join(s_dir, task['filename'])
                
                if os.path.exists(target_file):
                    try:
                        with open(target_file, 'rb') as f:
                            data = pickle.load(f) # List of dicts
                            if data:
                                # Ensure 'seed' is in the dictionary
                                for record in data:
                                    if 'seed' not in record:
                                        record['seed'] = seed_val
                                
                                algo_all_records.extend(data)
                                # print(f"  Loaded {algo} - Seed {seed_val}: {len(data)} points")
                    except Exception as e:
                        print(f"  Error loading {target_file}: {e}")
            
            if algo_all_records:
                experiments_data[algo] = algo_all_records
                print(f"  [Loaded] {algo}: Total {len(algo_all_records)} records across all seeds.")

        if not experiments_data:
            print(f"No valid {task['type']} data found to plot.")
            continue

        # Generate Plot
        output_path = os.path.join(root_dir, f"benchmark_{env_name}{task['output_suffix']}")
        
        plot_learning_curve(
            experiments_data, 
            bin_size=task['bin_size'], 
            title=task['title'],
            output_filename=output_path
        )

def run_eval_only(cfg):
    """
    Evaluation mode.
    Can load 'best' or 'latest' model automatically from the standard directory structure,
    or a specific checkpoint file.
    """
    algo_name = cfg.algo.split(".")[-1]

    # Determine Checkpoint Path
    checkpoint_path = getattr(cfg, "checkpoint", None)
    model_type = getattr(cfg, "model_type", "best") # 'best' or 'latest'

    if checkpoint_path is None:
        # Try to infer path: results/{env}/{algo}/seed_{seed}/models/model_{type}.pth
        # Note: If multiple seeds exist and user didn't specify, we default to the first one in list or user must specify seed.
        seed = getattr(cfg, "seed", 1) # Default to seed 1 if not set
        
        inferred_dir = os.path.join(cfg.run_dir, cfg.env, algo_name, f"seed_{seed}", "models")
        target_file = f"model_{model_type}.pth"
        
        potential_path = os.path.join(inferred_dir, target_file)
        
        if os.path.exists(potential_path):
            checkpoint_path = potential_path
            print(f"[Eval] Auto-detected model: {checkpoint_path}")
        else:
            print(f"[Error] Could not find model at: {potential_path}")
            print("Please provide --checkpoint PATH or check your --seed and directory structure.")
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    
    # Determine Evaluation Environment
    # If eval_env is provided, use it. Otherwise default to cfg.env (training env)
    eval_env_id = getattr(cfg, "eval_env", cfg.env)
    print(f"[Eval] Target Environment: {eval_env_id}")

    
    eval_env_fn = make_env_fn(eval_env_id)
    eval_env = MeltingPotAsyncVectorEnv([eval_env_fn])

    env_info = eval_env.get_env_info()

    AlgoClass = load_algorithm(cfg.algo)
    agent = AlgoClass(env_info, cfg)
    
    print(f"Loading model from {checkpoint_path}...")
    agent.load(checkpoint_path)
    
    # Determine Eval Seed
    eval_seed = getattr(cfg, "eval_seed", None)
    if eval_seed is None:
        # If not specified, use a different seed from training to ensure fairness, or just random
        eval_seed = getattr(cfg, 'seed', 0) + 2024
    
    policy_type = cfg.policy_type
    # policy_type = getattr(cfg, "policy_type", getattr(cfg, "policy", "off-policy"))
    evaluator = Evaluator(eval_env, agent, cfg.device, policy_type=policy_type, seed=eval_seed)
    
    n_episodes = getattr(cfg, "eval_episodes", 10)
    
    print(f"[Eval] Running {n_episodes} episodes on seed {eval_seed}...")
    mean_ret = evaluator.evaluate(n_episodes=n_episodes, seed=eval_seed)
    print(f"\n>>> Final Evaluation Return ({model_type}) on {eval_env_id}: {mean_ret}")
    eval_env.close()

def main():
    parser = argparse.ArgumentParser(description="Run MARL for MeltingPot")
    parser.add_argument("--mode", type=str, default="train-multi",
                        choices=["train-multi", "eval", "plot"],
                        help="Mode: train-multi (default), eval, plot")
    
    parser.add_argument("--run_dir", type=str, default="results",
                        help="Root directory for results")
    
    # Eval specific args
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth model file")
    parser.add_argument("--model_type", type=str, default="best", choices=["best", "latest"], 
                        help="If checkpoint not specified, load 'best' or 'latest' from default path")
    
    parser.add_argument("--eval_env", type=str, default=None, help="Environment to evaluate on (overrides config env)")
    parser.add_argument("--eval_seed", type=int, default=None, help="Seed for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")

    args, unknown = parser.parse_known_args()
    
    # Pass args to sys.argv so get_config can parse them if needed, or manually update cfg
    sys.argv = [sys.argv[0]] + unknown
    
    # Load Config
    is_train = (args.mode == "train-multi")
    try:
        cfg = get_config(is_train=is_train)
    except Exception as e:
        if args.mode == "plot":
            # For plot mode, we might not need a full valid config file if we just need env name
            # But get_config requires --config. User should provide a dummy or real config.
            print(f"Warning parsing config for plot: {e}")
            cfg = argparse.Namespace()
        else:
            raise e

    # Override config with CLI args
    cfg.run_dir = args.run_dir
    cfg.model_type = args.model_type
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    
    if args.eval_env:
        cfg.eval_env = args.eval_env
    if args.eval_seed is not None:
        cfg.eval_seed = args.eval_seed
    cfg.eval_episodes = args.eval_episodes
    
    # For Plot mode, we need 'env' from CLI if not in config, but config is usually loaded.
    # Check if 'env' is in unknown args if not in config
    if not hasattr(cfg, 'env'):
        parser_env = argparse.ArgumentParser()
        parser_env.add_argument("--env", type=str, default=None)
        args_env, _ = parser_env.parse_known_args(unknown)
        if args_env.env:
            cfg.env = args_env.env

    if args.mode == "train-multi":
        run_train_multi_seeds(cfg)

    elif args.mode == "plot":
        run_plot(cfg)

    elif args.mode == "eval":
        run_eval_only(cfg)

if __name__ == "__main__":
    main()
