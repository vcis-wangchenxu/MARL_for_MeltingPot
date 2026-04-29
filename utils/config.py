import argparse
import torch
import yaml
import os
import sys

def get_config(is_train: bool = True):
    """
    Load configuration from a YAML file specified via command line arguments.

    Args:
        is_train (bool): Whether the configuration is for training (checks for 'policy_type').

    Returns:
        args (argparse.Namespace): Namespace object containing configuration parameters.
    """
    parser = argparse.ArgumentParser(description="MARL for MeltingPot Benchmark (YAML Strict Mode)")

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (Required)")
    
    # Parse command line (only parse --config)
    args_cli, unknown = parser.parse_known_args()

    # Check if config file exists
    if not os.path.exists(args_cli.config):
        raise FileNotFoundError(f"Config file not found: {args_cli.config}")
    
    print(f"[Config] Loading config from: {args_cli.config}")
    
    # Load config directly from YAML
    with open(args_cli.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        raise ValueError("Config file is empty or invalid YAML!")

    # Convert dictionary to Namespace object
    args = argparse.Namespace(**config)
    
    # Save config file path for logging
    args.config_file = args_cli.config
    
    # Ensure Device is correct (Check only if device is defined in YAML)
    if hasattr(args, 'device'):
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[Config] CUDA not available, switching to CPU.")
            args.device = "cpu"
    else:
        # If device is not in YAML, warn user (do not set default)
        print("[Config Warning] 'device' parameter not found in YAML. Relying on PyTorch defaults.")

    # Policy Type must be defined (Used to distinguish which Trainer to load later)
    if is_train:
        if not hasattr(args, 'policy_type') or args.policy_type is None:
            print("\n" + "!"*50)
            print("[Config Error] 'policy_type' is NOT defined in YAML!")
            print("Please set 'policy_type: on-policy' or 'policy_type: off-policy' in your YAML file.")
            print("!"*50 + "\n")
            raise ValueError("policy_type is required")

    return args

def load_algorithm(algo_path):
    """
    Dynamically load an algorithm class from a string path.

    Args:
        algo_path (str): Dot-separated path to the class (e.g., "algorithms.vdn.VDN").

    Returns:
        class: The imported algorithm class.
    """
    import importlib
    try:
        module_name, class_name = algo_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not load algorithm from {algo_path}. Error: {e}")

if __name__ == "__main__":
    print("--- Testing Config Logic (YAML Strict Mode) ---")
    
    # For testing, we need to create a temporary YAML file
    dummy_yaml = "test_config_temp.yaml"
    with open(dummy_yaml, "w") as f:
        f.write("algo: algorithms.vdn.VDN\n")
        f.write("seeds: [1, 42, 100]\n") # Test list parsing
        f.write("policy_type: off-policy\n")
        f.write("device: cuda\n")
        f.write("rnn_hidden_dim: 128\n")
    
    try:
        # Simulate command line input: python config.py --config test_config_temp.yaml
        sys.argv = ['config.py', '--config', dummy_yaml]
        args = get_config()
        
        print(f"Loaded Seeds: {args.seeds} (Type: {type(args.seeds)})")
        print(f"Loaded Algo: {args.algo}")
        print(f"Loaded Policy Type: {args.policy_type}")
        print(f"Loaded Hidden Dim: {args.rnn_hidden_dim}")
        print(f"Device Check: {args.device}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(dummy_yaml):
            os.remove(dummy_yaml)

    print("\nConfig test finished.")