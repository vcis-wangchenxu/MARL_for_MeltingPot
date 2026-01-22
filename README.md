# MARL for Melting Pot

An implementation of Multi-Agent Reinforcement Learning (MARL) algorithms, specifically **MAPPO (Multi-Agent PPO)** and **VDN (Value Decomposition Networks)**, designed for the [DeepMind Melting Pot](https://github.com/google-deepmind/meltingpot) benchmark.

## 🚀 Features

*   **Algorithms**: 
    *   **MAPPO** (On-Policy): Cooperative PPO with Centralized Value Function.
    *   **VDN** (Off-Policy): Value Decomposition Networks independent Q-learning with shared rewards.
*   **Environment**: Full support for DeepMind Melting Pot substrates (e.g., `clean_up`, `collaborative_cooking`).
*   **Vectorization**: Efficient `MeltingPotAsyncVectorEnv` for parallel environment rollouts.
*   **Architectures**: 
    *   **CNN + RNN**: Handles partial observability and image inputs.
    *   **PopArt / Normalization**: Input normalization for stable training.
*   **Logging**: Integrated with **[SwanLab](https://swanlab.cn)** for experiment tracking and visualization.

## 📂 Project Structure

```
marl_for_meltingpot/
├── algorithms/             # Algorithm implementations
│   ├── mappo.py            # MAPPO (PPO with Centralized Critic)
│   └── vdn.py              # VDN (Value Decomposition Networks)
├── configs/                # Configuration files
│   ├── mappo_meltingpot.yaml
│   └── vdn_meltingpot.yaml
├── envs/                   # Environment wrappers
│   ├── MeltingPotWrapper.py   # Gym-like wrapper for Melting Pot
│   └── multi_envs.py          # Multiprocessing VectorEnv
├── memories/               # Experience Replay Buffers
│   ├── ReplayBuffer.py     # For Off-Policy (VDN)
│   └── RolloutBuffer.py    # For On-Policy (MAPPO)
├── networks/               # Neural Network Architectures
│   ├── DRQN.py             # Recurrent Q-Network
│   └── MAPPO_Network.py    # Actor-Critic Networks
├── results/                # Training outputs (models, logs)
├── utils/                  # Utility functions
│   ├── config.py           # Config loader
│   ├── evaluator.py        # Evaluation logic
│   └── util.py             # Seeding and plotting
├── run.py                  # Main entry point for training
├── train_offpolicy.py      # Training loop for Off-Policy
└── train_onpolicy.py       # Training loop for On-Policy
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/marl_for_meltingpot.git
    cd marl_for_meltingpot
    ```

2.  **Install Dependencies:**
    Ensure you have Python 3.8+ and PyTorch installed. You also need DeepMind Melting Pot.
    ```bash
    pip install torch numpy pygame pyyaml tqdm pandas matplotlib seaborn dm-meltingpot
    pip install swanlab  # For logging
    ```

## 🏃 Usage

### Training

You can train agents using the `run.py` script. You must specify a configuration file using the `--config` argument.

**Train MAPPO (On-Policy):**
```bash
python run.py --config configs/mappo_meltingpot.yaml
```

**Train VDN (Off-Policy):**
```bash
python run.py --config configs/vdn_meltingpot.yaml
```

### Configuration (`.yaml`)
Modify the files in `configs/` to adjust hyperparameters.
Key parameters:
*   `env`: The Melting Pot substrate name (e.g., `clean_up`).
*   `algo`: Path to algorithm class (e.g., `algorithms.mappo.MAPPO`).
*   `num_envs`: Number of parallel environments for data collection.
*   `share_parameters`: Whether agents share weights (True/False).

### Evaluation
Evaluation runs automatically during training based on the `eval_freq` parameter in the config.
*   Models are saved in `results/<experiment_name>/models/`.
*   Best models are saved as `model_best.pth`.

## 📊 Logging & Visualization

This project uses **SwanLab** for logging metrics (Reward, Loss, Episode Length).
*   Logs are saved in `results/logs/`.
*   You can view training curves in the cloud if SwanLab is configured, or locally.

## 🤖 Algorithms Details

*   **MAPPO**: Implements PPO with a centralized value function. It uses a CNN encoder for visual observations and an optional vector encoder for other data. It supports Recurrent Neural Networks (GRU) to handle memory.
*   **VDN**: Implements Value Decomposition Networks. It approximates the joint Q-value as the sum of individual local Q-values. It uses a DRQN (Deep Recurrent Q-Network) architecture.

## 📝 Notes
*   **Global State**: The environment wrapper automatically attempts to extract `WORLD.RGB` for centralized critics if available in the substrate.
*   **Vector Observations**: The wrapper automatically flattens and concatenates all vector observations defined in the Melting Pot spec, excluding RGB.

## 📄 License
MIT License
