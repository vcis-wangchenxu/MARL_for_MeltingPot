# 🤖 Make Handcrafting Great Again 🔥

##  🔬 Reduce reliance on libraries (RLlib, etc.) and make algorithms more transparent
This is a Multi-Agent Reinforcement Learning (MARL) algorithm library focused on providing clear, transparent, from-scratch implementations. We avoid heavy frameworks like RLib, ensuring all code is easy to read, modify, and understand.
---

All experiments are conducted on DeepMind's [MeltingPot](https://github.com/google-deepmind/meltingpot) environment.
---

## 🍲 Core Environment: MeltingPot
All our algorithms are experimented on MeltingPot. Before starting, please ensure you have correctly installed the environment.
### 🛠️ Installation Guide
1. Create a new Conda environment
    ```
    conda create -n marl_handcraft python=3.11 -y
    conda activate marl_handcraft
    ```
    >Note: ⚠️ Python 3.11 is required. MeltingPot is built on dm_lab, which has a strict requirement for this Python version.
2. Install the MeltingPot package From the root of the $meltingpot$ repository, run:
    ```
    pip install --editable .[dev]
    ```
3. Silence Warnings (Optional but recommended) 

    Due to interference between multiple dependencies, you may see numerous $absl$ warnings. To silence these, please use the files in the Eli_Warnings folder to replace the target files in your environment.

## 🖥️ Hardware Requirements
1. Devcontainer (x86 only)

    NOTE: The Devcontainer provided in this project only works for x86 platforms. Users on arm64 (e.g., M1/M2/M3 Macs) must follow the manual installation steps.

2. CUDA Support

    To enable CUDA support (required for GPU training), ensure you have the nvidia-container-toolkit package installed.


## 🚀 Usage
### Training
You can directly run the corresponding algorithm file.
```
# Example: Train QMIX
python QMIX_MeltingPot.py
```
> Note: Before training, please modify the env_name in the if __name__ == "__main__": block of the algorithm file (e.g., QMIX_MeltingPot.py).

### Evaluation (Visualization)
Run the evaluation file to compete against Bots or other AI agents.
```
# Example: Evaluate QMIX
python Eval_MeltingPot.py
```
> Note: 
    1. Before evaluating, please configure the MODEL_PATH (pointing to your trained model) and ENVS_TO_TEST (the list of environments you want to visualize) in the configuration section at the top of visualize.py;
    2. The evaluation script is universal; it can load both Scenario (AI vs. Bot) and Substrate (AI vs. AI) environments.

## ✅ Supported Algorithms & Environments
Our currently implemented and verified algorithms and environments are as follows:
<!-- | Algorithm | `collaborative_cooking__asymmetric` | `clean_up` |
| :--- | :---: | :---: |
| **QMIX** (Off-Policy) | ✔ | ✔ |
| **MAPPO** (On-Policy) | ✔ | (Pending) | -->
| Algorithm | Training | Eval | Original Paper Address | Paper Env |
| :--- | :---: | :---: |  :---: |  :---: | 
| **QMIX** (Off-Policy) | 'collaborative_cooking__asymmetric' | 'collaborative_cooking__asymmetric' | [Address](https://arxiv.org/abs/1803.11485) | SMAC |
| **VDN** (Off-Policy) | 'collaborative_cooking__asymmetric' | 'collaborative_cooking__asymmetric' | [Address](https://arxiv.org/abs/1706.05296) | Switch; Fetch; Checkers |

[![Star History Chart](https://api.star-history.com/svg?repos=vcis-wangchenxu/MARL_for_MeltingPot.git&type=date&legend=top-left)](https://www.star-history.com/#vcis-wangchenxu/MARL_for_MeltingPot.git&type=date&legend=top-left)