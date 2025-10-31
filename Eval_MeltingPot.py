import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import torch
import pygame
import time
import os
from typing import Dict

from MeltingPotWrapper import build_meltingpot_env as GeneralMeltingPotWrapper
from models import AgentDRQN

MODEL_PATH = "models/QMIX-MeltingPot-1-20251031-165801.pth" 
MODEL_OBS_SHAPE = (3, 40, 40)   # (C, H, W)
MODEL_N_ACTIONS = 8             # 
MODEL_RNN_HIDDEN_DIM = 64       # Must match the trained model's hidden dim

ENVS_TO_TEST = [
    # Scenario 1: AI (Blue) vs. Bot (Red) (N=1)
    'collaborative_cooking__asymmetric_0',

    # Scenario 2: AI (Red) vs. Bot (Blue) (N=1)
    'collaborative_cooking__asymmetric_1',

    # Scenario 3: AI (Blue) vs. AI (Red) (N=2)
    # (Two agents controlled by the same AI model)
    'collaborative_cooking__asymmetric'
]

# Pygame parameters
PYGAME_SCALE = 10
FPS = 10

def run_visualization(env_name: str, 
                      env: GeneralMeltingPotWrapper, 
                      agent_net: AgentDRQN, 
                      device: torch.device):
    """
    Parameters:
        - env: An initialized wrapper (can be Scenario or Substrate)
        - agent_net: The AgentDRQN with loaded weights and set to .eval()
        - device: "cuda" or "cpu"
    """
    
    env_info = env.get_env_info()        # Environment information
    n_agents = env_info["n_agents"]      # Number of agents
    agent_ids = env_info["agent_ids"]    # List of agent IDs

    print(f"--- Starting visualization (N={n_agents} AI agents) ---")
    print(f"AI control: {agent_ids}")
    print(f"Press 'Q' to quit or close the window...")

    pygame.init()
    
    render_shape = env_info['render_shape']
    H, W, C = render_shape
    screen_size = (W * PYGAME_SCALE, H * PYGAME_SCALE)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(f"Evaluation: {env_name} (N={n_agents})")
    clock = pygame.time.Clock()

    obs_dict, state = env.reset()
    
    hidden_states_dict = {
        aid: np.zeros(MODEL_RNN_HIDDEN_DIM, dtype=np.float32) 
        for aid in agent_ids
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False

        frame = env.render()
        frame_transposed = frame.swapaxes(0, 1)
        surface = pygame.surfarray.make_surface(frame_transposed)
        scaled_surface = pygame.transform.scale(surface, screen_size)
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        obs_list = [obs_dict[aid]['RGB'] for aid in agent_ids]
        hidden_list = [hidden_states_dict[aid] for aid in agent_ids]
        
        obs_batch_np = np.stack(obs_list, axis=0)           # (N, C, H, W)
        hidden_batch_np = np.stack(hidden_list, axis=0)     # (N, hidden_dim)

        # (B=1, N, C, H, W)
        obs_tensor = torch.tensor(obs_batch_np, dtype=torch.float32).unsqueeze(0).to(device)
        # (B=1, N, hidden_dim)
        hidden_tensor = torch.tensor(hidden_batch_np, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values, next_hidden_tensor = agent_net(obs_tensor, hidden_tensor)

        actions_dict = {}
        for i, agent_id in enumerate(agent_ids):
            agent_q_values = q_values[0, i, :] 
            action = agent_q_values.argmax(dim=0).item() 
            actions_dict[agent_id] = action
        
        next_hidden_np = next_hidden_tensor.squeeze(0).cpu().numpy() # (N, hidden_dim)
        hidden_states_dict = {
            aid: next_hidden_np[i] for i, aid in enumerate(agent_ids)
        }
        
        next_obs_dict, next_state, rewards_dict, dones_dict, _ = env.step(actions_dict)

        obs_dict = next_obs_dict
        
        if any(dones_dict.values()):
            print("Episode finished, resetting...")
            obs_dict, state = env.reset()
            hidden_states_dict = {
                aid: np.zeros(MODEL_RNN_HIDDEN_DIM, dtype=np.float32) 
                for aid in agent_ids
            }

        clock.tick(FPS)

    pygame.quit()
    env.close()
    print("--- Visualization finished ---")


if __name__ == "__main__":
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"!!! Error: Model file not found !!!")
        print(f"Please check if MODEL_PATH points to: {MODEL_PATH}")
        exit()
    print(f"Successfully found model: {MODEL_PATH}")

    agent_net = AgentDRQN(MODEL_OBS_SHAPE, MODEL_N_ACTIONS, MODEL_RNN_HIDDEN_DIM)

    saved_state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    agent_state_dict = saved_state_dict['agent_network']
    
    agent_net.load_state_dict(agent_state_dict)
    agent_net.to(device)
    agent_net.eval() 

    print("Model weights loaded successfully.")

    for env_name in ENVS_TO_TEST:
        print("\n" + "="*50)
        print(f"Starting evaluation for environment: {env_name}")
        print("="*50)

        # Load environment
        env = GeneralMeltingPotWrapper(env_name=env_name)
        env_info = env.get_env_info()

        try:
            assert env_info['obs_shape'] == MODEL_OBS_SHAPE
            assert env_info['n_actions'] == MODEL_N_ACTIONS
        except AssertionError:
            print(f"!!! Error: Environment '{env_name}' is not compatible with the loaded model !!!")
            print(f"    Model expects Obs shape: {MODEL_OBS_SHAPE}, got: {env_info['obs_shape']}")
            print(f"    Model expects N-Actions: {MODEL_N_ACTIONS}, got: {env_info['n_actions']}")
            print(f"    Skipping this environment...")
            env.close()
            continue 

        print(f"Successfully loaded environment '{env_name}' (N={env_info['n_agents']}).")

        if "collaborative_cooking" in env_name:
             if env_info['n_agents'] == 1:
                idx = 0 if '_0' in env_name else 1
                color = "Blue" if idx == 0 else "Red"
                bot_color = "Red" if idx == 0 else "Blue"
                print(f"    [Mode: AI vs. Bot] AI control player_{idx} ({color}), Bot control ({bot_color})")
             elif env_info['n_agents'] == 2:
                print(f"    [Mode: AI vs. AI] AI control player_0 (Blue) and player_1 (Red)")

        try:
            run_visualization(env_name, env, agent_net, device)
        except Exception as e:
            print(f"\n!!! Runtime error occurred: {e}")
            import traceback
            traceback.print_exc()
            pygame.quit()
            env.close()
            
    print("\n" + "="*50)
    print("All evaluations completed.")

