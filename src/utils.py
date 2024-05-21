import pickle
import time

import matplotlib.pyplot as plt
import numpy as np


def save_pickle(q_table, object_name):
    with open(object_name, "wb") as f:  # Use "wb" for writing binary data
        pickle.dump(q_table, f)
    print('pickle file saved!!!!')

def load_pickle(object_name):
    with open(object_name, "rb") as f:
        deserialized_dict = pickle.load(f)
    return deserialized_dict

def run_learned_policy(env, agent):
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    print("Initial state: {0};".format(obs.reshape((4, 5))))
    
    total_reward = 0
    steps = 0
    
    # Continue through the episode until we reach a termination state
    while not terminated:
        # Agent decides on what action to choose
        action = np.argmax(agent.q_table[obs,:])
        
        # Mapping action number to action
        action_names = ['Down', 'Up', 'Right', 'Left']
        action_took = action_names[action]
        print("Agent opts to take the following action: {0}".format(action_took))
        
        # Environment performs taking an action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        print("New Observation: {0}; Immediate Reward: {1}, Termination Status: {2}, Termination Message: {3}".format(obs.reshape((4, 5)), reward, 
                                                                                                                        terminated, info['Termination Message']))
        time.sleep(1)
        print('**************')
        steps += 1
    print("Total Reward Collected Over the Episode: {0} in Steps: {1}".format(total_reward, steps))

def run_learned_policy_supressed_printing(env, agent):
    obs, _ = env.reset()
    terminated, truncated = False, False

    
    total_reward = 0
    steps = 0
    
    # Continue through the episode until we reach a termination state
    while not terminated:
        # Agent decides on what action to choose
        action = np.argmax(agent.q_table[obs,:])
        
        # Mapping action number to action
        action_names = ['Down', 'Up', 'Right', 'Left']
        action_took = action_names[action]
        
        # Environment performs taking an action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
    return total_reward





def plot_grid(env, agent, reward_across_episodes: list, epsilons_across_episodes: list) -> None:
    """Plot main and extra plots in a 4x4 grid."""
    
    env.train = False
    total_reward_learned_policy = []
    for i in range(30):
        total_reward_learned_policy.append(run_learned_policy_supressed_printing(env,agent))
        
    plt.figure(figsize=(15, 10))

    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(reward_across_episodes, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward Value')
    plt.title('Rewards Per Episode (Training)')
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(total_reward_learned_policy, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward Value')
    plt.title('Rewards Per Episode (Learned Policy Evaluation)')
    plt.grid()

    # Extra plots
    plt.subplot(2, 2, 3)
    plt.plot(reward_across_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Reward Per Episode (Training)')
    plt.title('Cummulative Reward vs Episode')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(epsilons_across_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Values')
    plt.title('Epsilon Decay')
    plt.grid()

    plt.tight_layout()
    plt.show()
