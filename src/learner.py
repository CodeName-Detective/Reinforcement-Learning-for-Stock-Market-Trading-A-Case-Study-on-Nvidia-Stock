import numpy as np

from src.agent import QLearningGreedyAgent


def q_learning_learning_loop(env,learning_rate: float, discount_factor: float, episodes: int,
                        min_epsilon_allowed: float, initial_epsilon_value: float) -> tuple:
  """Learning loop train Agent to reach GOAL state in the environment using Q-Learning Algorithm.

  Args:
      env (gymnasium.Env): object of Grid Environment.
      learning_rate (float): Learning rate used in SARSA algorithm
      discount_factor (float): Discount factor to quantify the importance of future reward.
      episodes (int): Number of episodes we should train.
      min_epsilon_allowed (float): Minimum epsilon that we should reach by the end of the training.
      initial_epsilon_value (float): Initial epsilon that we should use while starting the learning. 

  Returns:
      tuple[QLearningGreedyAgent, list, list]: Returns a tuple containing agent,
                                                  cumulative rewards across episodes,
                                                  epsilon used across episodes respectively.
  """
  
  # Agent.
  agent = QLearningGreedyAgent(env, learning_rate=learning_rate, discount_factor = discount_factor)
  print("Initial Q-Table; {0}".format(agent.q_table))
  
  # Initiating Number of episodes, epsilon values.
  episodes = episodes

  epsilon = initial_epsilon_value
  min_epsilon_allowed = min_epsilon_allowed
  
  # Calculating Epsilon Decay factor. 
  epsilon_decay_factor = np.power(min_epsilon_allowed/epsilon, 1/episodes)
  
  # Initiating list to store rewards and epsilons we use across episodes
  reward_across_episodes = []
  epsilons_across_episodes = []
  
  # Iterating over Episodes.
  for _ in range(episodes):
    # Resetting the environment.
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    # Fectcing Current State and Current Action details.
    current_state = obs
    current_action = agent.step(current_state, epsilon)
    
    reward_per_episode = 0
    epsilons_across_episodes.append(epsilon)
    
    # Iterating over an epsidoe untill termination status is reached.
    while not terminated:
      # Taking one step in the environment.
      obs, reward, terminated, truncated, _ = env.step(current_action)
      
      # Calculating cummulative reward for an Epoch.
      reward_per_episode = reward_per_episode+reward
      
      # Fetching future state and future reward.
      future_state = obs
      future_action = agent.step(future_state, epsilon)
      
      # Updating Q values.
      agent.update_qvalue(current_state, current_action, reward, future_state)

      current_state = future_state

      current_action = future_action
    
    # Decaying Epsilon
    epsilon = epsilon_decay_factor*epsilon
    reward_across_episodes.append(reward_per_episode)
  print('\n')
  print("Trained Q-Table; {0}".format(agent.q_table))

  return agent, reward_across_episodes, epsilons_across_episodes