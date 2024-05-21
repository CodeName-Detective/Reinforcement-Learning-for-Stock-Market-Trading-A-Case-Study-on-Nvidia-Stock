import numpy as np


class QLearningGreedyAgent:
  """
  Define an RL agent which follows an epsilon greedy algorithm throughout the training.
  It uses Q-Learning algorithm to update Q-values.
  """
  def __init__(self: 'QLearningGreedyAgent', env, learning_rate: 'float', discount_factor: 'float') -> None:
    """Initializing Epsilon Greedy Agent

    Args:
        env (gymnasium.Env): object of Grid Environment.
        learning_rate (float): Learning rate used in SARSA algorithm.
        discount_factor (float): Discount factor to quantify the importance of future reward.
    """
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    
    # Initiating the Q-table with all Zeros.
    self.q_table = np.zeros((self.observation_space.n, self.action_space.n))


  def step(self: 'QLearningGreedyAgent', state: int, epsilon: float) -> int:
    """Given the current state and probability of choosing a random action we will
    provide the action we should choose.

    Args:
        state (int): Current location of the agent in the environment.
        epsilon (float): Probability of taking random action.

    Returns:
        action (int): Action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
    """
    # Epsilon-Greedy Action Selection
    
    random_number = np.random.rand()

    if random_number <= epsilon:
        action = np.random.choice(a = self.action_space.n)
    else:
        action = np.argmax(self.q_table[state,:])
    
    return action
  
  def update_qvalue(self: 'QLearningGreedyAgent', current_state: int, current_action: int, 
                    reward: int, future_state: int) -> None:
    """Update the Q value based on the Q-Learning algorithm

    Args:
        current_state (int): Current State represented as integer
        current_action (int): Current action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
        reward (int): Immediate reward that was recieved after taking the current action.
        future_state (int): Future State represented as integer
    """
    # Q-Learning update Q table
    self.q_table[current_state,current_action] = self.q_table[current_state,current_action] + self.learning_rate*(reward + 
                                                                                  self.discount_factor*(np.max(self.q_table[future_state,:])) -
                                                                                  self.q_table[current_state,current_action]
                                                                                  )