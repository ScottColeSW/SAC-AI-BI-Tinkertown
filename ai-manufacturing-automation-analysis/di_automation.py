# # Gridworld environment 
# Create gridworld with rewards, hazards
import numpy as np
import matplotlib.pyplot as plt

def generate_rewards(num_rewards):
  # Generate rewards
  return reward_locs

def generate_hazards(num_hazards):
  # Generate hazards
  return hazard_locs

# Grid size
grid_size = 10 
# Get grid size from user
#grid_size = int(input("Enter grid size (2-25): "))
grid_size = min(max(grid_size, 2), 25)

# Create empty grid
grid = np.zeros((grid_size,grid_size))

# # Add reward states
# grid[1,2] = 1  
# grid[3,grid_size-1] = 2

# # Add hazard states 
# grid[2,1] = -1 
# grid[grid_size-1,3] = -1

# # Add random rewards
num_rewards = grid_size // 2
#Tie rewards to agent performance - e.g. 
#num_rewards = 10 + int(0.1*agent_reward)
num_rewards = np.gradient(2, grid_size//2)
#. More rewards if agent is doing well.
# #reward_locs = np.random.randint((grid_size,grid_size), size=num_rewards)
# reward_locs = np.random.randint(0, grid_size, size=num_rewards)
# grid[reward_locs] = 1

# # Add random hazards
num_hazards = np.gradient(2, grid_size//2)
#grid_size // 4
# #hazard_locs = np.random.randint((grid_size,grid_size), size=num_hazards) 
# hazard_locs = np.random.randint(0, grid_size, size=num_hazards)
# grid[hazard_locs] = -1
# Reward locations
# reward_x = np.random.randint(0, grid_size, size=num_rewards)
# reward_y = np.random.randint(0, grid_size, size=num_rewards)
# reward_locs = list(zip(reward_x, reward_y))
reward_x = np.random.randint(0, grid_size, size=num_rewards)
reward_x = reward_x.reshape(-1) 

reward_y = np.random.randint(0, grid_size, size=num_rewards) 
reward_y = reward_y.reshape(-1)

assert reward_x.ndim >= 1 and reward_y.ndim >= 1
reward_locs = list(zip(reward_x, reward_y))

# Hazard locations 
hazard_x = np.random.randint(0, grid_size, size=num_hazards)
hazard_y = np.random.randint(0, grid_size, size=num_hazards)
hazard_locs = list(zip(hazard_x, hazard_y))

for x, y in reward_locs:
  grid[x, y] = 1

for x, y in hazard_locs:
  grid[x, y] = -1

# Start and goal locations
start = [0,0]  
goal = [grid_size-1,grid_size-1]


print("Grid size:", grid_size)
print("Number of rewards:", num_rewards)
print("Number of hazards:", num_hazards)

# Render visualization
plt.matshow(grid)
plt.tight_layout()
plt.show()

# # AI agent
# Create agent with epsilon-greedy policy

# # Training loop
# for episode in episodes:
  
#   # Human guidance  
#   Get high-level human guidance
  
#   # Safe exploration
#   Agent takes exploratory actions based on epsilon-greedy policy and human guidance
  
#   # Remember experiences
#   Record state transitions, rewards
  
#   # Explain decisions    
#   Use LIME to explain agent's actions to human
#   Get human feedback
  
#   # Policy optimization
#   Train reinforcement learning model on experiences
#   Update agent policy
  
#   # Causality modeling
#   Add experiences to dataset
#   Periodically fit causal Bayesian network on dataset
  
#   # Human collaboration
#   Provide agent guidance based on model explanations and observed behavior
  
# # Interactive evaluation
# Visualize agent traversing gridworld aided by human
# Show causal model, training progress, explanation examples