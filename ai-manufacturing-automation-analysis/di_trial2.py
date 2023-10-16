import numpy as np
import matplotlib.pyplot as plt

def generate_rewards(num_rewards):

  reward_x = np.random.randint(0, grid_size, size=num_rewards)
  reward_x = reward_x.reshape(-1)
  

  reward_y = np.random.randint(0, grid_size, size=num_rewards)
  reward_y = reward_y.reshape(-1)
  assert reward_x.ndim >= 1 and reward_y.ndim >= 1

  reward_locs = list(zip(reward_x, reward_y))
  
  return reward_locs

def generate_hazards(num_hazards):
  hazard_x = np.random.randint(0, grid_size, size=num_hazards)
  hazard_x = hazard_x.reshape(-1)

  hazard_y = np.random.randint(0, grid_size, size=num_hazards)
  hazard_y = hazard_y.reshape(-1)
  assert hazard_x.ndim >= 1 and hazard_y.ndim >= 1

  hazard_locs = list(zip(hazard_x, hazard_y))
  return hazard_locs

# Grid size
grid_size = 25  

# Base densities
base_reward_density = grid_size // (grid_size // grid_size)
base_hazard_density = base_reward_density // 2

# Std dev 
std_dev = grid_size // 10

# Sample reward density
reward_density = int(np.random.normal(base_reward_density, std_dev))
reward_density = np.clip(reward_density, 0, grid_size)

# Sample hazard density
hazard_density = int(np.random.uniform(base_hazard_density, base_hazard_density*2))
hazard_density = np.clip(hazard_density, 0, grid_size)

# Minimum difference
min_diff = 5
diff = abs(reward_density - hazard_density)
if diff < min_diff:
    hazard_density += min_diff - diff
    
# Generate counts
num_rewards = reward_density // (reward_density // grid_size)
num_hazards = hazard_density // (hazard_density // grid_size) 

# Create grid
grid = np.zeros((grid_size, grid_size))

# Generate locations
reward_locs = generate_rewards(num_rewards)
hazard_locs = generate_hazards(num_hazards)

# Update grid
for x, y in reward_locs:
    grid[x, y] = 1
for x, y in hazard_locs:
    grid[x, y] = -1
    
# Print info
print("Grid size:", grid_size*grid_size) 
print("Number of rewards:", num_rewards)
print("Number of hazards:", num_hazards) 

# Visualize
plt.matshow(grid)
plt.show()