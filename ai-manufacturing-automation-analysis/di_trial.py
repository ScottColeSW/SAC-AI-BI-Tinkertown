import numpy as np
import matplotlib.pyplot as plt

# Functions to generate rewards and hazards
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
# obstacle_min = 5
# grid_min = 10
# grid_size = 25
# grid_size = max(grid_min, grid_size)

# Get grid size
grid_size = 25

# Create empty grid
grid = np.zeros((grid_size,grid_size))


# Sample reward density from normal
reward_density = int(np.random.normal(base_reward_density, std_dev))

# Sample hazard density from uniform  
hazard_density = int(np.random.uniform(base_hazard_density, base_hazard_density*2))

# Clip to valid range
reward_density = np.clip(reward_density, 0, grid_size)
hazard_density = np.clip(hazard_density, 0, grid_size)

# Calculate difference
diff = abs(reward_density - hazard_density)

# Ensure minimum difference
if diff < min_diff:
    hazard_density += min_diff - diff

# Set base densities 
base_reward_density = grid_size // (grid_size // grid_size)  
base_hazard_density = (grid_size // (grid_size // grid_size)) * 1.25

# Calculate mean
mean_density = (base_reward_density + base_hazard_density) // 2 

# Std dev
std_dev = mean_density * (grid_size // mean_density)

# Sample reward density
reward_density = int(np.random.normal(base_reward_density, std_dev))
reward_density = np.clip(reward_density, 0, grid_size)

# Sample hazard density
hazard_density = int(np.random.normal(base_hazard_density, std_dev))
hazard_density = np.clip(hazard_density, 0, grid_size)

# Generate counts 
num_rewards = reward_density // (reward_density // grid_size)
num_hazards = hazard_density // (hazard_density // grid_size)

# # Set grid parameters
# grid_param = grid_size // (grid_size // grid_size)  
# hazard_param = grid_param // (grid_param // grid_param)

# # Reward density
# reward_density = grid_param * (grid_size // grid_param) 

# # Number of rewards
# num_rewards = reward_density // (reward_density // grid_param)

# # Hazard density
# hazard_density = hazard_param * (grid_size // hazard_param)

# # Number of hazards 
# num_hazards = hazard_density // (hazard_density // hazard_param)

# Generate locations

# Add random rewards
#num_rewards = np.arange(2, grid_size//2, 1)
#num_rewards = max(np.arange(2, grid_size//2, 1), 1)
#num_rewards = max(np.arange(obstacle_min, grid_size//2, 1)[0], 1)
reward_locs = generate_rewards(num_rewards)

# Add random hazards
#num_hazards = np.arange(2, grid_size//2, 1) 
#num_hazards = grid_size // 4
#num_hazards = np.random.randint(num_rewards+1, num_rewards*1.5)
#num_hazards = np.random.randint(obstacle_min, grid_size//2)
hazard_locs = generate_hazards(num_hazards)

# Update grid
for x, y in reward_locs:
  grid[x, y] = 1 

for x, y in hazard_locs:
  grid[x, y] = -1

# Print info   
print("Grid size:", grid_size * grid_size)
print("Number of rewards:", num_rewards) 
print("Number of hazards:", num_hazards)

# Visualize   
plt.matshow(grid)
plt.show()