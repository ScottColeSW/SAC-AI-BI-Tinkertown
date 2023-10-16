import random
import cupy as cp
# Get CUDA up and running
#TODO: refactor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Specify the GPU device index to use (e.g., '0' for the first GPU)
import tensorflow as tf
print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
print()

# Maze must have an ODD number of rows and columns.
# Walls go on EVEN rows/columns.
# Openings go on ODD rows/columns
MAZE_HEIGHT = 21
MAZE_WIDTH = 21

def create_empty_grid(width, height, default_value=0):
    """ Create an empty grid. """
    grid = []
    for row in range(height):
        grid.append([])
        for column in range(width):
            grid[row].append(default_value)
    return grid


def create_outside_walls(maze):
    """ Create outside border walls."""
    # Create left and right walls
    for row in range(len(maze)):
        maze[row][0] = -1
        maze[row][len(maze[row]) - 1] = -1

    # Create top and bottom walls
    for column in range(1, len(maze[0]) - 1):
        maze[0][column] = -1
        maze[len(maze) - 1][column] = -1


def make_maze_recursive_call(maze, top, bottom, left, right):
    """
    Recursive function to divide up the maze in four sections
    and create three gaps.
    Walls can only go on even numbered rows/columns.
    Gaps can only go on odd numbered rows/columns.
    Maze must have an ODD number of rows and columns.
    """
    # Figure out where to divide horizontally
    start_range = bottom + 2
    end_range = top - 1
    y = random.randrange(start_range, end_range, 2)

    # Do the division
    for column in range(left + 1, right):
        maze[y][column] = -1

    # Figure out where to divide vertically
    start_range = left + 2
    end_range = right - 1
    x = random.randrange(start_range, end_range, 2)

    # Do the division
    for row in range(bottom + 1, top):
        maze[row][x] = -1

    # Now we'll make a gap on 3 of the 4 walls.
    # Figure out which wall does NOT get a gap.
    wall = random.randrange(4)
    if wall != 0:
        gap = random.randrange(left + 1, x, 2)
        maze[y][gap] = 0

    if wall != 1:
        gap = random.randrange(x + 1, right, 2)
        maze[y][gap] = 0

    if wall != 2:
        gap = random.randrange(bottom + 1, y, 2)
        maze[gap][x] = 0

    if wall != 3:
        gap = random.randrange(y + 1, top, 2)
        maze[gap][x] = 0

    # If there's enough space, to a recursive call.
    if top > y + 3 and x > left + 3:
        make_maze_recursive_call(maze, top, y, left, x)

    if top > y + 3 and x + 3 < right:
        make_maze_recursive_call(maze, top, y, x, right)

    if bottom + 3 < y and x + 3 < right:
        make_maze_recursive_call(maze, y, bottom, x, right)

    if bottom + 3 < y and x > left + 3:
        make_maze_recursive_call(maze, y, bottom, left, x)


def make_maze():
    # Create an empty grid
    maze = create_empty_grid(MAZE_WIDTH, MAZE_HEIGHT)

    # Create outside border walls
    create_outside_walls(maze)

    # Call the recursive function to generate the maze
    make_maze_recursive_call(maze, MAZE_HEIGHT - 1, 0, 0, MAZE_WIDTH - 1)

    return maze


def print_maze(maze):
    """ Print the maze grid. """
    for row in maze:
        for cell in row:
            if cell == -1:
                print("W", end=" ")  # Wall
            elif cell == 0:
                print(" ", end=" ")  # Passage
            elif cell == "P" or cell == "F" or cell == "G" or cell == "I" or cell == "E":
                print(f"{cell}", end=" ")
            else:
                print("#", end=" ")  # Unvisited cell
        print()

    
def clean_maze(maze_data):
    element_labels = {
        "-1": -1,
        "0": 0,
        "E": 1,
        "I": 2,
        "F": 3,
        "G": 5,
        "P": 42
    }

    cleaned_maze = []

    for row in maze_data:
        cleaned_row = []
        for cell in row:
            if str(cell) in element_labels:
                cleaned_row.append(element_labels[str(cell)])
            else:
                cleaned_row.append(-99)

        cleaned_maze.append(cleaned_row)

    return cleaned_maze


# Generate the maze
maze = make_maze()

# Print the maze
print("Pretty print object free 'maze': ")
print_maze(maze)

def plant_objects(maze_data):
    # Get the dimensions of the maze
    height = len(maze_data)
    width = len(maze_data[0])
    print(f"max map size: {height * width}")

    # Plant the player at a random location
    player_x = random.randint(1, width - 1)
    player_y = random.randint(1, height - 1)
    maze_data[player_y][player_x] = "P"

    # Plant items at random locations
    num_items = 75
    while range(num_items):
        item_x = random.randint(0, width - 1)
        item_y = random.randint(0, height - 1)
        # Make sure the item doesn't overlap with existing objects
        if maze_data[item_y][item_x] == 0:
            maze_data[item_y][item_x] = "I"
            num_items = num_items-1

    # Plant enemies at random locations
    num_enemies = 100
    for _ in range(num_enemies):
        enemy_x = random.randint(0, width - 1)
        enemy_y = random.randint(0, height - 1)
        if maze_data[enemy_y][enemy_x] == 0:
            maze_data[enemy_y][enemy_x] = "E"

    # Plant friends at random locations
    num_friends = 10
    for _ in range(num_friends):
        friend_x = random.randint(0, width - 1)
        friend_y = random.randint(0, height - 1)
        if maze_data[friend_y][friend_x] == 0:
            maze_data[friend_y][friend_x] = "F"

    # Plant the goal at a random location
    goal_unset = True
    while goal_unset:
        goal_x = random.randint(0, width - 1)
        goal_y = random.randint(0, height - 1)
        if maze_data[goal_y][goal_x] == 0:
            maze_data[goal_y][goal_x] = "G"
            goal_unset = False

    # Add the actions attribute to each state in the maze data
    # for state in maze_data:
    #     state.actions = maze.get_valid_actions(state)

    return maze_data


# # Generate the maze data
# maze_data = make_maze()
# print(maze_data)

# Plant the objects in the maze
maze_data = plant_objects(maze)

# Print the modified maze
print(f"First Maze Data: {maze_data}")
print_maze(maze_data)
print("::and then cleaned now below...")
maze_data = clean_maze(maze_data)
print_maze(maze_data)

print("::now learn...")




from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   


def q_learning(maze_data, num_episodes, alpha, gamma, epsilon):
    # Extract necessary information from maze data
    maze_size = len(maze_data)
    num_actions = 4  # Assuming 4 possible actions: up, down, left, right

    # Define Q-table
    q_table = cp.zeros((maze_size, maze_size, num_actions))

    for episode in range(num_episodes):
        state = reset_maze(maze_data)  # Reset the maze to start a new episode
        episode_done = False

        # Print feedback after each episode
        if episode % 10 == 1 or episode == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")
        while not episode_done:

            # Choose action using epsilon-greedy policy
            if cp.random.uniform(0, 1) < epsilon:
                action = cp.random.randint(num_actions)
            else:
                action = cp.argmax(q_table[state])

            # Perform action and observe next state and reward
            next_state, reward, done = step_maze(state, action, maze_data)

            # Update Q-table using the Q-learning update equation
            q_table[state][action] += alpha * (reward + gamma * cp.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            if done:
                episode_done = True

    print("Training completed!")
    return q_table

def reset_maze(maze_data):
    maze_size = len(maze_data)
    for i in range(maze_size):
        for j in range(maze_size):
            if maze_data[i][j] == 5:  # Find the starting position (assuming value 5 represents the starting point)
                return (i, j)  # Return the coordinates as a tuple

    return None  # Return None if the starting position is not found


def step_maze(state, action, maze_data):
    # Perform the action and observe the next state and reward
    x, y = state
    next_x, next_y = x, y  # Initialize the next state with the current state

    if action == 0:  # Up
        next_x -= 1
    elif action == 1:  # Down
        next_x += 1
    elif action == 2:  # Left
        next_y -= 1
    elif action == 3:  # Right
        next_y += 1

    # Check if the next state is valid or an obstacle
    maze_size = len(maze_data)
    if 0 <= next_x < maze_size and 0 <= next_y < maze_size and maze_data[next_x][next_y] != -1:
        reward = maze_data[next_x][next_y]
        state = (next_x, next_y)

        # Check if the goal state is reached
        if reward == 5:
            done = True
        else:
            done = False
    else:
        reward = -1
        done = False

    return state, reward, done

def visualize_maze(maze_data, state):
    # Define the characters to represent different elements in the maze
    CHAR_MAP = {
        -1: '#',  # Walls
        0: ' ',   # Empty space
        1: 'S',   # Start
        2: 'G',   # Goal
        5: '*',   # Agent
    }

    # Create a copy of the maze data to avoid modifying the original
    maze = maze_data.copy()

    # Set the character representing the agent in the current state
    x, y = state
    maze[x][y] = 5

    # Print the maze
    for row in maze:
        print(''.join(CHAR_MAP[cell] for cell in row))

    # Add some separation between mazes
    print('-' * 20)


# def navigate_maze(q_values, maze_data):
#     start_state = find_start_state(maze_data)  # Find the start state in the maze
#     current_state = start_state
#     path = [current_state]  # Track the path taken in the maze

#     while not is_goal_state(current_state, maze_data):
#         action = select_best_action(q_values, current_state)
#         next_state = get_next_state(current_state, action)
#         path.append(next_state)
#         current_state = next_state

#     return path


def navigate_maze(q_values, maze_data):
    maze_size = len(maze_data)
    num_actions = 4  # Assuming 4 possible actions: up, down, left, right
    state = reset_maze(maze_data)  # Reset the maze to the starting state
    done = False

    # Visualize the initial state
    visualize_maze(maze_data, state)

    while not done:
        action = cp.argmax(q_values[state])  # Choose the action with the highest Q-value
        next_state, reward, done = step_maze(state, action, maze_data)  # Perform the action
        state = next_state  # Update the current state

        # Visualize the updated state
        visualize_maze(maze_data, state)

    if reward == 1:
        print("Goal reached!")
    else:
        print("Agent failed to reach the goal.")

    return

import pickle

def save_q_values(q_values, filename):
    with open(filename, 'wb') as file:
        pickle.dump(q_values, file)

def load_q_values(filename):
    with open(filename, 'rb') as file:
        q_values = pickle.load(file)
    return q_values

import torch
# Get CUDA up and running
#TODO: refactor
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

num_episodes = 50 # originally 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

print(maze_data)
print()

q_values = q_learning(maze_data, num_episodes, alpha, gamma, epsilon)
print(f"q_values: {q_values}")

i=1
save_q_values(q_values, f'q_values{i}.pkl')

loaded_q_values = load_q_values('q_values{i}.pkl')


navigate_maze(q_values, maze_data)


import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def dqn(maze, num_episodes, batch_size, gamma, epsilon, lr):
    # Initialize DQN
    dqn = DQN(maze.size, len(maze.actions))
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay_buffer = []

    for episode in range(num_episodes):
        state = maze.reset()  # Reset the maze to start a new episode
        done = False
        
        while not done:
            # Choose action using epsilon-greedy policy
            if cp.random.uniform(0, 1) < epsilon:
                action = cp.random.choice(maze.actions)
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            # Perform action and observe next state and reward
            next_state, reward, done = maze.step(action)

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Update DQN
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                q_values = dqn(torch.tensor(states, dtype=torch.float32))
                next_q_values = dqn(torch.tensor(next_states, dtype=torch.float32))
                targets = q_values.clone().detach()

                for i in range(batch_size):
                    targets[i][actions[i]] = rewards[i] + gamma * torch.max(next_q_values[i]) * (1 - dones[i])

                optimizer.zero_grad()
                loss = criterion(q_values, targets)
                loss.backward()
                optimizer.step()

            state = next_state

    return dqn


import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


def calculate_returns(episode_rewards, gamma):
    returns = []
    G = 0
    for reward in reversed(episode_rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns



def policy_gradient(maze, num_episodes, batch_size, gamma, lr):
    # Initialize Policy Network
    policy_net = PolicyNetwork(maze.size, len(maze.actions))
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    episode_rewards = []

    for episode in range(num_episodes):
        state = maze.reset()  # Reset the maze to start a new episode
        episode_reward = 0
        episode_log_probs = []
        
        while not maze.is_terminal_state(state):
            # Get action probabilities from the policy network
            with torch.no_grad():
                action_probs = policy_net(torch.tensor(state, dtype=torch.float32))
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Perform action and observe next state and reward
            next_state, reward, _ = maze.step(action.item())

            episode_reward += reward
            episode_log_probs.append(log_prob)

            state = next_state

        episode_rewards.append(episode_reward)

        # Calculate returns and update policy network
        returns = calculate_returns(episode_rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)

        loss = torch.zeros(1)
        for log_prob, G in zip(episode_log_probs, returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy_net