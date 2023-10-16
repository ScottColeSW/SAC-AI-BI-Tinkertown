import numpy as np
import random
import torch
import text_assets as ta

class Tile:
    def __init__(self, tile_type, thing_type, metadata, rewards=None):
        self.tile_type = tile_type
        self.thing_type = thing_type
        self.metadata = metadata
        self.rewards = rewards or {}

# Define the tile_rewards structure
tile_rewards = {
    "path": 0,
    "wall": -10,
    "puddle": -5,
    "mud": -7,
    "spring": 5,
    "teleport": 10,
    "trampoline": 8,
    "cloud": 3,
    "ladder": 2,
    "item": 20,
    "enemy": -15,
    "friend": 15
}

# Function to increase the player's score
def increase_score(score):
    # Code to increase the player's score
    pass

# Function to collect an item
def collect_item(item_type):
    # Code to handle collecting the item
    pass

# Function to decrease the player's score
def decrease_score(score):
    # Code to decrease the player's score
    pass

# Function to handle an enemy encounter
def handle_enemy_encounter(enemy_type):
    # Code to handle the enemy encounter
    pass

# Function to continue the game
def continue_game():
    # Code to continue the game
    pass

def get_current_tile(player_position):
    # Get the player's current position
    x, y = player_position

    # Define the size of the tilemap
    map_width = 10
    map_height = 10

    # Calculate the current tile indices
    tile_row = y
    tile_col = x

    # Create a Tile object representing the current tile
    current_tile = Tile(tile_type="path", thing_type=None, metadata="")

    # You can modify the logic here to assign the appropriate tile type, thing type, and metadata based on the current tile indices

    return current_tile

def get_player_choice():
    # TODO: Implement the logic to retrieve the player's choice
    # This could involve capturing input from the player, such as mouse clicks or key presses
    # and mapping it to the available choices in the BI mini-game
    player_choice = ...

    return player_choice


def get_correct_choice():
    # TODO: Implement the logic to determine the correct choice for the BI mini-game
    # This could involve using AI or Business Intelligence algorithms to predict the best path
    # and mapping it to the available choices in the BI mini-game
    correct_choice = ...

    return correct_choice

def show_correct_message():
    # TODO: Implement the logic to display a message when the player chooses the correct path
    # This could involve showing a message on the game screen or UI to inform the player
    print("Congratulations! You chose the correct path.")

    # Additional code to handle any specific actions or updates related to the correct choice
    # For example, you might update the player's score or trigger some in-game events

    # You can also add a delay or wait for player input before proceeding

    # Example delay using time.sleep()
    import time
    time.sleep(2)  # Sleep for 2 seconds before continuing

    # Example waiting for player input using input()
    #input("Press Enter to continue...")  # Wait for the player to press Enter

    # Optionally, you can return any relevant data from this function
    # For example, you might return a flag indicating the correct choice was made

    return True


def show_incorrect_message():
    print("Sorry, you chose the incorrect path.")

import json
def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data


# Function to retrieve the reward value based on the tile type
def get_reward(tile_type):
    return tile_rewards.get(tile_type, 0)

# Function to process the rewards and consequences of a tile during gameplay
def process_tile(tile):
    reward = get_reward(tile.tile_type)

    if reward > 0:
        increase_score(reward)
        collect_item(tile.thing_type)
    elif reward < 0:
        decrease_score(abs(reward))
        handle_enemy_encounter(tile.thing_type)
    else:
        continue_game()

    # Other game logic for the tile interaction

# Function to process the BI mini-game prediction and rewards
def process_prediction(player_choice, correct_choice):
    if player_choice == correct_choice:
        increase_score(50)  # High reward for correct prediction
        show_correct_message()
    else:
        decrease_score(20)  # Penalty for incorrect prediction
        show_incorrect_message()

    # Other BI mini-game logic

def handle_collision(self, tile_type):
    if tile_type in tile_rewards:
        reward = tile_rewards[tile_type]
        if reward > 0:
            self.increase_score(reward)
        elif reward < 0:
            self.decrease_score(abs(reward))
        if tile_type == "enemy":
            self.handle_enemy_encounter(tile_type)
        elif tile_type == "item":
            self.collect_item(tile_type)
        else:
            self.continue_game()
    else:
        print("Invalid tile type encountered.")

# def generate_level_data_grid(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with the specified width and height."""
#     level_data = torch.zeros((width, height), dtype=torch.int)

#     # Define element identifiers
#     player_start = 42
#     free_space_id = 0
#     enemy_id = 1
#     item_id = 2
#     friend_id = 3
#     goal_id = 4
#     difficulty_modifier = 1 if difficulty_level == "normal" else 2
#     level_data[0, 0] = player_start
#     level_data[width-1, height-1] = goal_id
    

#     # Generate the level data
#     for i in range(width):
#         for j in range(height):
#             # Assign elements based on percentages
#             if random.random() < empty_percentage:  # 80% free space
#                 level_data[i, j] = free_space_id
#             elif random.random() < enemy_probability * difficulty_modifier:  # 10% enemy
#                 level_data[i, j] = enemy_id
#             elif random.random() < item_probability:  # 5% item
#                 level_data[i, j] = item_id
#             elif random.random() < friend_probability:  # 3% friend
#                 level_data[i, j] = friend_id
#             elif random.random() < 1:  # 2% goal
#                 if goal_id not in level_data:
#                     level_data[i, j] = goal_id
#                 else:
#                     level_data[i, j] = item_id

#     # Assign free spaces to friends and enemies
#     for i in range(width):
#         for j in range(height):
#             if level_data[i, j] == friend_id or level_data[i, j] == enemy_id:
#                 level_data[i, j] = free_space_id

#     return level_data

# def generate_level_data_grid(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with the specified width and height."""
#     level_data = torch.zeros((width, height), dtype=torch.int)

#     # Define element identifiers
#     player_id = 42
#     free_space_id = 0
#     goal_id = 5
#     enemy_id = 1
#     item_id = 2
#     friend_id = 3

#     obstacle_ids = [enemy_id, item_id, friend_id]

#     # Generate the level data
#     for i in range(width):
#         for j in range(height):
#             # Assign elements based on percentages
#             if random.random() < empty_percentage:  # 80% free space
#                 level_data[i, j] = free_space_id
#             elif random.random() < enemy_probability:  # 10% enemy
#                 level_data[i, j] = enemy_id
#             elif random.random() < item_probability:  # 5% item
#                 level_data[i, j] = item_id
#             elif random.random() < friend_probability:  # 3% friend
#                 level_data[i, j] = friend_id

#     # Assign player and goal
#     player_position = (random.randint(0, width-1), random.randint(0, height-1))
#     level_data[player_position] = player_id

#     goal_position = (random.randint(0, width-1), random.randint(0, height-1))
#     while goal_position == player_position:
#         goal_position = (random.randint(0, width-1), random.randint(0, height-1))
#     level_data[goal_position] = goal_id

#     return level_data

def generate_level_data_grid(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
    """Generates a level data tensor with the specified width and height."""
    level_data = torch.zeros((width, height), dtype=torch.int)

    # Generate the level data
    for i in range(width):
        for j in range(height):
            # Assign elements based on percentages
            if random.random() < empty_percentage:  # 80% free space
                level_data[i, j] = free_space_id
            elif random.random() < enemy_probability:  # 10% enemy
                level_data[i, j] = enemy_id
            elif random.random() < item_probability:  # 5% item
                level_data[i, j] = item_id
            elif random.random() < friend_probability:  # 3% friend
                level_data[i, j] = friend_id

    # Assign player and goal
    while True:
        player_position = (random.randint(0, width-1), random.randint(0, height-1))
        if not is_next_to_enemy(level_data, player_position):
            break
    level_data[player_position] = player_id

    goal_position = (random.randint(0, width-1), random.randint(0, height-1))
    while goal_position == player_position:
        goal_position = (random.randint(0, width-1), random.randint(0, height-1))
    level_data[goal_position] = goal_id

    return level_data


def is_next_to_enemy(level_data, position):
    """Checks if the given position is next to an enemy."""
    x, y = position
    width, height = level_data.shape

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = x + dx
            ny = y + dy
            if nx >= 0 and nx < width and ny >= 0 and ny < height:
                if level_data[nx, ny] == enemy_id:
                    return True

    return False

# import torch
# import random

# def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with the specified width and height."""
#     level_data = torch.zeros((width, height), dtype=torch.int32)

#     # Define element identifiers
#     player_id = 42
#     wall_id = -1
#     goal_id = 5

#     # Assign wall elements to the outer boundary
#     for i in range(width):
#         level_data[i, 0] = wall_id
#         level_data[i, height-1] = wall_id

#     for j in range(height):
#         level_data[0, j] = wall_id
#         level_data[width-1, j] = wall_id

#     # Assign player and goal
#     player_position = find_farthest_wall_position(level_data)
#     level_data[player_position] = player_id

#     goal_position = find_farthest_wall_position(level_data)
#     level_data[goal_position] = goal_id

#     return level_data

# def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with the specified width and height."""
#     level_data = torch.full((width, height), -1, dtype=torch.int32)

#     # Define element identifiers
#     player_id = 42
#     goal_id = 5
#     wall_id = -1

#     # Build walls
#     level_data[:, 0] = wall_id  # Left wall
#     level_data[:, height - 1] = wall_id  # Right wall
#     level_data[0, :] = wall_id  # Top wall
#     level_data[width - 1, :] = wall_id  # Bottom wall

#     # Assign player and goal
#     level_data[0, 0] = player_id
#     level_data[width - 1, height - 1] = goal_id

#     # Calculate the number of elements to be placed
#     total_elements = (width - 2) * (height - 2) - 2
#     empty_elements = int(total_elements * empty_percentage)
#     enemy_elements = int(total_elements * enemy_probability)
#     item_elements = int(total_elements * item_probability)
#     friend_elements = int(total_elements * friend_probability)

#     # Generate the level data
#     element_counts = {
#         0: empty_elements,
#         1: enemy_elements,
#         2: item_elements,
#         3: friend_elements
#     }

#     for i in range(1, width - 1):
#         for j in range(1, height - 1):
#             if level_data[i, j] == -1:
#                 element_id = random.choice(list(element_counts.keys()))
#                 element_counts[element_id] -= 1
#                 level_data[i, j] = element_id
#                 if element_counts[element_id] == 0:
#                     del element_counts[element_id]

#     return level_data

# def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with the specified width and height."""
#     level_data = torch.full((width, height), -1, dtype=torch.int32)

#     # Define element identifiers
#     player_id = 42
#     goal_id = 5
#     wall_id = -1

#     # Build walls
#     level_data[:, 0] = wall_id  # Left wall
#     level_data[:, height - 1] = wall_id  # Right wall
#     level_data[0, :] = wall_id  # Top wall
#     level_data[width - 1, :] = wall_id  # Bottom wall

#     # Assign player and goal
#     level_data[0, 0] = player_id
#     level_data[width - 1, height - 1] = goal_id

#     element_counts = {
#         wall_id: int((width + height - 2) * 2),  # Total number of wall elements needed
#         player_id: 1,  # Player element
#         goal_id: 1  # Goal element
#     }

#     # Calculate the remaining number of elements to be placed
#     remaining_elements = width * height - sum(element_counts.values())

#     # Calculate the number of elements to be placed
#     total_elements = (width - 2) * (height - 2) - 2
#     empty_elements = int(total_elements * empty_percentage)
#     enemy_elements = int(total_elements * enemy_probability)
#     item_elements = int(total_elements * item_probability)
#     friend_elements = int(total_elements * friend_probability)

#     # Generate the level data
#     element_counts = {
#         0: empty_elements,
#         1: enemy_elements,
#         2: item_elements,
#         3: friend_elements
#     }

#     for i in range(1, width - 1):
#         for j in range(1, height - 1):
#             if level_data[i, j] == -1:
#                 element_id = random.choice(list(element_counts.keys()))
#                 element_counts[element_id] -= 1
#                 level_data[i, j] = element_id
#                 if element_counts[element_id] == 0:
#                     del element_counts[element_id]

#     return level_data

# def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with walls surrounding the edges and random placement of elements inside."""

#     level_data = torch.zeros((width, height), dtype=torch.int)
#     print_level_data_grid(level_data)

#     element_counts = {
#         wall_id: int(2*width + 2*height)-2,  # Total number of wall elements needed
#         player_id: 1,  # Player element
#         goal_id: 1  # Goal element
#     }

#     # Calculate the remaining number of elements to be placed
#     x=sum(element_counts.values())
#     remaining_elements = width * height - x

#     # Calculate the percentages for the remaining elements
#     total_percentage = enemy_probability + item_probability + friend_probability
#     enemy_probability /= total_percentage
#     item_probability /= total_percentage
#     friend_probability /= total_percentage

#     # Assign elements based on percentages
#     for _ in range(remaining_elements):
#         element_id = random.choices(
#             population=list(element_counts.keys()),
#             weights=list(element_counts.values())
#         )[0]

#         # Find a random empty position for the element
#         while True:
#             i = random.randint(1, width - 2)
#             j = random.randint(1, height - 2)
#             if level_data[i, j] == 0:  # Empty position
#                 level_data[i, j] = element_id
#                 break

#         element_counts[element_id] -= 1

#         # Generate the walls
#     level_data[0, :] = wall_id
#     level_data[-1, :] = wall_id
#     level_data[:, 0] = wall_id
#     level_data[:, -1] = wall_id

#     # Assign player and goal
#     level_data[0, 0] = player_id
#     level_data[-1, -1] = goal_id

#     return level_data

# def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with walls surrounding the edges and random placement of elements inside."""

#     level_data = torch.zeros((width, height), dtype=torch.int)
#     print_level_data_grid(level_data)

#     element_counts = {
#         wall_id: (width + height - 2) * 2 - 2,  # Total number of wall elements needed, subtracting player and goal
#         player_id: 1,  # Player element
#         goal_id: 1  # Goal element
#     }

#     # Calculate the remaining number of elements to be placed
#     x = sum(element_counts.values())
#     remaining_elements = width * height - x

#     # Calculate the percentages for the remaining elements
#     total_percentage = enemy_probability + item_probability + friend_probability
#     enemy_probability /= total_percentage
#     item_probability /= total_percentage
#     friend_probability /= total_percentage

#     # Generate the walls
#     level_data[0, :] = wall_id
#     level_data[-1, :] = wall_id
#     level_data[:, 0] = wall_id
#     level_data[:, -1] = wall_id

#     # Assign player and goal
#     level_data[0, 0] = player_id
#     level_data[-1, -1] = goal_id

#     # Assign elements based on percentages
#     for _ in range(remaining_elements):
#         element_id = random.choices(
#             population=[enemy_id, item_id, friend_id],
#             weights=[enemy_probability, item_probability, friend_probability]
#         )[0]

#         # Find a random empty position for the element
#         while True:
#             i = random.randint(1, width - 2)
#             j = random.randint(1, height - 2)
#             if level_data[i, j] == 0:  # Empty position
#                 level_data[i, j] = element_id
#                 break

#     return level_data

# def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
#     """Generates a level data tensor with walls surrounding the edges and a maze-like path for the player."""

#     level_data = torch.zeros((width, height), dtype=torch.int)
#     print_level_data_grid(level_data)

#     # Define element identifiers
#     wall_id = -1
#     player_id = 42
#     goal_id = 5
#     enemy_id = 1
#     item_id = 2
#     friend_id = 3

#     element_counts = {
#         wall_id: (width + height - 2) * 2 - 2,  # Total number of wall elements needed, subtracting player and goal
#         player_id: 1,  # Player element
#         goal_id: 1  # Goal element
#     }

#     # Calculate the remaining number of elements to be placed
#     x = sum(element_counts.values())
#     remaining_elements = width * height - x

#     # Calculate the percentages for the remaining elements
#     total_percentage = enemy_probability + item_probability + friend_probability
#     enemy_probability /= total_percentage
#     item_probability /= total_percentage
#     friend_probability /= total_percentage

#     # Generate the walls surrounding the edges
#     level_data[0, :] = wall_id
#     level_data[-1, :] = wall_id
#     level_data[:, 0] = wall_id
#     level_data[:, -1] = wall_id

#     # Generate the maze-like path using Randomized Prim's algorithm
#     visited = set()
#     frontier = set()
#     start = (1, 1)
#     visited.add(start)
#     frontier.add(start)

#     while frontier:
#         current = random.choice(list(frontier))
#         frontier.remove(current)

#         neighbors = []
#         x, y = current
#         if x >= 3 and (x - 2, y) not in visited:
#             neighbors.append((x - 2, y))
#         if y >= 3 and (x, y - 2) not in visited:
#             neighbors.append((x, y - 2))
#         if x < width - 3 and (x + 2, y) not in visited:
#             neighbors.append((x + 2, y))
#         if y < height - 3 and (x, y + 2) not in visited:
#             neighbors.append((x, y + 2))

#         if neighbors:
#             next_cell = random.choice(neighbors)
#             nx, ny = next_cell
#             level_data[nx, ny] = 0
#             level_data[x + (nx - x) // 2, y + (ny - y) // 2] = 0

#             visited.add(next_cell)
#             frontier.add(next_cell)
#             frontier.add((x, y))

#     # Assign player and goal
#     level_data[0, 0] = player_id
#     level_data[-1, -1] = goal_id

#     # Assign elements based on percentages
#     for _ in range(remaining_elements):
#         element_id = random.choices(
#             population=[enemy_id, item_id, friend_id],
#             weights=[enemy_probability, item_probability, friend_probability]
#         )[0]

#         # Find a random empty position for the element
#         while True:
#             i = random.randint(1, width - 2)
#             j = random.randint(1, height - 2)
#             if level_data[i, j] == 0:  # Empty position
#                 level_data[i, j] = element_id
#                 break

#     return level_data

def generate_maze(width, height):
    level_data = torch.ones((width, height), dtype=torch.int)
    wall_positions = []

    def generate_path(x, y):
        level_data[x, y] = 0
        level_data[x - 1, y] = 0
        level_data[x, y - 1] = 0
        wall_positions.append((x - 1, y))
        wall_positions.append((x, y - 1))
        wall_positions.append((x, y))

    start_x = random.randint(2, width - 3)
    start_y = random.randint(2, height - 3)
    generate_path(start_x, start_y)

    # Randomly place the goal along the edges of the maze
    goal_x = random.choice([0, width - 1])
    goal_y = random.randint(1, height - 2)
    level_data[goal_x, goal_y] = 5  # Assign the goal element

    while wall_positions:
        wall_x, wall_y = random.choice(wall_positions)
        valid_neighbors = []

        if wall_x >= 3 and level_data[wall_x - 2, wall_y] == 1:
            valid_neighbors.append((wall_x - 2, wall_y))
        if wall_y >= 3 and level_data[wall_x, wall_y - 2] == 1:
            valid_neighbors.append((wall_x, wall_y - 2))
        if wall_x < width - 3 and level_data[wall_x + 2, wall_y] == 1:
            valid_neighbors.append((wall_x + 2, wall_y))
        if wall_y < height - 3 and level_data[wall_x, wall_y + 2] == 1:
            valid_neighbors.append((wall_x, wall_y + 2))

        if valid_neighbors:
            neighbor_x, neighbor_y = random.choice(valid_neighbors)
            generate_path(neighbor_x, neighbor_y)
            level_data[(wall_x + neighbor_x) // 2, (wall_y + neighbor_y) // 2] = 0

        wall_positions.remove((wall_x, wall_y))

    return level_data



def generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
    """Generates a level data tensor with walls surrounding the edges and a maze-like path for the player."""

    level_data = torch.zeros((width, height), dtype=torch.int)
    #print_level_data_grid(level_data)

    # Define element identifiers
    wall_id = -1
    player_id = 42
    goal_id = 5
    enemy_id = 1
    item_id = 2
    friend_id = 3

    element_counts = {
        wall_id: (width + height - 2) * 2 - 2,  # Total number of wall elements needed, subtracting player and goal
        player_id: 1,  # Player element
        goal_id: 1  # Goal element
    }

    # Calculate the remaining number of elements to be placed
    x = sum(element_counts.values())
    remaining_elements = width * height - x

    # Calculate the percentages for the remaining elements
    total_percentage = enemy_probability + item_probability + friend_probability
    enemy_probability /= total_percentage
    item_probability /= total_percentage
    friend_probability /= total_percentage

    # Generate the walls surrounding the edges
    level_data[0, :] = wall_id
    level_data[-1, :] = wall_id
    level_data[:, 0] = wall_id
    level_data[:, -1] = wall_id

    
    level_data = generate_maze(width, height)

    # Find a random position along the wall for the goal
    wall_positions = []
    for i in range(1, width - 1):
        if level_data[i, 0] == 0:  # Check left wall
            wall_positions.append((i, 0))
        if level_data[i, height - 1] == 0:  # Check right wall
            wall_positions.append((i, height - 1))
    for j in range(1, height - 1):
        if level_data[0, j] == 0:  # Check top wall
            wall_positions.append((0, j))
        if level_data[width - 1, j] == 0:  # Check bottom wall
            wall_positions.append((width - 1, j))

    goal_position = random.choice(wall_positions)
    level_data[goal_position] = goal_id

    # Find a random position along the wall for the player
    player_position = random.choice(wall_positions)
    level_data[player_position] = player_id

    # Generate the remaining elements randomly
    for _ in range(remaining_elements):
        element_id = random.choices(
            population=[enemy_id, item_id, friend_id],
            weights=[enemy_probability, item_probability, friend_probability]
        )[0]

        while True:
            i = random.randint(1, width - 2)
            j = random.randint(1, height - 2)
            if level_data[i, j] == 0:  # Empty position
                level_data[i, j] = element_id
                break

    return level_data


def print_level_description(level_data):
    """Prints a human-readable description of the level data."""
    element_labels = {
        0: "Wall",
        -2: "Free Space",
        1: "Enemy",
        2: "Item",
        3: "Friend",
        5: "Goal",
        42: "Player"
    }

    width, height = level_data.shape

    print("Level Description:")
    for i in range(width):
        for j in range(height):
            element_id = int(level_data[i, j])
            element_label = element_labels[element_id]
            print(element_label, end=" ")
        print()


def find_farthest_wall_position(level_data):
    """Finds the farthest wall position from the center of the level data."""
    width, height = level_data.shape
    center_x = width // 2
    center_y = height // 2
    max_distance = -1
    farthest_position = None

    for i in range(width):
        for j in range(height):
            if level_data[i, j] == 0:  # Free space (Wall)
                distance = abs(i - center_x) + abs(j - center_y)
                if distance > max_distance:
                    max_distance = distance
                    farthest_position = (i, j)

    return farthest_position


# def print_level_description(level_data):
#     """Prints a human-readable description of the level data."""
#     element_labels = {
#         0: "Free Space",
#         1: "Wall",
#         5: "Goal",
#         42: "Player"
#     }

#     width, height = level_data.shape

#     print("Level Description:")
#     for i in range(width):
#         for j in range(height):
#             element_id = int(level_data[i, j])
#             element_label = element_labels[element_id]
#             print(element_label, end=" ")
#         print()



def print_level_data_grid(level_data):
    """Prints the level data grid."""
    width, height = level_data.shape
    for i in range(height):
        for j in range(width):
            print(int(level_data[j, i]), end=' ')
        print()


element_labels = {
    -2: "O",
    1: "E",
    2: "I",
    3: "F",
    5: "G",
    42: "P",
    0: "W"
}
def print_level_description(level_data, element_labels):
    for row in level_data:
        for element_id in row:
            element_label = element_labels[element_id]
            print(element_label, end=' ')
        print()




# def print_level_description(level_data):
#     """Prints a human-readable description of the level data."""
#     element_labels = {
#         -2: "O",
#         1: "E",
#         2: "I",
#         3: "F",
#         5: "G",
#         42: "P",
#         -1: "W"
#     }

#     width, height = level_data.shape

#     for i in range(width):
#         for j in range(height):
#             element_id = int(level_data[i, j])
#             element_label = element_labels[element_id]
#             print(element_label, end=" ")
#         print()


empty_percentage = .40
item_probability = .1
enemy_probability = .15
friend_probability = .28
if (empty_percentage + item_probability + enemy_probability + friend_probability) < 1:
    item_probability = 1 - (empty_percentage + item_probability + enemy_probability + friend_probability)
# Define element identifiers
player_id = 42
free_space_id = -2 #open space was 0
goal_id = 5
enemy_id = 1
item_id = 2
friend_id = 3
wall_id = -1

collision_obstacle_ids = [enemy_id, item_id, friend_id]

# #def generate_level_data_grid(empty_percentage, enemy_probability, item_probability, friend_probability, difficulty_level, width, height):
# level_data = generate_level_data_grid(empty_percentage, enemy_probability, item_probability, friend_probability, "normal", 10, 10)

# print("Level Data:")
# print_level_data_grid(level_data)

# print("\nLevel Description:")
# print_level_description(level_data)

level_data = generate_level_data_grid_walls(empty_percentage, enemy_probability, item_probability, friend_probability, "normal", 10, 10)
print("wall boundry layout:")
print_level_data_grid(level_data)

print("\nLevel Description:")
print_level_description(level_data, element_labels)




# Example usage in regular gameplay
i=1
level_map = load_level_map(f'level{i}.json')
current_tile = get_current_tile((0,0))
process_tile(current_tile)

# Example usage in BI mini-game
player_choice = get_player_choice()
correct_choice = get_correct_choice()
process_prediction(player_choice, correct_choice)



import json
import pygame
import networkx as nx

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data

# Assume level_map is your 2D level map with 1s representing solid tiles
# and 0s representing empty tiles
def generate_graph(level_map):
    graph = nx.Graph()

    rows = len(level_map)
    cols = len(level_map[0])

    # Add nodes to the graph
    for row in range(rows):
        for col in range(cols):
            if level_map[row][col] == 0:
                graph.add_node((row, col))

    # Add edges to the graph
    for row in range(rows):
        for col in range(cols):
            if level_map[row][col] == 0:
                # Check neighboring cells to add edges
                neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                for neighbor_row, neighbor_col in neighbors:
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        if level_map[neighbor_row][neighbor_col] == 0:
                            graph.add_edge((row, col), (neighbor_row, neighbor_col))

    return graph

# Usage example
level_map = load_level_map('level1.json')
print(level_map)
# level_map = [
#     [1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1],
#     [0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1]
# ]

graph = generate_graph(level_map)

# Test pathfinding
start = (0, 0)
end = (4, 4)

# if nx.has_path(graph, start, end):
#     path = nx.shortest_path(graph, start, end)
#     print("Path exists:", path)
# else:
#     print("No path exists")




def render_level(level_data):
    tile_size = 32  # Set the size of each tile
    level_width = len(level_data[0]) * tile_size
    level_height = len(level_data) * tile_size

    screen = pygame.display.set_mode((level_width, level_height))

    for row in range(len(level_data)):
        for col in range(len(level_data[row])):
            tile_value = level_data[row][col]
            tile_x = col * tile_size
            tile_y = row * tile_size

            # Check if the row and col indexes are within the bounds of the level map
            if row < 0 or row >= len(level_data) or col < 0 or col >= len(level_data[row]):
                continue

            # Render different tiles based on their values
            if tile_value == 0:
                # Render empty tile
                pygame.draw.rect(screen, (0, 0, 0), (tile_x, tile_y, tile_size, tile_size))
            elif tile_value == 1:
                # Render solid tile
                pygame.draw.rect(screen, (255, 255, 255), (tile_x, tile_y, tile_size, tile_size))
            elif tile_value == 2:
                # Render obstacle tile
                pygame.draw.rect(screen, (255, 0, 0), (tile_x, tile_y, tile_size, tile_size))
            elif tile_value == 3:
                # Render jump tile
                pygame.draw.rect(screen, (0, 255, 0), (tile_x, tile_y, tile_size, tile_size))

    # # Check if the start and end points are in the graph
    # start_in_graph = (start[0] >= 0 and start[0] < len(level_data)) and (start[1] >= 0 and start[1] < len(level_data[0]))
    # end_in_graph = (end[0] >= 0 and end[0] < len(level_data)) and (end[1] >= 0 and end[1] < len(level_data[0]))

    # if start_in_graph and end_in_graph:
    #     if nx.has_path(graph, start, end):
    #         path = nx.shortest_path(graph, start, end)
    #         print("Path exists:", path)
    #     else:
    #         print("No path exists")
    # else:
    #     print("Start or end point is not in the graph")

    pygame.display.update()

pygame.init()  # Initialize Pygame

i = 1
while True:
    while i < 12:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        # Usage example
        level_map = load_level_map(f'level{i}.json')
        print(level_map)
        print("||**SAC**||")
        render_level(level_map)
        pygame.time.wait(1000)  # Wait for 1 second before moving to the next level
        i += 1











# Example reward values
tile_rewards = {
    "points": {
        "free_space": 10,
        "enemy": 0,
        "item": 5,
        "friend": 20
    },
    "health": {
        "free_space": 0,
        "enemy": -10,
        "item": 20,
        "friend": 10
    },
    "items": {
        "free_space": None,
        "enemy": None,
        "item": {
            "speed": 5,
            "slow": -5,
            "jump": 10,
            "kick": 15,
            "stomp": 20,
            "slide": 5,
            "shield": 10,
            "double_jump": 15,
            "invincibility": 20,
            "health": 10,
            "extra_life": 50
        },
        "friend": None
    },
    "xp": {
        "free_space": 0,
        "enemy": 25,
        "item": 10,
        "friend": 50
    }
}


# Define reward values for tile types
tile_rewards = {
    "path": 0,
    "wall": -10,
    "puddle": -5,
    "mud": -7,
    "spring": 5,
    "teleport": 10,
    "trampoline": 8,
    "cloud": 3,
    "ladder": 2,
    "item": 20,
    "enemy": -15,
    "friend": 15
}

# Access rewards during gameplay
def process_tile(tile):
    # Retrieve the reward value based on tile type
    reward = tile_rewards.get(tile.tile_type, 0)

    # Update game mechanics based on the reward value
    if reward > 0:
        # Player gains reward points or benefits
        increase_score(reward)
        collect_item(tile.thing_type)
    elif reward < 0:
        # Player faces a penalty or negative consequence
        decrease_score(abs(reward))
        handle_enemy_encounter(tile.thing_type)
    else:
        # Neutral tile, no reward or penalty
        continue_game()

    # Other game logic for the tile interaction

# Example usage
current_tile = get_current_tile((0,0))  # Get the tile the player is currently on
process_tile(current_tile)  # Process the rewards and consequences of the tile
