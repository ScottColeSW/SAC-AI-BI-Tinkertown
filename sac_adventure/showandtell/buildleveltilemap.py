# here we only want to fill in the new level with each tile filled in completely

import random
import pygame
import torch
import text_assets as ta

# class Metadata:
#     def __init__(self, description, health, attack, defense):
#         self.description = description
#         self.health = health
#         self.attack = attack
#         self.defense = defense

# class Tile:
#     def __init__(self, tile_contents):
#         self.tile_contents = tile_contents

# class TileContents:
#     def __init__(self, tile_type, contents):
#         self.tile_type = tile_type
#         self.contents = contents

# class Contents:
#     def __init__(self, name, metadata):
#         self.name = name
#         self.metadata = metadata

# class Grid:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#         self.tiles = [[None for _ in range(width)] for _ in range(height)]

#     def set_tile(self, x, y, tile):
#         self.tiles[y][x] = tile

#     def get_tile(self, x, y):
#         return self.tiles[y][x]

#     def display(self):
#         for row in self.tiles:
#             for tile in row:
#                 if tile is None:
#                     print("Empty", end="\t")
#                 else:
#                     if tile.tile_contents.contents is not None:
#                         print(f"{tile.tile_contents.tile_type}: {tile.tile_contents.contents.name}", end="\t")
#                     else:
#                         print(f"{tile.tile_contents.tile_type}", end="\t")
#             print()

#     def generate_level_data_grid(self, empty_percentage, item_probability, difficulty_level):
#         """Generates a level data tensor with the specified width, height, empty percentage, item probability, and difficulty level."""
#         level_data = torch.randint(0, len(ta.tile_types), (self.width, self.height))

#         # TODO: difficulty_modifier used to change item drop and enemy strength
#         difficulty_modifier = 1 if difficulty_level == "normal" else 2
#         for i in range(self.width):
#             for j in range(self.height):
#                 if random.random() < empty_percentage:
#                     level_data[i, j] = 0 #"path"
#                 elif random.random() > item_probability:
#                     random_item_type = random.randint(0, len(ta.item_types) - 1)
#                     level_data[i, j] = ta.item_types[random_item_type] # * difficulty_modifier
#                 else:
#                     if random.random() < 0.5:
#                         random_tile_type = random.choice(list(ta.tile_types.keys()))
#                     else:
#                         random_tile_type = random.choice(list(ta.enemy_metadata.keys()))
#                     print(f"tile: {random_tile_type}")
#                     level_data[i, j] = random_tile_type
#         return level_data

import random
import pygame
import json
import text_assets as ta

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data

def fill_grid(self, level_data):
    for i in range(self.width):
        for j in range(self.height):
            if level_data[i, j] == "enemy":
                level_data[i, j] = random.choice(list(ta.enemy_metadata.keys()))
            elif level_data[i, j] == "friend":
                level_data[i, j] = random.choice(list(ta.friend_metadata.keys()))
            elif level_data[i, j] == "item":
                level_data[i, j] = random.choice(list(ta.item_metadata.keys()))
            else:
                level_data[i, j] = level_data[i, j]

    return level_data

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

    pygame.display.update()

pygame.init()  # Initialize Pygame

i = 1
while i < 11:
    level_data = generate_level_data_grid(0.5, 0.5, "normal")
    level_map = load_level_map(f'level{i}.json')
    render_level(level_map)
    filled_level_data = fill_grid(level_data)
    print(level_map)
    print("||**SAC**||")
    render_level(level_map)
    pygame.time.wait(1000)  # Wait for 1 second before moving to the next level
    i += 1