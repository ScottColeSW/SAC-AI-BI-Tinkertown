import random
import pygame
import torch
import text_assets as ta

class Metadata:
    def __init__(self, description, health, attack, defense):
        self.description = description
        self.health = health
        self.attack = attack
        self.defense = defense

class Tile:
    def __init__(self, tile_contents):
        self.tile_contents = tile_contents

class TileContents:
    def __init__(self, tile_type, contents):
        self.tile_type = tile_type
        self.contents = contents

class Contents:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.tiles = [[None for _ in range(width)] for _ in range(height)]

    def set_tile(self, x, y, tile):
        self.tiles[y][x] = tile

    def get_tile(self, x, y):
        return self.tiles[y][x]

    def display(self):
        for row in self.tiles:
            for tile in row:
                if tile is None:
                    print("Empty", end="\t")
                else:
                    if tile.tile_contents.contents is not None:
                        print(f"{tile.tile_contents.tile_type}: {tile.tile_contents.contents.name}", end="\t")
                    else:
                        print(f"{tile.tile_contents.tile_type}", end="\t")
            print()

    def generate_level_data_grid(self, empty_percentage, item_probability, difficulty_level):
        """Generates a level data tensor with the specified width, height, empty percentage, item probability, and difficulty level."""
        level_data = torch.randint(0, len(ta.tile_types), (self.width, self.height))

        # TODO: difficulty_modifier used to change item drop and enemy strength
        difficulty_modifier = 1 if difficulty_level == "normal" else 2
        for i in range(self.width):
            for j in range(self.height):
                if random.random() < empty_percentage:
                    level_data[i, j] = 0 #"path"
                elif random.random() > item_probability:
                    random_item_type = random.randint(0, len(ta.item_types) - 1)
                    level_data[i, j] = ta.item_types[random_item_type] # * difficulty_modifier
                else:
                    if random.random() < 0.5:
                        random_tile_type = random.choice(list(ta.tile_types.keys()))
                    else:
                        random_tile_type = random.choice(list(ta.enemy_metadata.keys()))
                    print(f"tile: {random_tile_type}")
                    level_data[i, j] = 1
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

# Define the size of the tilemap
map_width = 10
map_height = 10

# Create a grid
grid = Grid(map_width, map_height)

level_data = grid.generate_level_data_grid(0.5, 0.5, "normal")
filled_level_data = grid.fill_grid(level_data)


# Generate an empty tilemap
tilemap = [[0 for _ in range(map_width)] for _ in range(map_height)]

# Generate random tiles for the tilemap
for y in range(map_height):
    for x in range(map_width):
        # Choose a random tile type
        tile_type = random.choice(list(ta.tile_types.keys()))
        tilemap[y][x] = tile_type

# Print the generated tilemap
for row in tilemap:
    print(row)


# Create instances of Contents with appropriate metadata
enemy_contents = Contents("enemy", ta.enemy_metadata["Thornback"])
item_contents = Contents("item", ta.item_metadata["speed"])

# Create instances of TileContents
enemy_tile_contents = TileContents("enemy", enemy_contents)
item_tile_contents = TileContents("item", item_contents)

# Create instances of Tile
enemy_tile = Tile(enemy_tile_contents)
item_tile = Tile(item_tile_contents)

# Create instances of Tile
tile1 = Tile(TileContents("enemy", Contents("Thornback", ta.enemy_metadata["Thornback"])))
tile2 = Tile(TileContents("item", Contents("speed", ta.item_metadata["speed"])))
tile3 = Tile(TileContents("path", None))
tile4 = Tile(TileContents("wall", None))
tile5 = Tile(TileContents("enemy", Contents("Viperstrike", ta.enemy_metadata["Viperstrike"])))
tile6 = Tile(TileContents("path", None))
tile7 = Tile(TileContents("item", Contents("slow", ta.item_metadata["slow"])))
tile8 = Tile(TileContents("enemy", Contents("Frostbite", ta.enemy_metadata["Frostbite"])))
tile9 = Tile(TileContents("path", None))

# Set the tiles on the grid
grid.set_tile(0, 0, tile1)
grid.set_tile(1, 0, tile2)
grid.set_tile(2, 0, tile3)
grid.set_tile(0, 1, tile4)
grid.set_tile(1, 1, tile5)
grid.set_tile(2, 1, tile6)
grid.set_tile(0, 2, tile7)
grid.set_tile(1, 2, tile8)
grid.set_tile(2, 2, tile9)

# Display the grid with contents
print("This is the first grid:")
grid.display()
print("")

#Create instances of TileContents with the appropriate Contents objects
tile1 = Tile(TileContents("enemy", Contents("Thornback", ta.enemy_metadata["Thornback"])))
tile2 = Tile(TileContents("item", Contents("speed", ta.item_metadata["speed"])))
tile3 = Tile(TileContents("path", None))
tile4 = Tile(TileContents("wall", None))
tile5 = Tile(TileContents("enemy", Contents("Frostbite", ta.enemy_metadata["Frostbite"])))
tile6 = Tile(TileContents("path", None))

# Display the grid with contents
print("This is the second grid set to a list of tiles:")
grid = [[tile1, tile2, tile3],
        [tile4, tile5, tile6]]

for row in grid:
    for tile in row:
        if tile is None:
            print("Empty", end="\t")
        elif tile.tile_contents.contents is not None:
            print(f"{tile.tile_contents.tile_type}: {tile.tile_contents.contents.name}", end="\t")
        else:
            print(f"{tile.tile_contents.tile_type}", end="\t")
    
print("Done")