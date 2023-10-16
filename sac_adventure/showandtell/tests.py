import numpy as np
from collections import namedtuple

class Tile:
    def __init__(self, tile_type, thing_type, metadata):
        self.tile_type = tile_type
        self.thing_type = thing_type
        self.metadata = metadata

# ItemType named tuple
ItemType = namedtuple("ItemType", ["name", "metadata"])
item_types = {
    0: ItemType("speed", "SpeedItem"),
    1: ItemType("slow", "SlowItem"),
    2: ItemType("jump", "JumpItem"),
    3: ItemType("kick", "KickItem"),
    4: ItemType("stomp", "StompItem"),
    5: ItemType("slide", "SlideItem"),
    6: ItemType("shield", "ShieldItem"),
    7: ItemType("double_jump", "DoubleJumpItem"),
    8: ItemType("invincibility", "InvincibilityItem"),
    9: ItemType("health", "HealthItem"),
    10: ItemType("extra_life", "ExtraLifeItem")
}

# ActionMetadata named tuple
ActionMetadata = namedtuple("ActionMetadata", ["description", "interactions"])
action_metadata = {
    "path": ActionMetadata(
        description="A traversable tile that allows the player to move freely.",
        interactions=["open"]
    ),
    "wall": ActionMetadata(
        description="An impassable tile that blocks the player's movement.",
        interactions=["BLOCKED"]
    ),
    "puddle": ActionMetadata(
        description="A slippery tile that slows the player's movement.",
        interactions=["random_slide"]
    ),
    "mud": ActionMetadata(
        description="A sticky tile that slows the player's movement.",
        interactions=["slow"]
    ),
    "spring": ActionMetadata(
        description="A tile that launches the player into the air.",
        interactions=["jump"]
    ),
    "teleport": ActionMetadata(
        description="A tile that teleports the player to a new location.",
        interactions=["blink"]
    ),
    "trampoline": ActionMetadata(
        description="A tile that bounces the player into the air.",
        interactions=["jump"]
    ),
    "cloud": ActionMetadata(
        description="A floating tile that allows the player to move freely.",
        interactions=["fly"]
    ),
    "ladder": ActionMetadata(
        description="A tile that allows the player to climb up or down.",
        interactions=["climb"]
    )
}
    
# ItemMetadata named tuple
ItemMetadata = namedtuple("ItemMetadata", ["description", "duration"])
item_metadata = {
    "SpeedItem": ItemMetadata(
        description="Increases the player's movement speed.",
        duration=10
    ),
    "SlowItem": ItemMetadata(
        description="Decreases the player's movement speed.",
        duration=10
    ),
    "JumpItem": ItemMetadata(
        description="Enables the player to jump higher.",
        duration=10
    ),
    "KickItem": ItemMetadata(
        description="Grants the player a powerful kicking ability.",
        duration=10
    ),
    "StompItem": ItemMetadata(
        description="Allows the player to stomp on enemies.",
        duration=10
    ),
    "SlideItem": ItemMetadata(
        description="Enables the player to slide under obstacles.",
        duration=10
    ),
    "ShieldItem": ItemMetadata(
        description="Provides the player with a protective shield.",
        duration=10
    ),
    "DoubleJumpItem": ItemMetadata(
        description="Allows the player to perform a double jump.",
        duration=10
    ),
    "InvincibilityItem": ItemMetadata(
        description="Makes the player invincible for a certain duration.",
        duration=10
    ),
    "HealthItem": ItemMetadata(
        description="Restores the player's health.",
        duration=0
    ),
    "ExtraLifeItem": ItemMetadata(
        description="Grants the player an extra life.",
        duration=0
    )
}

# Friend named tuple
Friend = namedtuple("Friend", ["description", "relationship", "item"])
friend_metadata = {
    "Willow": Friend(
        description="A wise and serene companion, guiding you with her deep knowledge of the world.",
        relationship="positive",
        item="Note"
    ),
    "Harmony": Friend(
        description="A harmonious friend, emanating peaceful vibes and offering valuable advice.",
        relationship="positive",
        item="Note"
    ),
    "Blaze": Friend(
        description="A passionate ally, igniting the spirit and bolstering your courage in the face of adversity.",
        relationship="positive",
        item="Note"
    ),
    "Aurora": Friend(
        description="A radiant friend, illuminating your path with her gentle glow and granting clarity.",
        relationship="positive",
        item="Note"
    ),
    "Zephyr": Friend(
        description="A playful companion, bringing refreshing winds and swift movements to aid you.",
        relationship="positive",
        item="Note"
    ),
    "Nova": Friend(
        description="A celestial friend, bestowing cosmic powers and lighting up the darkest of moments.",
        relationship="positive",
        item="Note"
    ),
    "Ivy": Friend(
        description="A nature-loving ally, harnessing the power of plants and offering healing and protection.",
        relationship="positive",
        item="Note"
    ),
    "Luna": Friend(
        description="A mysterious and enchanting friend, with powers linked to the moon and the night sky.",
        relationship="positive",
        item="Note"
    ),
    "Ember": Friend(
        description="A fiery spirit, imbuing your steps with passion and strength to overcome challenges.",
        relationship="positive",
        item="Note"
    ),
    "Jasper": Friend(
        description="A steadfast and loyal companion, offering unwavering support and unwritten bravery.",
        relationship="positive",
        item="Note"
    )
}

# Populate the enemy metadata
EnemyMetadata = namedtuple("EnemyMetadata", ["description", "health", "attack", "defense", "xp"])
enemy_metadata = {
    "Thornback": EnemyMetadata(
        description="A formidable adversary with razor-sharp thorns that can pierce any surface.",
        health=100,
        attack=10,
        defense=5,
        xp=25
    ),
    "Frostbite": EnemyMetadata(
        description="A chilling opponent that freezes everything it touches, turning the ground to ice.",
        health=300,
        attack=30,
        defense=25,
        xp=25
    ),
    "Viperstrike": EnemyMetadata(
        description="A chilling opponent that freezes everything it touches, turning the ground to ice.",
        health=200,
        attack=50,
        defense=5,
        xp=25
    ),
    "Shadowclaw": EnemyMetadata(
        description="A stealthy and elusive adversary, capable of disappearing into the darkness.",
        health=200,
        attack=20,
        defense=15,
        xp=0
    ),
    "Venomfang": EnemyMetadata(
        description="A deadly enemy with a venomous bite, leaving a trail of poison in its wake.",
        health=250,
        attack=25,
        defense=20,
        xp=0
    ),
    "Blazewing": EnemyMetadata(
        description="A fiery adversary, soaring through the air, leaving trails of scorching flames.",
        health=350,
        attack=35,
        defense=30,
        xp=0
    ),
    "Scorchscale": EnemyMetadata(
        description="A fearsome dragon-like enemy, breathing scorching hot flames that melt anything in its path.",
        health=400,
        attack=40,
        defense=35,
        xp=0
    ),
    "Gloomspike": EnemyMetadata(
        description="A dark and gloomy foe, striking with spikes that drain the energy from its targets.",
        health=450,
        attack=45,
        defense=40,
        xp=0
    ),
    "Stormrider": EnemyMetadata(
        description="A tempestuous enemy, conjuring powerful storms and unleashing destructive lightning bolts.",
        health=500,
        attack=50,
        defense=45,
        xp=0
    ),
    "Bouldercrush": EnemyMetadata(
        description="A massive and mighty opponent, capable of crushing anything with its colossal strength.",
        health=550,
        attack=55,
        defense=50,
        xp=0
    )
}

tile_types = {
    0: "path", #action_metadata
    1: "enemy", #enemy_metadata
    2: "item", #item_metadata
    3: "wall",#action_metadata
    4: "puddle",#action_metadata
    5: "mud",#action_metadata
    6: "friend", #friend_metadata
    7: "spring",#action_metadata
    8: "teleport",#action_metadata
    9: "trampoline",#action_metadata
    10: "cloud",#action_metadata
    11: "ladder",#action_metadata
    # 12: "open",#action_metadata
    # 13: "closed",#action_metadata
    # 14: "blink",
    # 15: "slow",
}

import random

# Define the function to built the tile tensor
def assign_tile_info(rows, cols):
    # Create an empty tensor with the specified dimensions
    tensor = np.empty((rows, cols), dtype=object)

    # Populate the tensor with tile information
    for row in range(rows):
        for col in range(cols):
            # Generate a random tile type
            tile_type = np.random.choice(list(tile_types.values()))

            # Assign the tile information to the tensor
            if tile_type == "item":

                # Randomly pick an item
                random_item = random.choice(list(item_metadata.keys()))

                # Access the values of the randomly picked item
                item_name = item_metadata[random_item].duration

                # Access the description of the random item
                item_desc = item_metadata[random_item].description

                tensor[row, col] = Tile(random_item, item_name, item_desc)

            elif tile_type == "enemy":
                enemy_type = np.random.choice(list(enemy_metadata.keys()))
                enemy_xp = enemy_metadata[enemy_type].xp
                metadata = enemy_metadata[enemy_type].description
                tensor[row, col] = Tile(enemy_type, metadata, enemy_xp)
            elif tile_type == "friend":
                friend_type = np.random.choice(list(friend_metadata.keys()))
                metadata = friend_metadata[friend_type].description
                friend_item = friend_metadata[friend_type].item
                tensor[row, col] = Tile(friend_type, metadata, friend_item)
            else:
                # TODO 
                action_type = np.random.choice(list(action_metadata.keys()))
                metadata = action_metadata[action_type].description
                character_actions = action_metadata[action_type].interactions[0]
                tensor[row, col] = Tile(action_type, metadata, character_actions)

    return tensor

import numpy as np

# Test the function
rows = 10
cols = 10
tile_tensor = assign_tile_info(rows, cols)
print(f"tile tensor size {tile_tensor.itemsize}")

print("the whole enchilada")
# Print the populated tensor
# for row in tile_tensor:
#     for t in row:
# #        print(t.metadata)
#         print(t.tile_type)
#     print()

print("Matrix of tile types:")
invalid_tile_types = []
for row in tile_tensor:
    for tile in row:
        tile_type = tile.metadata
#        if tile_type in tile_types:
        if tile.metadata in tile_types:
            #tile_index = list(tile_types.values()).index(tile_type)
            #print(tile_index, end=" ")
            if tile_type in tile_types.values():
                tile_index = list(tile_types.keys())[list(tile_types.values()).index(tile_type)]
                print(f"({tile_index})", end=" ")

#            tile_index = list(tile_types.keys())[list(tile_types.values()).index(tile_type)]
#            print(f"({tile_index})", end=" ")
        else:
            #print(f"Invalid tile type: {tile_type} in {tile.metadata}", end=" ")
            invalid_tile_types.append(tile_type)
#    print()

print(f"\nInvalid tile types: {invalid_tile_types}")
# for tt in invalid_tile_types:
#     print(f"Invalid: {tt}")


# import numpy as np

# # ...

# # Create the tile tensor
# tile_tensor = np.empty((rows, cols), dtype=object)
# for i in range(rows):
#     for j in range(cols):
#         tile = Tile()
#         tile.metadata = np.random.choice(list(tile_types.keys()))
#         tile_tensor[i][j] = tile

# Create a vectorized function to extract numeric values from tiles

# #get_numeric_value = np.vectorize(lambda tile: list(tile_types.keys())[list(tile_types.values()).index(tile_type)])
# get_numeric_value = np.vectorize(lambda tile_type, tile: list(tile_types.keys())[list(tile_types.values()).index(tile_type)])

# # Get the matrix of indices
# indices_matrix = get_numeric_value(tile_tensor)

# Define the tile types with string keys
tile_type_inverted = {
    "path": 0, #action_metadata
    "enemy": 1, #enemy_metadata
    "item": 2, #item_metadata
    "wall": 3, #action_metadata
    "puddle": 4, #action_metadata
    "mud": 5, #action_metadata
    "friend": 6, #friend_metadata
    "spring": 7, #action_metadata
    "teleport": 8, #action_metadata
    "trampoline": 9, #action_metadata
    "cloud": 10, #action_metadata
    "ladder": 11, #action_metadata
    # "open": 12, #action_metadata
    # "closed": 13, #action_metadata
    # "blink": 14,
    # "slow": 15,
}

import numpy as np

# Create a vectorized function to extract numeric values from tiles
#get_numeric_value = np.vectorize(lambda tile: list(tile_types.keys())[list(tile_types.values()).index(tile.metadata)])
#get_numeric_value = np.vectorize(lambda tile: list(tile_types.keys())[list(tile_types.values()).index(tile.metadata)], tile.metadata)
#get_numeric_value = np.vectorize(lambda tile: list(tile_types.keys()).index(tile.metadata))
#get_numeric_value = np.vectorize(lambda tile: list(tile_type_inverted.keys()).index(tile.metadata) if tile.metadata in tile_type_inverted else -1)
# get_numeric_value = np.vectorize(lambda tile: tile_type_inverted.get(tile.metadata, -1))
# get_numeric_value = np.vectorize(lambda tile: tile_type_inverted.get(tile.metadata, -1))
#get_numeric_value = np.vectorize(lambda tile: tile_type_inverted.get(tile, -1))
#get_numeric_value = np.vectorize(lambda tile: list(tile_type_inverted.values()).index(tile_type_inverted.get(tile, -1)))
#get_numeric_value = np.vectorize(lambda tile: list(tile_type_inverted.values()).index(tile_type_inverted.get(tile, -1)) if tile in tile_type_inverted else -1)
#get_numeric_value = np.vectorize(lambda tile: tile_type_inverted.get(tile, -1))
#get_numeric_value = np.vectorize(lambda tile: list(tile_type_inverted.keys())[list(tile_type_inverted.values()).index(tile)])
#get_numeric_value = np.vectorize(lambda tile: list(tile_type_inverted.keys())[list(tile_type_inverted.values()).index(tile.metadata)])
#get_numeric_value = np.vectorize(lambda tile: list(tile_type_inverted.keys()).index(str(tile.metadata)))
get_numeric_value = np.vectorize(lambda tile: tile_type_inverted.get(str(tile.metadata), -1))


print(f"tile keys: {list(tile_types.keys())}")
print()

# Get the matrix of indices
indices_matrix = get_numeric_value(tile_tensor)

# Print the matrix of indices
print("Matrix of indices:")
print(indices_matrix)

# # Custom conversion method to extract numeric values
# get_numeric_value = np.vectorize(lambda tile: tile.metadata)

# # Convert tile_tensor to a NumPy array of numeric values
# numeric_array = get_numeric_value(tile_tensor)

# print("XXXX Matrix of numbers:")
# print(numeric_array)

print("\nMatrix of numbers:")
for row in tile_tensor:
    for tile in row:
        print(tile.metadata, end=" ")
    print()




# from collections import namedtuple

# from enum import Enum

# class ItemType(Enum):
#     SPEED = 0
#     SLOW = 1
#     JUMP = 2
#     KICK = 3
#     STOMP = 4
#     SLIDE = 5
#     SHIELD = 6
#     DOUBLE_JUMP = 7
#     INVINCIBILITY = 8
#     HEALTH = 9
#     EXTRA_LIFE = 10

# ItemMetadata = namedtuple("ItemMetadata", ["description", "duration", "jump", "power", "friction", "boost", "lives"])

# item_metadata = {
#     ItemType.SPEED: ItemMetadata("Increases movement speed.", 10, None, None, None, None, None),
#     ItemType.SLOW: ItemMetadata("Slows down movement speed.", 10, None, None, None, None, None),
#     ItemType.JUMP: ItemMetadata("Jump Boost", 10, 110, None, None, None, None),
#     ItemType.KICK: ItemMetadata("Power Kick", 10, None, 110, None, None, None),
#     ItemType.STOMP: ItemMetadata("Stomp!", 10, None, 110, None, None, None),
#     ItemType.SLIDE: ItemMetadata("Super Slide", 10, None, None, 1, None, None),
#     ItemType.SHIELD: ItemMetadata("Quick Shield", 10, None, 110, None, None, None),
#     ItemType.DOUBLE_JUMP: ItemMetadata("Double Jump", 10, 110, None, None, None, None),
#     ItemType.INVINCIBILITY: ItemMetadata("Invincible!", 10, None, None, None, None, None),
#     ItemType.HEALTH: ItemMetadata("Health Boost", 10, None, None, None, 110, None),
#     ItemType.EXTRA_LIFE: ItemMetadata("Extra Life!", 10, None, None, None, None, 1),
# }

# combined_items = {item_type: item_metadata[item_type] for item_type in ItemType}

# print(combined_items)
# #############WORKS UP####

# import random
# from collections import namedtuple

# Tile = namedtuple("Tile", ["type", "metadata"])

# tile_types = {
#     0: "path",  # action_metadata
#     1: "enemy",  # enemy_metadata
#     2: "item",  # item_metadata
#     3: "wall",  # action_metadata
#     4: "puddle",  # action_metadata
#     5: "mud",  # action_metadata
#     6: "friend",  # friend_metadata
#     7: "spring",  # action_metadata
#     8: "teleport",  # action_metadata
#     9: "trampoline",  # action_metadata
#     10: "cloud",  # action_metadata
#     11: "ladder"  # action_metadata
# }

# Enemy = namedtuple("Enemy", ["description", "health", "attack", "defense", "xp"])

# enemy_metadata = {
#     "Thornback": Enemy(
#         description="A formidable adversary with razor-sharp thorns that can pierce any surface.",
#         health=100,
#         attack=10,
#         defense=5,
#         xp=25
#     ),
#     "Frostbite": Enemy(
#         description="A chilling opponent that freezes everything it touches, turning the ground to ice.",
#         health=300,
#         attack=30,
#         defense=25,
#         xp=25
#     ),
#     "Viperstrike": Enemy(
#         description= "A chilling opponent that freezes everything it touches, turning the ground to ice.",
#         health= 200,
#         attack= 50,
#         defense= 5,
#         xp= 25
#     ),
#     "Shadowclaw": Enemy(
#         description= "A stealthy and elusive adversary, capable of disappearing into the darkness.",
#         health= 200,
#         attack= 20,
#         defense= 15,
#         xp= 25
#     ),
#     "Venomfang": Enemy(
#         description= "A deadly enemy with a venomous bite, leaving a trail of poison in its wake.",
#         health= 250,
#         attack= 25,
#         defense= 20,
#         xp= 25
#     ),
#     "Blazewing": Enemy(
#         description= "A fiery adversary, soaring through the air, leaving trails of scorching flames.",
#         health= 350,
#         attack= 35,
#         defense= 30,
#         xp= 25
#     ),
#     "Scorchscale": Enemy(
#         description= "A fearsome dragon-like enemy, breathing scorching hot flames that melt anything in its path.",
#         health= 400,
#         attack= 40,
#         defense= 35,
#         xp= 25
#     ),
#     "Gloomspike": Enemy(
#         description= "A dark and gloomy foe, striking with spikes that drain the energy from its targets.",
#         health= 450,
#         attack= 45,
#         defense= 40,
#         xp= 25
#     ),
#     "Stormrider": Enemy(
#         description= "A tempestuous enemy, conjuring powerful storms and unleashing destructive lightning bolts.",
#         health= 500,
#         attack= 50,
#         defense= 45,
#         xp= 25
#     ),
#     "Bouldercrush": Enemy(
#         description= "A massive and mighty opponent, capable of crushing anything with its colossal strength.",
#         health= 550,
#         attack= 55,
#         defense= 50,
#         xp= 25
#     )
# }

# Friend = namedtuple("Friend", ["description", "relationship", "item"])

# friend_metadata = {
#     "Willow": Friend(
#         description="A wise and serene companion, guiding you with her deep knowledge of the world.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Harmony": Friend(
#         description="A harmonious friend, emanating peaceful vibes and offering valuable advice.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Blaze": Friend(
#         description="A passionate ally, igniting the spirit and bolstering your courage in the face of adversity.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Aurora": Friend(
#         description="A radiant friend, illuminating your path with her gentle glow and granting clarity.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Zephyr": Friend(
#         description="A playful companion, bringing refreshing winds and swift movements to aid you.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Nova": Friend(
#         description="A celestial friend, bestowing cosmic powers and lighting up the darkest of moments.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Ivy": Friend(
#         description="A nature-loving ally, harnessing the power of plants and offering healing and protection.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Luna": Friend(
#         description="A mysterious and enchanting friend, with powers linked to the moon and the night sky.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Ember": Friend(
#         description="A fiery spirit, imbuing your steps with passion and strength to overcome challenges.",
#         relationship="positive",
#         item="Note"
#     ),
#     "Jasper": Friend(
#         description="A steadfast and loyal companion, offering unwavering support and unwritten bravery.",
#         relationship="positive",
#         item="Note"
#     )
# }

# ActionMetadata = namedtuple("ActionMetadata", ["description", "interactions"])

# action_metadata = {
#     "path": ActionMetadata(
#         description="A traversable tile that allows the player to move freely.",
#         interactions=[]
#     ),
#     "wall": ActionMetadata(
#         description="An impassable tile that blocks the player's movement.",
#         interactions=[]
#     ),
#     "puddle": ActionMetadata(
#         description="A slippery tile that slows the player's movement.",
#         interactions=[]
#     ),
#     "mud": ActionMetadata(
#         description="A sticky tile that slows the player's movement.",
#         interactions=[]
#     ),
#     "spring": ActionMetadata(
#         description="A tile that launches the player into the air.",
#         interactions=["jump"]
#     ),
#     "teleport": ActionMetadata(
#         description="A tile that teleports the player to a new location.",
#         interactions=[]
#     ),
#     "trampoline": ActionMetadata(
#         description="A tile that bounces the player into the air.",
#         interactions=["jump"]
#     ),
#     "cloud": ActionMetadata(
#         description="A floating tile that allows the player to move freely.",
#         interactions=[]
#     ),
#     "ladder": ActionMetadata(
#         description="A tile that allows the player to climb up or down.",
#         interactions=[]
#     )
# }

# def assign_tile_info(tile_type):
#     if tile_type == 1:  # Enemy tile type
#         enemy_name = random.choice(list(enemy_metadata))
#         enemy = enemy_metadata[enemy_name]
#         tile_info = Tile(type="enemy", metadata=enemy)
#     elif tile_type == 6:  # Friend tile type
#         friend_name = random.choice(list(friend_metadata))
#         friend = friend_metadata[friend_name]
#         tile_info = Tile(type="friend", metadata=friend)
#     else:
#         # Handle other tile types
#         tile_info = Tile(type=tile_types[tile_type], metadata=None)
    
#     return tile_info

# # Example usage:

# i=1
# while i<12:
#     random_tile_type = random.randint(0, 11)
#     tile_info = assign_tile_info(random_tile_type)
#     print("Tile Type:", tile_info.type)
#     print("Tile Metadata:", tile_info.metadata)

# print("AND NOW FOR SOMETHING...")

# import numpy as np

# # Define the dimensions of the tensor
# rows = 10
# cols = 10

# # Create an empty tensor with the specified dimensions
# tensor = np.empty((rows, cols), dtype=object)

# # Populate the tensor with tile information
# for row in range(rows):
#     for col in range(cols):
#         # Generate a random tile type
#         tile_type = np.random.choice(list(tile_types.keys()))

#         # Populate the tile information based on the tile type
#         if tile_type == 0:
#             tile_info = action_metadata[np.random.choice(list(action_metadata.keys()))]
#         elif tile_type == 1:
#             tile_info = enemy_metadata[np.random.choice(list(enemy_metadata.keys()))]
#         elif tile_type == 2:
#             tile_info = item_metadata[np.random.choice(list(item_metadata.keys()))]
#         elif tile_type == 6:
#             tile_info = friend_metadata[np.random.choice(list(friend_metadata.keys()))]
#         else:
#             tile_info = action_metadata[np.random.choice(list(action_metadata.keys()))]

#         # Assign the tile information to the tensor
#         tensor[row, col] = tile_info

# # Print the populated tensor
# print(tensor)





# ############################## WORKS ########################################
# # # import pygame

# # def draw_rectangle(screen, x, y, width, height):
# #     pygame.draw.rect(screen, (255, 0, 0), (x, y, width, height))

# # def main():
# #     screen = pygame.display.set_mode((640, 480))
# #     draw_rectangle(screen, 100, 100, 200, 100)
# #     pygame.display.flip()

# #     while True:
# #         for event in pygame.event.get():
# #             if event.type == pygame.QUIT:
# #                 NotImplemented
# #                 exit()

# # if __name__ == "__main__":
# #   main()
# ############################## WORKS ########################################





# # import random
# # import torch
# # import text_assets as ta

# # def fill_tensor(tensor, tile_types, item_types, enemy_metadata, friend_metadata, item_metadata):
# #   for i in range(tensor.shape[0]):
# #     for j in range(tensor.shape[1]):
# #       tile_type = random.choice(list(tile_types.keys()))
# #       if tile_type == 1:
# #         enemy_type = random.choice(list(enemy_metadata.keys()))
# #         tensor[i, j] = enemy_metadata[enemy_type]
# #       elif tile_type == 2:
# #         item_type = random.choice(list(item_types.keys()))
# #         tensor[i, j] = item_metadata[item_type]
# #       elif tile_type == 6:
# #         friend_type = random.choice(list(friend_metadata.keys()))
# #         tensor[i, j] = friend_metadata[friend_type]
# #       else:
# #         tensor[i, j] = tile_type

# # def print_tensor(tensor, tile_types, item_types, enemy_metadata, friend_metadata, item_metadata):
# #   for i in range(tensor.shape[0]):
# #     for j in range(tensor.shape[1]):
# #       tile_type, metadata = tensor[i, j]
# #       print(f"({i}, {j}): {tile_types[tile_type]} {metadata}")

# # if __name__ == "__main__":
# #   tensor = torch.zeros((10, 10))
# #   fill_tensor(tensor, ta.tile_types, ta.item_types, ta.enemy_metadata, ta.friend_metadata, ta.item_metadata)
# #   print_tensor(tensor, ta.tile_types, ta.item_types, ta.enemy_metadata, ta.friend_metadata, ta.item_metadata)
