# Define the metadata for different contents and dictionaries

# User selected game setting
difficulty_types = {
    0: "normal",
    1: "fast",
    2: "slow",
    3: "powerful",
    4: "agile"
}

item_types = {
    0: "speed",
    1: "slow",
    2: "jump",
    3: "kick",
    4: "stomp",
    5: "slide",
    6: "shield",
    7: "double_jump",
    8: "invincibility",
    9: "health",
    10: "extra_life"
}

item_metadata = {
    "speed": {
        "description": "Increases movement speed.",
        "duration": 10
    },
    "slow": {
        "description": "Slows down movement speed.",
        "duration": 10
    },
    "jump": {
        "description": "Jump Boost",
        "jump": 110,
        "duration": 10
    },
    "kick": {
        "description": "Power Kick",
        "power": 110,
        "duration": 10
    },
    "stomp": {
        "description": "Stomp!",
        "power": 110,
        "duration": 10
    },
    "slide": {
        "description": "Super Slide",
        "friction": 1,
        "duration": 10
    },
    "shield": {
        "description": "Quick Shield",
        "power": 110,
        "duration": 10
    },
    "double_jump": {
        "description": "Double Jump",
        "jump": 110,
        "duration": 10
    },
    "invincibility": {
        "description": "Invincible!",
        "duration": 10
    },
    "health": {
        "description": "Health Boost",
        "boost": 110,
        "duration": 10
    },
    "extra_life": {
        "description": "Extra Life!",
        "lives": 1,
        "duration": 10
    }
}


# Every Tile has a type that is used to determine the metadata to be used. 
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
    11: "ladder"#action_metadata
}

item_types = {
    0: "speed",
    1: "slow",
    2: "jump",
    3: "kick",
    4: "stomp",
    5: "slide",
    6: "shield",
    7: "double_jump",
    8: "invincibility",
    9: "health",
    10: "extra_life"
}

item_metadata = {
    "speed": {
        "description": "Increases movement speed.",
        "duration": 10
    },
    "slow": {
        "description": "Slows down movement speed.",
        "duration": 10
    },
    "jump": {
        "description": "Jump Boost",
        "jump": 110,
        "duration": 10
    },
    "kick": {
        "description": "Power Kick",
        "power": 110,
        "duration": 10
    },
    "stomp": {
        "description": "Stomp!",
        "power": 110,
        "duration": 10
    },
    "slide": {
        "description": "Super Slide",
        "friction": 1,
        "duration": 10
    },
    "shield": {
        "description": "Quick Shield",
        "power": 110,
        "duration": 10
    },
    "double_jump": {
        "description": "Double Jump",
        "jump": 110,
        "duration": 10
    },
    "invincibility": {
        "description": "Invincible!",
        "duration": 10
    },
    "health": {
        "description": "Health Boost",
        "boost": 110,
        "duration": 10
    },
    "extra_life": {
        "description": "Extra Life!",
        "lives": 1,
        "duration": 10
    }
}



item_types = {
    0: "speed",
    1: "slow",
    2: "jump",
    3: "kick",
    4: "stomp",
    5: "slide",
    6: "shield",
    7: "double_jump",
    8: "invincibility",
    9: "health",
    10: "extra_life"
}

friend_metadata = {
    'Willow': {
        "description": 'A wise and serene companion, guiding you with her deep knowledge of the world.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Harmony': {
        "description": 'A harmonious friend, emanating peaceful vibes and offering valuable advice.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Blaze': {
        "description": 'A passionate ally, igniting the spirit and bolstering your courage in the face of adversity.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Aurora': {
        "description": 'A radiant friend, illuminating your path with her gentle glow and granting clarity.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Zephyr': {
        "description": 'A playful companion, bringing refreshing winds and swift movements to aid you.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Nova': {
        "description": 'A celestial friend, bestowing cosmic powers and lighting up the darkest of moments.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Ivy': {
        "description": 'A nature-loving ally, harnessing the power of plants and offering healing and protection.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Luna': {
        "description": 'A mysterious and enchanting friend, with powers linked to the moon and the night sky.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Ember': {
        "description": 'A fiery spirit, imbuing your steps with passion and strength to overcome challenges.',
        "relationship": 'positive',
        "item": 'Note'
    },
    'Jasper': {
        "description": 'A steadfast and loyal companion, offering unwavering support and unwritten bravery.',
        "relationship": 'positive',
        "item": 'Note'
    }
}

enemy_metadata = {
    "Thornback": {
        "description": "A formidable adversary with razor-sharp thorns that can pierce any surface.",
        "health": 100,
        "attack": 10,
        "defense": 5,
        "xp": 25
    },
    "Frostbite": {
        "description": "A chilling opponent that freezes everything it touches, turning the ground to ice.",
        "health": 300,
        "attack": 30,
        "defense": 25,
        "xp": 25
    },
    "Viperstrike": {
        "description": "A chilling opponent that freezes everything it touches, turning the ground to ice.",
        "health": 200,
        "attack": 50,
        "defense": 5,
        "xp": 25
    },
    "Shadowclaw": {
        "description": "A stealthy and elusive adversary, capable of disappearing into the darkness.",
        "health": 200,
        "attack": 20,
        "defense": 15
    },
    "Venomfang": {
        "description": "A deadly enemy with a venomous bite, leaving a trail of poison in its wake.",
        "health": 250,
        "attack": 25,
        "defense": 20
    },
    "Frostbite": {
        "description": "A chilling opponent that freezes everything it touches, turning the ground to ice.",
        "health": 300,
        "attack": 30,
        "defense": 25
    },
    "Blazewing": {
        "description": "A fiery adversary, soaring through the air, leaving trails of scorching flames.",
        "health": 350,
        "attack": 35,
        "defense": 30
    },
    "Scorchscale": {
        "description": "A fearsome dragon-like enemy, breathing scorching hot flames that melt anything in its path.",
        "health": 400,
        "attack": 40,
        "defense": 35
    },
    "Gloomspike": {
        "description": "A dark and gloomy foe, striking with spikes that drain the energy from its targets.",
        "health": 450,
        "attack": 45,
        "defense": 40
    },
    "Stormrider": {
        "description": "A tempestuous enemy, conjuring powerful storms and unleashing destructive lightning bolts.",
        "health": 500,
        "attack": 50,
        "defense": 45
    },
    "Bouldercrush": {
        "description": "A massive and mighty opponent, capable of crushing anything with its colossal strength.",
        "health": 550,
        "attack": 55,
        "defense": 50
    }
}

action_metadata = {
    "path": {
        "description": "A traversable tile that allows the player to move freely.",
        "interactions": []
    },
    "wall": {
        "description": "An impassable tile that blocks the player's movement.",
        "interactions": []
    },
    "puddle": {
        "description": "A slippery tile that slows the player's movement.",
        "interactions": []
    },
    "mud": {
        "description": "A sticky tile that slows the player's movement.",
        "interactions": []
    },
    "spring": {
        "description": "A tile that launches the player into the air.",
        "interactions": [
            "jump"
        ]
    },
    "teleport": {
        "description": "A tile that teleports the player to a new location.",
        "interactions": []
    },
    "trampoline": {
        "description": "A tile that bounces the player into the air.",
        "interactions": [
            "jump"
        ]
    },
    "cloud": {
        "description": "A floating tile that allows the player to move freely.",
        "interactions": []
    },
    "ladder": {
        "description": "A tile that allows the player to climb up or down.",
        "interactions": []
    }
}

# import random
# import torch

# def fill_tensor(tensor, tile_list):
#   for i in range(len(tensor)):
#     if i >= len(tile_list):
#       break
#     assert isinstance(tile_list[i][0], int), 'Tile type must be an int'
#     tensor[i] = tile_list[i][0]

#   return tensor


# def main():
#     tile_list = []

#     for tile_type, metadata in zip(tile_types.values(), [enemy_metadata, item_metadata, friend_metadata]):
#         for name, data in metadata.items():
#             tile_list.append((tile_type, name, data))
#     print(f"tile list: {tile_list}",end='\n')

#     tensor = torch.empty(len(tile_list), 1)
#     filled_tensor = fill_tensor(tensor, tile_list)

#     print(filled_tensor)


# if __name__ == "__main__":
#   main()


