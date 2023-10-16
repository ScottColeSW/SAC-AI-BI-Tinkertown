# import random
# import torch

# tile_list = [
#   "path",
#   "enemy",
#   "item",
#   "wall",
#   "puddle",
#   "mud",
#   "obstacle",
#   "spring",
#   "teleport",
#   "trampoline",
#   "cloud",
#   "ladder"
# ]

# item_types = {
#   "speed": 0,
#   "slow": 1,
#   "jump": 2,
#   "kick": 3,
#   "stomp": 4,
#   "slide": 5,
#   "shield": 6,
#   "double_jump": 7,
#   "invincibility": 8,
#   "health": 9,
#   "extra_life": 10
# }

# difficulty_types = {
#   "normal": 0,
#   "fast": 1,
#   "slow": 2,
#   "powerful": 3,
#   "agile": 4
# }

# enemy = {
#     'Thornback': 'A formidable adversary with razor-sharp thorns that can pierce any surface.',
#     'Viperstrike': 'A cunning serpent-like foe, striking with lightning-fast venomous attacks.',
#     'Shadowclaw': 'A stealthy and elusive adversary, capable of disappearing into the darkness.',
#     'Venomfang': 'A deadly enemy with a venomous bite, leaving a trail of poison in its wake.',
#     'Frostbite': 'A chilling opponent that freezes everything it touches, turning the ground to ice.',
#     'Blazewing': 'A fiery adversary, soaring through the air, leaving trails of scorching flames.',
#     'Scorchscale': 'A fearsome dragon-like enemy, breathing scorching hot flames that melt anything in its path.',
#     'Gloomspike': 'A dark and gloomy foe, striking with spikes that drain the energy from its targets.',
#     'Stormrider': 'A tempestuous enemy, conjuring powerful storms and unleashing destructive lightning bolts.',
#     'Bouldercrush': 'A massive and mighty opponent, capable of crushing anything with its colossal strength.'
# }

# friend = {
#     'Willow': 'A wise and serene companion, guiding you with her deep knowledge of the world.',
#     'Harmony': 'A harmonious friend, emanating peaceful vibes and offering valuable advice.',
#     'Blaze': 'A passionate ally, igniting the spirit and bolstering your courage in the face of adversity.',
#     'Aurora': 'A radiant friend, illuminating your path with her gentle glow and granting clarity.',
#     'Zephyr': 'A playful companion, bringing refreshing winds and swift movements to aid you.',
#     'Nova': 'A celestial friend, bestowing cosmic powers and lighting up the darkest of moments.',
#     'Ivy': 'A nature-loving ally, harnessing the power of plants and offering healing and protection.',
#     'Luna': 'A mysterious and enchanting friend, with powers linked to the moon and the night sky.',
#     'Ember': 'A fiery spirit, imbuing your steps with passion and strength to overcome challenges.',
#     'Jasper': 'A steadfast and loyal companion, offering unwavering support and unwritten bravery.'
# }


# def generate_level_data(width, height, empty_percentage, item_probability):
#   """Generates a level data tensor with the specified width, height, empty percentage, and item probability."""
#   level_data = torch.zeros((width, height))
#   for i in range(width):
#     for j in range(height):
#       if random.random() < empty_percentage:
#         level_data[i, j] = "path"
#       elif random.random() > item_probability:
#         level_data[i, j] = "item"
#         random_item_type = random.randint(0, len(item_types) - 1)
#         level_data[i, j] = item_types[random_item_type]
#       else:
#         level_data[i, j] = tile_list[random.randint(0, len(tile_list) - 1)]
#   return level_data

# print(generate_level_data(10,10,.7,.8))


# # tile_types = {
# #     'path': 0,
# #     'wall': 3,
# #     'puddle': 4,
# #     'mud': 5,
# #     'obstacle': 6,
# #     'spring': 7,
# #     'teleport': 8,
# #     'trampoline': 9,
# #     'cloud': 10,
# #     'ladder': 11,

# #     'item': 2 : { 
# #         'fast' : 0,
# #         'slow' : 1,
# #         'jump' : 2,
# #         'kick' : 3,
# #         'stomp': 4, 
# #         'slide': 5
# #     },

# #     'difficulty': 12 : {
# #         'normal': 0,
# #         'fast': 1,
# #         'slow': 2,
# #         'powerful': 3,
# #         'agile': 4
# #     },

# #     'item': 13 : { 
# #         'speed' : 0,
# #         'slow' : 1,
# #         'jump' : 2,
# #         'kick' : 3,
# #         'stomp': 4, 
# #         'slide': 5,
# #         'shield': 6,
# #         'double_jump': 7,
# #         'invincibility': 8,
# #         'health': 9,
# #         'extra_life': 10
# #     }
# # }



# # tile_types = {
# #     'path': 0,
# #     'enemy': 1,
# #     'item': 2 : { 
# #         'speed' : 0,
# #         'slow' : 1,
# #         'jump' : 2,
# #         'kick' : 3,
# #         'stomp': 4, 
# #         'slide': 5
# #     },
# #     'wall': 3
# # }

# # tile_types = {
# #     'path': 0,
# #     'enemy': 1,
# #     'item': 2 : { 
# #         'speed' : 0,
# #         'slow' : 1,
# #         'jump' : 2,
# #         'kick' : 3,
# #         'stomp': 4, 
# #         'slide': 5
# #     },
# #     'wall': 3,
# #     'puddle': 4,
# #     'mud': 5,
# #     'obstacle': 6,
# #     'spring': 7,
# #     'teleport': 8,
# #     'trampoline': 9,
# #     'cloud': 10,
# #     'ladder': 11
# # }

# # tile_types = {
# #     'path': 0,
# #     'enemy': 1 : {
# #         'normal': 0,
# #         'fast': 1,
# #         'slow': 2,
# #         'powerful': 3,
# #         'agile': 4
# #     },
# #     'item': 2 : { 
# #         'speed' : 0,
# #         'slow' : 1,
# #         'jump' : 2,
# #         'kick' : 3,
# #         'stomp': 4, 
# #         'slide': 5,
# #         'shield': 6,
# #         'double_jump': 7,
# #         'invincibility': 8,
# #         'health': 9,
# #         'extra_life': 10
# #     },
# #     'wall': 3,
# #     'puddle': 4,
# #     'mud': 5,
# #     'obstacle': 6,
# #     'spring': 7,
# #     'teleport': 8,
# #     'trampoline': 9,
# #     'cloud': 10,
# #     'ladder': 11
# #     'enemy': 12: {
# #         'Thornback': 'A formidable adversary with razor-sharp thorns that can pierce any surface.',
# #         'Viperstrike': 'A cunning serpent-like foe, striking with lightning-fast venomous attacks.',
# #         'Shadowclaw': 'A stealthy and elusive adversary, capable of disappearing into the darkness.',
# #         'Venomfang': 'A deadly enemy with a venomous bite, leaving a trail of poison in its wake.',
# #         'Frostbite': 'A chilling opponent that freezes everything it touches, turning the ground to ice.',
# #         'Blazewing': 'A fiery adversary, soaring through the air, leaving trails of scorching flames.',
# #         'Scorchscale': 'A fearsome dragon-like enemy, breathing scorching hot flames that melt anything in its path.',
# #         'Gloomspike': 'A dark and gloomy foe, striking with spikes that drain the energy from its targets.',
# #         'Stormrider': 'A tempestuous enemy, conjuring powerful storms and unleashing destructive lightning bolts.',
# #         'Bouldercrush': 'A massive and mighty opponent, capable of crushing anything with its colossal strength.'
# #     },
# #     'friend': 13: {
# #         'Willow': 'A wise and serene companion, guiding you with her deep knowledge of the world.',
# #         'Harmony': 'A harmonious friend, emanating peaceful vibes and offering valuable advice.',
# #         'Blaze': 'A passionate ally, igniting the spirit and bolstering your courage in the face of adversity.',
# #         'Aurora': 'A radiant friend, illuminating your path with her gentle glow and granting clarity.',
# #         'Zephyr': 'A playful companion, bringing refreshing winds and swift movements to aid you.',
# #         'Nova': 'A celestial friend, bestowing cosmic powers and lighting up the darkest of moments.',
# #         'Ivy': 'A nature-loving ally, harnessing the power of plants and offering healing and protection.',
# #         'Luna': 'A mysterious and enchanting friend, with powers linked to the moon and the night sky.',
# #         'Ember': 'A fiery spirit, imbuing your steps with passion and strength to overcome challenges.',
# #         'Jasper': 'A steadfast and loyal companion, offering unwavering support and unwritten bravery.'
# #     }
# # }