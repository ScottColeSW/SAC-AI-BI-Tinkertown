import random
import pygame
import torch

tile_types = {
  0: "path",
  1: "enemy",
  2: "item",
  3: "wall",
  4: "puddle",
  5: "mud",
  6: "obstacle",
  7: "spring",
  8: "teleport",
  9: "trampoline",
  10: "cloud",
  11: "ladder"
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

enemy = {
    'Thornback': 'A formidable adversary with razor-sharp thorns that can pierce any surface.',
    'Viperstrike': 'A cunning serpent-like foe, striking with lightning-fast venomous attacks.',
    'Shadowclaw': 'A stealthy and elusive adversary, capable of disappearing into the darkness.',
    'Venomfang': 'A deadly enemy with a venomous bite, leaving a trail of poison in its wake.',
    'Frostbite': 'A chilling opponent that freezes everything it touches, turning the ground to ice.',
    'Blazewing': 'A fiery adversary, soaring through the air, leaving trails of scorching flames.',
    'Scorchscale': 'A fearsome dragon-like enemy, breathing scorching hot flames that melt anything in its path.',
    'Gloomspike': 'A dark and gloomy foe, striking with spikes that drain the energy from its targets.',
    'Stormrider': 'A tempestuous enemy, conjuring powerful storms and unleashing destructive lightning bolts.',
    'Bouldercrush': 'A massive and mighty opponent, capable of crushing anything with its colossal strength.'
}

enemy[0] = "Thornback"
enemy[1] = "Viperstrike"
enemy[2] = "Shadowclaw"
enemy[3] = "Venomfang"
enemy[4] = "Frostbite"
enemy[5] = "Blazewing"
enemy[6] = "Scorchscale"
enemy[7] = "Gloomspike"
enemy[8] = "Stormrider"
enemy[9] = "Bouldercrush"

enemy["metadata"] = {
  "Thornback": {
    "description": "A formidable adversary with razor-sharp thorns that can pierce any surface.",
    "health": 100,
    "attack": 10,
    "defense": 5
  },
  "Viperstrike": {
    "description": "A cunning serpent-like foe, striking with lightning-fast venomous attacks.",
    "health": 150,
    "attack": 15,
    "defense": 10
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

friend = {
    'Willow': 'A wise and serene companion, guiding you with her deep knowledge of the world.',
    'Harmony': 'A harmonious friend, emanating peaceful vibes and offering valuable advice.',
    'Blaze': 'A passionate ally, igniting the spirit and bolstering your courage in the face of adversity.',
    'Aurora': 'A radiant friend, illuminating your path with her gentle glow and granting clarity.',
    'Zephyr': 'A playful companion, bringing refreshing winds and swift movements to aid you.',
    'Nova': 'A celestial friend, bestowing cosmic powers and lighting up the darkest of moments.',
    'Ivy': 'A nature-loving ally, harnessing the power of plants and offering healing and protection.',
    'Luna': 'A mysterious and enchanting friend, with powers linked to the moon and the night sky.',
    'Ember': 'A fiery spirit, imbuing your steps with passion and strength to overcome challenges.',
    'Jasper': 'A steadfast and loyal companion, offering unwavering support and unwritten bravery.'
}

difficulty_types = {
  0: "normal",
  1: "fast",
  2: "slow",
  3: "powerful",
  4: "agile"
}

def generate_level_data(width, height, empty_percentage, item_probability, difficulty_level):
  """Generates a level data tensor with the specified width, height, empty percentage, item probability, and difficulty level."""
  level_data = torch.randint(0, len(tile_list), (width, height))
  for i in range(width):
    for j in range(height):
      if random.random() < empty_percentage:
        level_data[i, j] = "path"
      elif random.random() > item_probability:
        difficulty_modifier = 1 if difficulty_level == "normal" else 2
        random_item_type = random.randint(0, len(item_types) - 1)
        level_data[i, j] = item_types[random_item_type] * difficulty_modifier
      else:
        random_tile_type = random.randint(0, len(tile_list) - 1)
        level_data[i, j] = tile_list[random_tile_type]

  return level_data

# Define the size of the tilemap
map_width = 10
map_height = 10

# Generate an empty tilemap
tilemap = [[0 for _ in range(map_width)] for _ in range(map_height)]

# Generate random tiles for the tilemap
for y in range(map_height):
    for x in range(map_width):
        # Choose a random tile type
        tile_type = random.choice(list(tile_types.keys()))
        tilemap[y][x] = tile_type

# Print the generated tilemap
for row in tilemap:
    print(row)


def play_game(level_data):
  """Plays the game using the specified level data."""
  player = pygame.Rect(0, 0, 10, 10)

  # Start the game loop
  while True:
    # Get the player's input
    keys = pygame.key.get_pressed()

    # Move the player
    if keys[pygame.K_LEFT]:
      player.x -= 10
    elif keys[pygame.K_RIGHT]:
      player.x += 10
    elif keys[pygame.K_UP]:
      player.y -= 10
    elif keys[pygame.K_DOWN]:
      player.y += 10

    # Check if the player has collided with a square
    for i in range(level_data.shape[0]):
      for j in range(level_data.shape[1]):
        if player.colliderect(level_data[i, j]):
          # Get the type of the square
          square_type = level_data[i, j].item()

          # Do something based on the type of the square
          if square_type == "enemy":
            print("You encountered an enemy!")
          elif square_type == "item":
            print("You found an item!")
          elif square_type == "path":
            print("You moved to a new location!")

    # Render the game
    pygame.draw(screen, player.color, player)
    for i in range(level_data.shape[0]):
      for j in range(level_data.shape[1]):
        pygame.draw(screen, level_data[i, j].color, level_data[i, j])

    # Update the screen
    pygame.display.update()


#level_data = torch.randint(0, len(enemy), (5, 5))

level_data = generate_level_data(5, 5, .2, .6, "normal")

play_game(level_data)