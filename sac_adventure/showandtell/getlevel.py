import json
import pygame

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
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

pygame.init()

i=1
events = pygame.event.get()
while i < 12:   
    level_map = load_level_map(f'level{i}.json')
    render_level(level_map)        
    if events[0].type == pygame.QUIT:
            break
    i = i+1
    pygame.display.flip()
    for _ in 100000000:
        _=_+1