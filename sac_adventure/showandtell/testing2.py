import json
import pygame
import random
import networkx as nx

# Define the dimensions of the tilemap
TILE_SIZE = 32
MAP_WIDTH = 10
MAP_HEIGHT = 8

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data

def has_multiple_paths(graph, start, end):
    num_paths = 0

    while num_paths < 1:
        for path in nx.all_simple_paths(graph, start, end):
            num_paths += 1

            if num_paths > 1:
                return True
            else:
                graph = generate_graph(int(random.int)) 

    return False


def generate_level(level_n):
    # Create an empty tilemap
    tilemap = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]

    # Place random obstacles and enemies
    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            if random.random() < 0.2:  # Adjust the probability as desired
                tilemap[row][col] = 1  # 1 represents an obstacle tile
            elif random.random() < 0.1:  # Adjust the probability as desired
                tilemap[row][col] = 2  # 2 represents an enemy tile

    # Add places to jump (e.g., platforms)
    for col in range(MAP_WIDTH):
        if random.random() < 0.3:  # Adjust the probability as desired
            row = random.randint(1, MAP_HEIGHT - 2)
            tilemap[row][col] = 3  # 3 represents a platform tile

    # Save the tilemap as a JSON file
    with open(f'level{level_n}.json', 'w') as file:
        json.dump(tilemap, file)    


def generate_graph(level_map):
    graph = nx.Graph()

    rows = len(level_map)
    cols = len(level_map[0])

    for row in range(rows):
        for col in range(cols):
            if level_map[row][col] == 0:
                graph.add_node((row, col))

    for row in range(rows):
        for col in range(cols):
            if level_map[row][col] == 0:
                neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                for neighbor_row, neighbor_col in neighbors:
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        if level_map[neighbor_row][neighbor_col] == 0:
                            graph.add_edge((row, col), (neighbor_row, neighbor_col))

    return graph

def render_level(level_data):
    tile_size = 32
    level_width = len(level_data[0]) * tile_size
    level_height = len(level_data) * tile_size

    screen = pygame.display.set_mode((level_width, level_height))

    for row in range(len(level_data)):
        for col in range(len(level_data[row])):
            tile_value = level_data[row][col]
            tile_x = col * tile_size
            tile_y = row * tile_size

            if tile_value == 0:
                pygame.draw.rect(screen, (0, 0, 0), (tile_x, tile_y, tile_size, tile_size))
            elif tile_value == 1:
                pygame.draw.rect(screen, (255, 255, 255), (tile_x, tile_y, tile_size, tile_size))
            elif tile_value == 2:
                pygame.draw.rect(screen, (255, 0, 0), (tile_x, tile_y, tile_size, tile_size))
            elif tile_value == 3:
                pygame.draw.rect(screen, (0, 255, 0), (tile_x, tile_y, tile_size, tile_size))

    pygame.display.update()


def main():
    pygame.init()


    window_width = 600
    window_height = 600
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Sole Caliber")

    splash_screen_image = pygame.image.load("sole_caliber.gif")
    splash_screen_rect = splash_screen_image.get_rect()

    window.blit(splash_screen_image, splash_screen_rect)
    pygame.display.flip()

    # screen = pygame.display.set_mode((window_width, window_height))
    # pygame.display.set_caption("Sole Quest - Sole Mate Rescue")

    # center = screen.get_rect().center
    # screen.blit(pygame.Surface((window_width, window_height)), center)
    # pygame.display.flip()

    # new_window_width = int(pygame.display.Info().current_w * 0.65)
    # new_window_height = int(pygame.display.Info().current_h * 0.65)

    # new_window = pygame.display.set_mode((new_window_width, new_window_height))
    # pygame.display.set_caption("Sole Quest - Sole Window")

    # new_window_center = new_window.get_rect().center
    # new_window.blit(pygame.Surface((new_window_width, new_window_height)), new_window_center)

    # pygame.display.flip()


    # window_width = 1536
    # window_height = 864
    # window = pygame.display.set_mode((window_width, window_height))
    # pygame.display.set_caption("Sole Caliber")

    # splash_screen_image = pygame.image.load("sole_caliber.gif")
    # splash_screen_rect = splash_screen_image.get_rect()

    # window.blit(splash_screen_image, splash_screen_rect)
    # pygame.display.flip()




    # Set up the game window
    # window_width = 1536
    # window_height = 864
    # window = pygame.display.set_mode((window_width, window_height))
    # pygame.display.set_caption("Sole Caliber")

    # Set up the splash screen
    splash_screen_image = pygame.image.load("sole_caliber.gif")  # Replace "splash_screen.jpg" with your actual image file
    splash_screen_rect = splash_screen_image.get_rect()

    # Display the splash screen
    window.blit(splash_screen_image, splash_screen_rect)
    pygame.display.flip()


    # screen = pygame.display.set_mode((window_width, window_height))

    # pygame.display.set_caption("Sole Quest - Sole Mate Rescue")

    # # Get the center of the screen
    # center = screen.get_rect().center

    # # Set the position of the window to the center of the screen
    # screen.blit(pygame.Surface((window_width, window_height)), center)

    # pygame.display.flip()

    # # Open a new dialog window in the middle of the display
    # new_window_width = int(pygame.display.Info().current_w * 0.65)
    # new_window_height = int(pygame.display.Info().current_h * 0.65)

    # new_window = pygame.display.set_mode((new_window_width, new_window_height))
    # pygame.display.set_caption("Sole Quest - Sole Window")


    # new_window = pygame.display.set_mode((new_window_width, new_window_height))
    # new_window = pygame.display.set_caption("Sole Quest - Sole Window")

    # new_window_center = new_window.get_rect().center

    # new_window.blit(pygame.Surface((new_window_width, new_window_height)), new_window_center)

    # pygame.display.flip()

    i = 1
    while i < 12:
        level_map = load_level_map(f'level{i}.json')
        graph = generate_graph(level_map)
        print(graph.nodes)
        print(graph.edges)
        i += 1

        # Render the level and wait for a key event
        render_level(level_map)
        pygame.time.wait(1000)  # Wait for 1 second before moving to the next level

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()         


if __name__ == "__main__":
    main()