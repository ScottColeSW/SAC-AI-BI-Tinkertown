import json
import pygame
import networkx as nx
import npc

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data


def generate_valid_level_map(level_n):
    while True:
        level_map = generate_level(level_n)  # Generate a new level map
        graph = generate_graph(level_map)  # Generate a graph from the level map
        start = (0, 0)
        end = (len(level_map) - 1, len(level_map[0]) - 1)

        while not has_multiple_paths(graph, start, end):
            level_map = generate_level(level_n)  # Generate a new level map
            graph = generate_graph(level_map)  # Generate a graph from the level map
            start = (0, 0)
            end = (len(level_map) - 1, len(level_map[0]) - 1)

            if (has_multiple_paths(graph, start, end)):
                break

        return level_map  # Return the valid level map


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

pygame.init()

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


def is_valid_graph(graph):
    return True 
    # Implement your validation criteria here...
    # Return True if the graph is valid, False otherwise.


def generate_valid_level_map():
    while True:
        level_map = npc.generate_level(0)  # Generate a new level map
        graph = generate_graph(level_map)  # Generate a graph from the level map

        if is_valid_graph(graph):
            return level_map  # Return the valid level map

# Usage example
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
                elif event.key == pygame.KEYUP:
                    level_map = load_level_map(f'level{i}.json')
                    graph = generate_graph(level_map)

                    if not is_valid_graph(graph):
                        level_map = generate_valid_level_map()  # Generate a new valid level map

                    render_level(level_map)
                    
        pygame.time.wait(1000)
        i += 1
