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