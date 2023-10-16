import json
import pygame
import random
import text_assets as ta

def load_level(filename):
    with open(filename) as f:
        level_data = json.load(f)

    return level_data

def draw_level(level, play_field):
    # Initializing Color
    color = (255,0,0)

    for row in level:
        for col in row:
            color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            if col == 0 and row == 0:
                color = (0, 0, 0) #black
            
            # Drawing Rectangle
            # A rect style object is a tuple with four elements: the x-coordinate, the y-coordinate, the width, and the height.
            rect = pygame.Rect(col, row, 10, 10)
            rect.update((col, row))

            # Drawing rectangle
            pygame.draw.rect(play_field, color, rect)
            pygame.display.flip()

def main():
    pygame.init()
    surface = pygame.display.set_mode((640, 480))

    level = load_level("level1.json")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        surface.fill((0, 0, 0))
        draw_level(level, surface)
        pygame.display.flip()
        pygame.time.wait(1000) 

if __name__ == "__main__":
    main()