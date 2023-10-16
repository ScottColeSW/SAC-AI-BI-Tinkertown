import pygame

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





    # pygame.init()

    # window_width = 800
    # window_height = 600

    # window = pygame.display.set_mode((window_width, window_height))
    # pygame.display.set_caption("Sole Caliber")
    # window.fill((0, 0, 0))

    # pygame.time.wait(1000)

    # dialog_width = int(pygame.display.Info().current_w * 0.65)
    # dialog_height = int(pygame.display.Info().current_h * 0.65)

    # dialog = pygame.display.set_mode((dialog_width, dialog_height))
    # pygame.display.set_caption("Sole Quest - Dialog Window")

    # dialog_center = pygame.mouse.get_pos()
    # dialog.move(dialog_center[0] - dialog_width / 2, dialog_center[1] - dialog_height / 2)

    # dialog.blit(pygame.Surface((dialog_width, dialog_height)), dialog_center)

    # pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

#    pygame.quit()

if __name__ == "__main__":
    main()
