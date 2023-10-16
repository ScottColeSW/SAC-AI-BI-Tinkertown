import pygame
import sys
import json
import random


class PlayerCharacter:
    def __init__(self, x, y):
        self.x = x  # Current x-coordinate of the character
        self.y = y  # Current y-coordinate of the character
        self.velocity_x = 0  # Horizontal velocity of the character
        self.velocity_y = 0  # Vertical velocity of the character
        self.is_jumping = False  # Flag indicating if the character is currently jumping
        self.position = (x,y)
        self.image = pygame.image.load("playershoe.png")  # Replace "player.png" with the actual image file
        self.velocity = pygame.Vector2(0, 0)
        self.speed = 5
        self.is_jumping = False
        self.jump_power = 10
        self.gravity = 0.5

    def update(self):
        # Update the character's position based on the current velocity
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.move()
        self.apply_gravity()

    def jump(self):
        # Apply an upward velocity to initiate a jump
        if not self.is_jumping:
            self.velocity_y = -self.jump_power  # Adjust the jump velocity as needed
            self.is_jumping = True

    def apply_gravity(self):
        # Apply gravity to the character's vertical velocity
        if self.is_jumping:
            self.velocity.y += self.gravity
        else:
            self.velocity.y = 0

    def draw(self, screen):
        screen.blit(self.image, self.position)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.jump()

    def move(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.velocity.x = -self.speed # Adjust the movement speed as needed
            #self.velocity_x = -5  
        elif keys[pygame.K_RIGHT]:
            self.velocity.x = self.speed  # Adjust the movement speed as needed
        else:
            self.velocity.x = 0  # Stop the character's horizontal movement

# Example usage
player = PlayerCharacter(0, (100, 100))

# class PlayerCharacter(Player):
#     def __init__(self, x, y):
#         super().__init__(position=(x, y))
#         self.velocity_x = 0
#         self.velocity_y = 0
#         self.is_jumping = False

#     def update(self):
#         super().update()
#         self.x += self.velocity_x
#         self.y += self.velocity_y

#     def jump(self):
#         super().jump()
#         self.velocity_y = -10

#     def apply_gravity(self):
#         super().apply_gravity()
#         self.velocity_y += 0.5

#     def move_left(self):
#         super().move_left()

#     def move_right(self):
#         super().move_right()

#     def stop(self):
#         super().stop()



class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((640, 480))
        self.player = PlayerCharacter(x=320, y=240)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            self.player.update()
            self.player.draw(self.screen)
            pygame.display.update()

if __name__ == "__main__":
    game = Game()
    game.run()


class NPC:
    def __init__(self, position):
        self.position = position
        self.image = pygame.image.load("npc-froggy.png")  # Replace "npc.png" with the actual image file

    def draw(self, screen):
        screen.blit(self.image, self.position)

class SoleCaliber(NPC):
    def __init__(self, position):
        super().__init__(position)
        self.image = pygame.image.load("sole_caliber.gif")  # Replace "sole_caliber.png" with the actual image file

        # Additional attributes for Sole Caliber
        self.jump_power = 10
        self.dash_speed = 8

    def leap_of_resilience(self):
        # Implement the leap of resilience ability
        pass

    def dash_of_determination(self):
        # Implement the dash of determination ability
        pass

    def stomp_impact(self):
        # Implement the stomp impact ability
        pass

    def play_footstep_sound(self):
        # Play the confident footsteps sound
        pass

    def play_landing_sound(self):
        # Play the powerful landing sound
        pass



# Define the dimensions of the tilemap
TILE_SIZE = 32
MAP_WIDTH = 10
MAP_HEIGHT = 8


def load_level(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


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


# Generate and save 11 levels
i=1
while i < 12:
    generate_level(i)
    i=i+1




# Initialize Pygame
pygame.init()

# Set up the game window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Sole Caliber")

# Set up the splash screen
splash_screen_image = pygame.image.load("sole_caliber.gif")  # Replace "splash_screen.jpg" with your actual image file
splash_screen_rect = splash_screen_image.get_rect()

# Display the splash screen
window.blit(splash_screen_image, splash_screen_rect)
pygame.display.flip()

# Wait for player input to start the game
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            waiting = False

# Game Loop
running = True

# Example usage
player = PlayerCharacter((100, 100))
# Example usage
npc = SoleCaliber((100, 100))

while running:
    player.handle_events()
    player.update()
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update game state

    # Render the game

    # Update the display
    pygame.display.update()

# Quit the game
# pygame.quit()
# sys.exit()





# class NPC2:
#     def __init__(self, name):
#         self.name = name
#         self.description = ""  # NPC description generated by OpenAI
#         self.behaviors = []  # List of NPC behaviors
#         self.gifts = []  # List of gifts the NPC can give to the player
#         self.conversation_enders = []  # List of conversation enders

#     def generate_description(self):
#         pass
#         # Use OpenAI API or other text generation techniques to generate the NPC's description
#         # Set the generated description to self.description

#     def add_behavior(self, behavior):
#         pass
#         # Add a behavior to the NPC's list of behaviors

#     def add_gift(self, gift):
#         # Add a gift to the NPC's list of gifts

#     def add_conversation_ender(self, conversation_ender):
#         # Add a conversation ender to the NPC's list of conversation enders

#     def engage_in_conversation(self):
#         # Define the logic for the NPC's conversation behavior with the player
#         # This could involve randomly selecting responses or considering player choices

#     def perform_action(self):
#         # Define the logic for the NPC's actions in the game
#         # This could involve interacting with the environment, other NPCs, or the player

#     def update(self):
#         # Define any updates or changes to the NPC's state during the game loop
#         # This could involve adjusting behavior based on game events or player interactions

#     def give_gift(self):
#         # Define the logic for the NPC to give a gift to the player
#         # This could involve randomly selecting a gift from the NPC's list of gifts
#         if self.gifts:
#             selected_gift = random.choice(self.gifts)
#             # Award the selected gift to the player or perform relevant actions
#             # ...

#     def end_conversation(self):
#         # Define the logic for the NPC to end the conversation with the player
#         # This could involve randomly selecting an ender from the NPC's list of conversation enders
#         if self.conversation_enders:
#             selected_ender = random.choice(self.conversation_enders)
#             # Display the selected conversation ender or perform relevant actions
#             # ...

#     def assign_quest(self):
#         # Define the logic for the NPC to assign a quest to the player
#         # This could involve creating a new quest object and providing it to the player
#         # ...

#     def trade_items(self, player):
#         # Define the logic for the NPC to trade items with the player
#         # This could involve displaying available items for trade and performing the exchange
#         # ...

#     def provide_information(self):
#         # Define the logic for the NPC to provide information to the player
#         # This could involve displaying hints, tips, or game-related information
#         # ...

#     def unlock_door(self):
#         # Define the logic for the NPC to unlock a door or area for the player
#         # This could involve removing barriers, granting access, or providing keys
#         # ...

#     def heal_player(self):
#         # Define the logic for the NPC to heal the player or restore their resources
#         # This could involve replenishing health, mana, or other game-related resources
#         # ...

#     def offer_upgrade(self):
#         # Define the logic for the NPC to offer upgrades to the player's character
#         # This could involve displaying available upgrades and performing the upgrade process
#         # ...

#     def engage_in_mini_game(self):
#         # Define the logic for the NPC to engage the player in a mini-game or side quest
#         # This could involve presenting unique challenges or gameplay mechanics
#         # ...

#     def join_player_as_companion(self, player):
#         # Define the logic for the NPC to join the player as a companion
#         # This could involve adding the NPC as a companion to the player's party or group
#         # ...

#     def handle_relationship(self, player):
#         # Define the logic for managing the NPC's relationship with the player
#         # This could involve tracking reputation, friendship levels, or alignments
#         # ...

#     def unique_dialogue(self):
#         # Define the logic for the NPC to have unique dialogues or easter eggs
#         # This could involve displaying special dialogues or triggering hidden interactions
#         # ...
