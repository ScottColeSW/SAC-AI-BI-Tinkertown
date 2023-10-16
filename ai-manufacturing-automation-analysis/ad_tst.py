import arcade
import arcade.gui
import random
import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600 

class Player(arcade.Sprite):

    def __init__(self, image_file, scale):
        super().__init__(image_file, scale=scale)

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

        # Check for out of bounds
        if self.left < 0:
            self.left = 0
        elif self.right > SCREEN_WIDTH - 1:
            self.right = SCREEN_WIDTH - 1

        if self.bottom < 0:
            self.bottom = 0
        elif self.top > SCREEN_HEIGHT - 1:
            self.top = SCREEN_HEIGHT - 1

class AI(arcade.Sprite):

    def __init__(self, image_file):
        super().__init__(image_file)
        self.path = None
        self.waypoint_index = 0

    def update(self):
        if self.path and self.waypoint_index < len(self.path):
            target = self.path[self.waypoint_index]
            if self.center_x < target[0]:
                self.change_x = 1
            elif self.center_x > target[0]:
                self.change_x = -1
            else:
                self.change_x = 0
            
            if self.center_y < target[1]:
                self.change_y = 1 
            elif self.center_y > target[1]:
                self.change_y = -1
            else:
                self.change_y = 0

            distance = math.sqrt((self.center_x - target[0]) ** 2 + (self.center_y - target[1]) ** 2)
            if distance < 2:
                self.center_x = target[0]
                self.center_y = target[1]
                self.waypoint_index += 1
        else:
            self.change_x = 0
            self.change_y = 0

        self.center_x += self.change_x
        self.center_y += self.change_y

class Star(arcade.Sprite):
    pass

class Hazard(arcade.Sprite):
    pass


# and so on for stars, hazards, etc
class GameOverView(arcade.View):

    def on_show(self, dialog_message_text):
        arcade.set_background_color(arcade.color.RED)

        # Create a UI button
        self.button = arcade.gui.UIFlatButton(
            center_x=SCREEN_WIDTH // 2,
            center_y=SCREEN_HEIGHT // 2,
            width=200,
            height=50,
            text=dialog_message_text)

        # Set the button's function to restart the game
        self.button.on_click = self.restart_game

    def on_draw(self):
        arcade.start_render()
        self.button.draw()

    def restart_game(self):
        game = AdventureGame()
        game.setup()
        self.window.show_view(game)

class AdventureGame(arcade.Window):

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Joby Adventure")
        self.player = None
        self.ai = None
        self.maze = None
        self.stars = None
        self.hazards = None
        self.health = 0
        self.score = 0
        self.timer = 360
        self.goal = 1
        self.goal_image = arcade.Sprite("sprites/goal.png", scale=0.5)
        self.goal_width = 25
        self.goal_height = 25

    # def setup(self):
    #     self.player = Player("sprites/joby-passenger.png", scale=.25)
    #     self.player.center_x = 64
    #     self.player.center_y = 64
    #     self.ai = AI()
    #     self.maze = self.generate_maze()
    #     self.stars = self.generate_stars()
    #     self.hazards = self.generate_hazards()
    #     self.health = 3
    #     self.ai.path = self.generate_ai_path()

    def setup(self):
        self.maze = self.generate_fixed_maze()
        self.player = self.create_player()
        self.ai = self.create_ai() 
        self.stars = self.generate_stars()
        self.hazards = self.generate_hazards()
        self.goal = self.create_goal()
        
    def create_goal(self):
        goal = arcade.Sprite("sprites/goal.png", scale=0.5)
        goal.center_x = SCREEN_WIDTH - self.goal_width / 2
        goal.center_y = SCREEN_HEIGHT - self.goal_height / 2
        return goal

    def create_player(self):
        player = Player("sprites/joby-passenger.png", scale=.25)
        player.center_x = 3
        player.center_y = SCREEN_HEIGHT - 3
        return player
    
    def create_ai(self):
        ai = AI("sprites/joby-ai-avatar.png")
        ai.path = self.generate_ai_path(self.maze)
        return ai

    def reached_goal(self):
        return arcade.check_for_collision(self.player, self.goal_image)

    def has_health(self):
        return self.health > 0

    def has_stars(self):
        return len(self.stars) > 0
        
    def time_ran_out(self):
        return self.timer <= 0

    def generate_fixed_maze(self):
        # maze = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        #         [1,0,0,0,1,1,1,0,0,0,1,1,1,0,1],
        #         [1,1,1,0,0,0,0,0,1,0,0,0,0,0,1],
        #         [1,0,1,0,1,1,1,1,1,1,1,1,1,0,1],
        #         [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],  
        #         [1,0,1,1,1,1,1,1,1,0,1,1,1,1,1],
        #         [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
        #         [1,0,1,0,1,0,1,1,1,1,1,1,1,0,1],
        #         [1,0,0,0,1,0,0,0,0,1,0,0,1,0,1],
        #         [1,1,1,1,1,1,1,0,0,1,0,1,1,0,1],
        #         [1,0,0,0,0,0,1,1,0,1,0,0,0,0,1],  
        #         [1,0,1,1,1,0,1,0,0,0,1,1,1,0,1],
        #         [1,0,0,0,1,0,1,0,1,0,1,0,0,0,1],
        #         [1,1,1,0,1,0,0,0,1,0,0,0,1,0,1],
        #         [1,1,1,0,0,0,1,1,1,1,1,1,1,0,1]]
        maze = [[1 for x in range(20)] for y in range(15)]

        return maze

    def generate_ai_path(self, maze):

        def heuristic(cell, goal):
            x1, y1 = cell
            x2, y2 = goal
            return abs(x1 - x2) + abs(y1 - y2)

        def get_neighbors(cell, maze):
            x, y = cell
            neighbors = []
            
            # Check North
            if y < len(maze) - 1 and maze[y+1][x] == 0:
                neighbors.append((x, y+1))
                
            # Check South        
            if y > 0 and maze[y-1][x] == 0:
                neighbors.append((x, y-1))
                
            # Check West
            if x > 0 and maze[y][x-1] == 0:
                neighbors.append((x-1, y))
                
            # Check East        
            if x < len(maze[0]) - 1 and maze[y][x+1] == 0:
                neighbors.append((x+1, y))

            return neighbors

        path = []
        
        # Set start and goal locations
        start = (1, 1)
        goal = (SCREEN_WIDTH - 32, 32)
        
        # Create a dictionary to store g, f, and parent values for each cell
        scores = {}
        
        # Initialize g and f scores for the start cell
        scores[start] = {'g':0, 'f':heuristic(start, goal)}
        
        # Initialize open and closed lists
        openlist = [start]
        closedlist = []
        
        # Loop until open list is empty
        while len(openlist) > 0:

            # Get current cell with lowest f score
            current = min(openlist, key=lambda x: scores[x]['f'])
            
            # Remove current from open list and add to closed list
            openlist.remove(current)
            closedlist.append(current)
            
            # Check if we reached the goal
            if current == goal:
                # Construct path by working backwards from goal
                while current != start:
                    path.insert(0, current)
                    current = scores[current]['parent']
                return path
                
            # Expand search to current's neighbors
            neighbors = get_neighbors(current, self.maze)
            for neighbor in neighbors:
                if neighbor in closedlist:
                    continue
                    
                # Calculate g score for this neighbor
                g = scores[current]['g'] + 1
                
                # Check if we already evaluated this cell
                if neighbor not in openlist:
                    openlist.append(neighbor)
                    
                # Update neighbor's g and f scores
                scores[neighbor] = {'g': g, 'f': g + heuristic(neighbor, goal), 'parent': current}

        # No path found
        return None
        
    def generate_stars(self):
        stars = arcade.SpriteList()

        for i in range(10):
            star = arcade.Sprite("sprites/star.png", scale=.25) 
            star.center_x = random.randint(1, 8)*64 + 32
            star.center_y = random.randint(1, 8)*64 + 32
            if star not in stars:
                stars.append(star)

        return stars

    def generate_hazards(self):
        hazards = arcade.SpriteList()

        for i in range(5):
            hazard = arcade.Sprite("sprites/hazard.png", scale=.25)
            hazard.center_x = random.randint(1, 8)*64 + 32
            hazard.center_y = random.randint(1, 8)*64 + 32
            if hazard not in hazards:
                hazards.append(hazard)

        return hazards

    def on_draw(self):
        arcade.start_render()
        
        # Draw maze
        for row in range(10):
            for col in range(10):
                if self.maze[row][col] == 1:
                    arcade.draw_rectangle_filled(col * 64, row * 64, 64, 64, arcade.color.DARK_BLUE)

        self.stars.draw()
        self.hazards.draw()
        self.player.draw()
        self.ai.draw()

        arcade.draw_text(f"Health: {self.health}", 10, 10, arcade.color.WHITE, 14)
        arcade.draw_text(f"Score: {self.score}", 10, 30, arcade.color.WHITE, 14)

        # Draw goal
        # x = SCREEN_WIDTH - self.goal_width / 2
        # y = self.goal_height / 2
        # arcade.draw_texture_rectangle(x, y, self.goal_width, self.goal_height, self.goal_image)

        x = SCREEN_WIDTH - self.goal_width / 2
        y = SCREEN_HEIGHT - self.goal_height / 2
        arcade.draw_texture_rectangle(x, y, self.goal_width, self.goal_height, self.goal_image)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.player.change_x = -3
        elif key == arcade.key.RIGHT: 
            self.player.change_x = 3
        elif key == arcade.key.UP:
            self.player.change_y = 3
        elif key == arcade.key.DOWN:
            self.player.change_y = -3

    def on_key_release(self, key, modifiers):
        if key in [arcade.key.LEFT, arcade.key.RIGHT]:
            self.player.change_x = 0
        elif key in [arcade.key.UP, arcade.key.DOWN]:
            self.player.change_y = 0

    # Handle clicking button to restart
    def update(self, delta_time):
        self.player.update()
        self.ai.update()

        #goal_collision = arcade.check_for_collision(self.player, self.goal)
        goal_collision = self.reached_goal()

        if goal_collision:
            if self.has_health() and self.has_stars():
                view = GameOverView("Winner!")
                self.show_view(view)
            else:
                # Handle other game over conditions
                view = GameOverView("Try Again?")
                self.show_view(view)

        if goal_collision:
            if self.player.health > 0 and self.stars_collected > 0:
                game_over_condition = 'success'
            else:
                game_over_condition = 'failure'

            if game_over_condition:
            #     view = GameOverView("Try Again?")
            #     self.show_view(view)
            # if game_over_condition:
                view = GameOverView()
                self.show_view(view)

        # Check for sprite collisions
        star_hit = arcade.check_for_collision_with_list(self.player, self.stars)
        hazard_hit = arcade.check_for_collision_with_list(self.player, self.hazards)

        for star in star_hit:
            star.remove_from_sprite_lists()
            self.score += 1

        for hazard in hazard_hit:
            hazard.remove_from_sprite_lists()
            self.score -= 0.5

        if self.health <= 0:
            #arcade.close_window()
            return GameOverView("End Game?")

class GameOverView(arcade.View):

    def on_show(self, dialog_message_text):
        arcade.set_background_color(arcade.color.RED)

        # Create a UI button
        self.button = arcade.gui.UIFlatButton(
            center_x=SCREEN_WIDTH // 2,
            center_y=SCREEN_HEIGHT // 2,
            width=200,
            height=50,
            text=dialog_message_text)

        # Set the button's function to restart the game
        self.button.on_click = self.restart_game

    def on_draw(self):
        arcade.start_render()
        self.button.draw()

    def restart_game(self):
        game = AdventureGame()
        game.setup()
        self.window.show_view(game)



# class GameOverView(arcade.View, text=dialog_message_text):

#     def on_show(self, dialog_message_text):
#         # arcade.set_background_color(arcade.color.RED)
        
#         # # Show game over text and button  
#         arcade.set_background_color(arcade.color.RED)
        
#         # Create a UI button
#         self.button = arcade.gui.UIFlatButton(
#             center_x=SCREEN_WIDTH // 2,
#             center_y=SCREEN_HEIGHT // 2,
#             width=200,
#             height=50,
#             text=dialog_message_text)

#         # Set the button's function to restart the game
#         self.button.on_click = self.restart_game

#     def on_draw(self):
#         arcade.start_render()
#         self.button.draw() 

#     def restart_game(self):
#         game = AdventureGame()
#         game.setup()
#         self.window.show_view(game)

if __name__ == "__main__":
    game = AdventureGame()
    game.setup()
    arcade.run()