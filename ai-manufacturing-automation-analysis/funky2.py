import arcade
import arcade.gui
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class MazeObject(arcade.Sprite):
    def __init__(self, image, scale=1):
        super().__init__(image, scale)
        self.is_collidable = True

class Star(MazeObject):
    def on_collision(self):
        self.game.score += 1
        self.remove_from_sprite_lists()

class Hazard(MazeObject):
    def on_collision(self):
        self.game.health -= 1
        self.remove_from_sprite_lists()

class Wall(arcade.Sprite):
    def __init__(self, x, y, scale=1):
        super().__init__()
        
        # Sprite properties
        self.texture = arcade.load_texture("sprites/wall.png", scale)
        self.scale = scale
        self.is_collidable = True
        
        # Position 
        self.center_x = x * 64 + 32
        self.center_y = y * 64 + 32

        # Color
        self.color = arcade.color.GRAY

    def check_collision(self, sprite):
        """Check for collision with another sprite"""
        if arcade.check_for_collision(self, sprite):
            sprite.on_wall_collision()
            
    def on_wall_collision(self):
        """Override this method in child classes"""
        pass

class Player(arcade.Sprite):
    def __init__(self, image, scale):
        super().__init__(image, scale)
    
    def on_star_collision(self):
        self.game.score += 1
        self.remove_from_sprite_lists()
        
    def on_hazard_collision(self):
        self.game.health -= 1
        self.remove_from_sprite_lists()

class AI(arcade.Sprite):
    def __init__(self, image):
        super().__init__(image)
        self.path = None

    def setup_path(self, path):
        self.path = path
        self.waypoint_index = 0
        
# Similar classes for Star, Wall, Goal        

class AdventureGame(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        self.game_over = False
        
    def setup(self):
        self.player = Player("sprites/joby-passenger.png", 0.25)
        self.ai = AI("sprites/joby-ai-avatar.png")
        self.maze = self.create_maze()
        self.stars = arcade.SpriteList()
        self.hazards = arcade.SpriteList()
        self.ui_manager = arcade.gui.UIManager()

        self.score = 0
        self.health = 3

        # Set the initial position of the player to the top-left corner
        self.player.center_x = 64  # Adjust as needed
        self.player.center_y = SCREEN_HEIGHT - 64  # Adjust as needed

        # Set the initial position of the AI to the top-left corner
        self.ai.center_x = 64  # Adjust as needed
        self.ai.center_y = SCREEN_HEIGHT - 64  # Adjust as needed

        self.player.game = self
        self.ai.game = self

        # Call create_maze() to generate the maze walls
        self.create_maze()
        
        # Create stars and hazards
        self.create_stars(5)  # Adjust the number of stars as needed
        self.create_hazards(5)  # Adjust the number of hazards as needed

        # Generate AI path (replace with a proper pathfinding algorithm)
        self.ai.setup_path(self.generate_path())        

    def generate_path(self):
        # Create pathfinding algorithm
        return []
    
    def create_maze_objects(self, num_objects, object_class, image):
        objects = arcade.SpriteList()
        for _ in range(num_objects):
            obj = object_class(image, 0.25)
            obj.center_x = random.randint(0, SCREEN_WIDTH)
            obj.center_y = random.randint(0, SCREEN_HEIGHT)
            objects.append(obj)
        return objects

    def create_stars(self, num_stars):
        self.stars = self.create_maze_objects(num_stars, Star, "sprites/star.png")

    def create_hazards(self, num_hazards):
        self.hazards = self.create_maze_objects(num_hazards, Hazard, "sprites/hazard.png")

    def create_maze(self):
        self.walls = arcade.SpriteList()

        # Define the maze layout using a 2D list of 1s (walls) and 0s (empty cells)
        maze_layout = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]

        # Create walls based on maze layout
        cell_size = 64
        for y, row in enumerate(maze_layout):
            for x, cell in enumerate(row):
                if cell == 1:
                    wall = Wall(x * cell_size + cell_size / 2, y * cell_size + cell_size / 2, 1)
                    self.walls.append(wall)

    def create_stars(self, num_stars):
        self.stars = arcade.SpriteList()
        for _ in range(num_stars):
            star = Star("sprites/star.png", 0.25)
            star.center_x = random.randint(0, SCREEN_WIDTH)
            star.center_y = random.randint(0, SCREEN_HEIGHT)
            self.stars.append(star)

    def create_hazards(self, num_hazards):
        self.hazards = arcade.SpriteList()
        for _ in range(num_hazards):
            hazard = Hazard("sprites/hazard.png", 0.25)
            hazard.center_x = random.randint(0, SCREEN_WIDTH)
            hazard.center_y = random.randint(0, SCREEN_HEIGHT)
            self.hazards.append(hazard)
            
    def on_draw(self):
        self.clear()
        self.walls.draw()
        self.stars.draw()
        self.player.draw()
        self.ai.draw()
     
    def update(self, delta_time):
        self.player.update()
        self.ai.update()
        
        if self.health <= 0:
            self.game_over = True
            self.show_gameover("Game Over")
        
    def show_gameover(self, text):
        view = GameOverView(text)
        self.show_view(view)
        
class GameOverView(arcade.View):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def on_show(self):
        self.button = arcade.gui.UIFlatButton(text="Restart", on_click=self.on_click)

    def on_click(self, event):
        game_view = AdventureGame()
        game_view.setup()
        self.window.show_view(game_view)

    def on_draw(self):
        arcade.draw_text(self.text, SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 
                        anchor_x="center")
        self.button.draw()
        
window = AdventureGame()
window.setup() 
arcade.run()