import arcade
import arcade.gui

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class MenuView(arcade.View):
    def __init__(self):
        super().__init__()
        self.button = arcade.gui.UIFlatButton(text="Start Game", width=200, height=40)

    def on_show(self):
       x = SCREEN_WIDTH / 2
       y = SCREEN_HEIGHT / 2  
       self.button.set_position(x, y)
       
    def on_draw(self):
        arcade.start_render()
        self.button.draw()
        
class GameWindow(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)

    def setup(self):
        menu_view = MenuView()
        self.show_view(menu_view)

if __name__ == "__main__":
    window = GameWindow() 
    window.setup()
    arcade.Run()