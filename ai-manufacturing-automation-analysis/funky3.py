import arcade
import arcade.gui

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class MainMenuView(arcade.View):
    def on_show(self):
        self.ui_manager = arcade.gui.UIManager()
#        self.ui_manager.set_viewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        #self.ui_manager.window.viewport.set(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        #self.ui_manager.set_viewport(360, 180)

        button_width = 100
        button_height = 50
        x = SCREEN_WIDTH // 2 - button_width // 2
        y = SCREEN_HEIGHT // 2 - button_height // 2

        ai_button = arcade.gui.UIFlatButton(
            center_x=x,
            center_y=y + 50,
            width=button_width,
            height=button_height,
            text="AI Run",
            on_click=self.start_ai_run
        )
        self.ui_manager.add(ai_button)

        player_button = arcade.gui.UIFlatButton(
            center_x=x,
            center_y=y - 50,
            width=button_width,
            height=button_height,
            text="Player Run",
            on_click=self.start_player_run
        )
        self.ui_manager.add(player_button)

        self.ui_manager.draw()
        #.set_viewport(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)

    def start_ai_run(self, event):
        # Start AI run
        pass

    def start_player_run(self, event):
        # Start Player run
        pass

    def on_draw(self):
        arcade.start_render()
        self.ui_manager.draw()

    def on_hide_view(self):
        self.ui_manager.unregister_handlers()

    def on_mouse_press(self, x, y, button, modifiers):
        self.ui_manager.process_mouse_event(x, y, button, modifiers)

    def on_key_press(self, symbol, modifiers):
        self.ui_manager.process_key_event(symbol, modifiers)

class AdventureGame(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        self.show_view(MainMenuView())
        left = 0
        width = SCREEN_WIDTH
        height = SCREEN_HEIGHT

        right = left + width
        bottom = left 
        top = bottom + height

        #self.set_viewport(left, right, bottom, top)
        self.set_viewport(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)

def main():
    window = AdventureGame()
    window.setup()
    arcade.run()

if __name__ == "__main__":
    main()