from game_app import GameApp

# The GameApp class contains all the logic for the game
# app.run() starts the game loop
# The config.json file contains configuration settings for the game
app = GameApp()
app.run()

if app.config.mpl_debug:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(-app.game_map.val_data[:,:,0], cmap='gray', vmin=-3, vmax=4)
            plt.title("Game Map val_data")
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Cannot display debug plots.")