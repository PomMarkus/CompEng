from game_app import GameApp

# The GameApp class contains all the logic for the game
# app.run() starts the game loop
# The config.json file contains configuration settings for the game
app = GameApp()
app.run()

if app.config.mpl_debug:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.imshow(-app.game_map.val_data[:,:,0], cmap='gray', vmin=-3, vmax=4)
            plt.title("Game Map val_data")
            plt.show()

            # Downsample for clarity (optional, otherwise plot will be very dense)
            plot_data = app.game_map.val_data
            step = 1  # plot every 10th pixel
            Y, X = np.mgrid[0:plot_data.shape[0]:step, 0:plot_data.shape[1]:step]
            U = plot_data[::step, ::step, 2]  # x-component
            V = plot_data[::step, ::step, 1]  # y-component

            plt.figure(figsize=(10, 10))
            plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
            plt.gca().invert_yaxis()  # To match image coordinates
            plt.title("Gradient Field (from data[:,:,1] and data[:,:,2])")
            plt.axis('equal')
            plt.show()

        except ImportError:
            print("Matplotlib not installed. Cannot display debug plots.")