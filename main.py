from game_app import GameApp

# The GameApp class contains all the logic for the game
# app.run() starts the game loop
# The config.json file contains configuration settings for the game
app = GameApp()
app.run()

if app.config.mpl_debug:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm
            import numpy as np

            # Grayscale map
            # ========================================================================
            plt.rcParams.update({'font.size': 18})  # Set global font size
            data = -app.game_map.val_data[:,:,0]
            vmin, vmax = -3, 4
            bounds = np.arange(vmin, vmax + 2) - 0.5  # boundaries for each integer
            # Use a discrete colormap with enough colors
            cmap = plt.get_cmap('gray', vmax - vmin + 1)

            colors = cmap(np.arange(cmap.N))  # alle Farben     #-----
            cmap_new = plt.cm.colors.ListedColormap(colors[1:-1]) #-------
            bounds_new = bounds[1:-1] # cut bounds for new cmap 
            norm = BoundaryNorm(bounds_new, cmap_new.N)

            plt.imshow(data, cmap=cmap_new, norm=norm, interpolation='nearest')
            ticks = np.arange(-2,4)
            custom_labels=["Wall", "Wall Periphery", "Valid", "Hole area", "Hole center", "Checkpoint"]
            cbar = plt.colorbar(ticks=ticks, shrink=0.666)
            cbar.ax.set_yticklabels(custom_labels)
            plt.title("Game Map")
            plt.tight_layout()
            plt.show()
            # ========================================================================

            # Vector field
            # ========================================================================
            # Downsample for clarity (optional, otherwise plot will be very dense)
            plot_data = app.game_map.val_data
            step = 1  # plot every 10th pixel
            Y, X = np.mgrid[0:plot_data.shape[0]:step, 0:plot_data.shape[1]:step]
            U = plot_data[::step, ::step, 2]  # x-component
            V = plot_data[::step, ::step, 1]  # y-component

            plt.figure(figsize=(10, 10))
            plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
            plt.gca().invert_yaxis()  # To match image coordinates
            plt.title("Gradient Field")
            plt.axis('equal')
            plt.show()

            # ========================================================================

        except ImportError:
            print("Matplotlib not installed. Cannot display debug plots.")