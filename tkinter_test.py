import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "objects.dat"
RADIUS = 10
HEIGHT = 480
WIDTH = 800

objects = []

with open(FILENAME, "r") as f:
    for line in f:
        objects.append(line.strip().split("\t"))

# objects.append(['r', 104, 72, 152, 143])
# objects.append(['r', 405, 82, 458, 179])
# objects.append(['r', 343, 179, 511, 212])
# objects.append(['r', 314, 248, 502, 292])
# objects.append(['r', 502, 248, 539, 365])
# objects.append(['o', 104, 295, 162, 353])

# 2D np array
data = np.zeros((HEIGHT, WIDTH, 3))

# Fill area of objects
for obj in objects:
    if obj[0] == 'r':
        # Rectangle coordinates: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        data[y1:y2, x1:x2, 0] = 2 # pixle occupied

    elif obj[0] == 'o':
        # Oval "corner" coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        # Create a mask for the oval
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        mask = ((X - (x1 + x2) / 2) ** 2) / ((x2 - x1) / 2) ** 2 + ((Y - (y1 + y2) / 2) ** 2) / ((y2 - y1) / 2) ** 2 <= 1
        # Set the pixels inside the oval to 1
        data[mask, 0] = 3 # pixle marked as hole

# Fill area of shifted rectangles
for obj in objects:
    if obj[0] == 'r':
        # Rectangle coordinates: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        # Create rectangle above
        subdata = data[y1-RADIUS:y1, x1:x2]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0])
        data[y1-RADIUS:y1, x1:x2] = subdata
        # Create rectangle below
        subdata = data[y2:y2+RADIUS, x1:x2]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0])
        data[y2:y2+RADIUS, x1:x2] = subdata
        # Create rectangle left
        subdata = data[y1:y2, x1-RADIUS:x1]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1])
        data[y1:y2, x1-RADIUS:x1] = subdata
        # Create rectangle right
        subdata = data[y1:y2, x2:x2+RADIUS]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1])
        data[y1:y2, x2:x2+RADIUS] = subdata

# Create circle 
circle = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
Y, X = np.ogrid[:2*RADIUS + 1, :2*RADIUS + 1]
mask = ((X - RADIUS) ** 2) + ((Y - RADIUS) ** 2) <= RADIUS ** 2
circle[mask] = 1 # pixle occupied
idx = np.indices((2*RADIUS + 1, 2*RADIUS + 1))
circle = np.stack((circle, (idx[0] - RADIUS) * circle / RADIUS, (idx[1]- RADIUS) * circle / RADIUS), axis=-1)

# Fill cirle areas at the corners
for obj in objects:
    if obj[0] == 'r':
        # Rectangle coordinates: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

        # Insert circle segments
        # Top left corner
        mask = data[y1-RADIUS:y1, x1-RADIUS:x1, 0] == 0
        data[y1-RADIUS:y1, x1-RADIUS:x1][mask] = circle[:RADIUS, :RADIUS][mask]
        # Top right corner
        mask = data[y1-RADIUS:y1, x2:x2+RADIUS, 0] == 0
        data[y1-RADIUS:y1, x2:x2+RADIUS][mask] = circle[:RADIUS, RADIUS + 1:][mask]
        # Bottom left corner
        mask = data[y2:y2+RADIUS, x1-RADIUS:x1, 0] == 0
        data[y2:y2+RADIUS, x1-RADIUS:x1][mask] = circle[RADIUS + 1:, :RADIUS][mask]
        # Bottom right corner
        mask = data[y2:y2+RADIUS, x2:x2+RADIUS, 0] == 0
        data[y2:y2+RADIUS, x2:x2+RADIUS][mask] = circle[RADIUS + 1:, RADIUS + 1:][mask]


        
# print(data[60:75, 95:110, 1])
# print(data[60:75, 95:110, 2])


plt.imshow(2 - data[:,:,0], cmap='gray', vmin=0, vmax=2)
plt.show()

plt.imshow(np.arctan2(data[:,:,1], data[:,:,2]), cmap='hsv', vmin=-np.pi, vmax=np.pi)
plt.show()

plt.imshow(1 - data[:,:,1])
plt.show()
plt.imshow(1 - data[:,:,2])
plt.show()

# Downsample for clarity (optional, otherwise plot will be very dense)
step = 1  # plot every 10th pixel
Y, X = np.mgrid[0:data.shape[0]:step, 0:data.shape[1]:step]
U = data[::step, ::step, 2]  # x-component
V = data[::step, ::step, 1]  # y-component

plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
plt.gca().invert_yaxis()  # To match image coordinates
plt.title("Gradient Field (from data[:,:,1] and data[:,:,2])")
plt.show()

window = tk.Tk()
window.title("Game map")
window.geometry("800x480")

canvas = tk.Canvas(window, width=800, height=480, bg="white")
canvas.pack()

for obj in objects:
    if obj[0] == 'r':
        # Draw a rectangle (x1, y1, x2, y2)
        canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="black", outline="black")
    elif obj[0] == 'o':
        # Draw an oval (x1, y1, x2, y2)
        canvas.create_oval(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="black", outline="black")

# with open("objects.dat", "w") as f:
#     for obj in objects:
#         f.write(f"{obj[0]}\t{obj[1]}\t{obj[2]}\t{obj[3]}\t{obj[4]}\n")

# window.mainloop()



