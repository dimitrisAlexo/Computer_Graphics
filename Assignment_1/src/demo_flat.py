import time
import matplotlib.pyplot as plt
from render import *

# load data
data = np.load('h1.npy', allow_pickle=True)[()]

verts2d = data['verts2d']
vcolors = data['vcolors']
faces = data['faces']
depth = data['depth']
shade_t = "flat"

start_time = time.time()
img = render(verts2d, faces, vcolors, depth, shade_t)
print(time.time() - start_time)

plt.imshow(img)
plt.show()
plt.imsave("flat_fish.png", img)
