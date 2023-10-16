import matplotlib.pyplot as plt
import time
from render import *

# Load data from file
data = np.load('h3.npy', allow_pickle=True)[()]
verts = data['verts']
vert_colors = data['vertex_colors'].T
faces = data['face_indices'].T
eye = data['cam_eye']
lookat = data['cam_lookat']
up = data['cam_up']
focal = data['focal']
k_a = data['ka']
k_d = data['kd']
k_s = data['ks']
n = data['n']
light_positions = data['light_positions']
light_intensities = data['light_intensities']
Ia = data['Ia'].T[0]
M = data['M']
N = data['N']
W = data['W']
H = data['H']
bg_color = data['bg_color'].T[0]

num_of_lights = len(light_positions)
lights = np.empty(num_of_lights, dtype=PointLight)

# Construct objects
mat = PhongMaterial(k_a, k_d, k_s, n)
for i in range(num_of_lights):
    lights[i] = PointLight(np.array([light_positions[i]]), np.array([light_intensities[i]]))

start_time = time.time()

img = render_object("gouraud", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "ambient")
plt.imsave('0.jpg', np.array(img[::-1]))

img = render_object("gouraud", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "diffusion")
plt.imsave('1.jpg', np.array(img[::-1]))

img = render_object("gouraud", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "specular")
plt.imsave('2.jpg', np.array(img[::-1]))

img = render_object("gouraud", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "full")
plt.imsave('3.jpg', np.array(img[::-1]))

img = render_object("phong", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "ambient")
plt.imsave('4.jpg', np.array(img[::-1]))

img = render_object("phong", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "diffusion")
plt.imsave('5.jpg', np.array(img[::-1]))

img = render_object("phong", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "specular")
plt.imsave('6.jpg', np.array(img[::-1]))

img = render_object("phong", focal, eye, lookat, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, mat, lights, Ia, "full")
plt.imsave('7.jpg', np.array(img[::-1]))

print("Objects rendered in", time.time() - start_time, "sec")
