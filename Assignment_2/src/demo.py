from render import *
import matplotlib.pyplot as plt
import time

data = np.load("h2.npy", allow_pickle=True)[()]

# 3D coordinates of all the vertices of the triangles
p3d = np.array(data['verts3d'])
# Indices of each triangle vertices from pd3
faces = np.array(data['faces'])
# The RGB values for each vertex
vcolors = np.array(data['vcolors'])
# Unit vector indicating the direction of the rotation axis
u = np.array(data['u'])
# The pointing coordinates of the camera
ck = np.array(data['c_lookat'])
# The up-vector of the camera
cup = np.array(data['c_up'])
# The camera vector perpendicular to the up vector
cv = np.array(data['c_org'])
# Translation displacement vectors
t1, t2 = np.array(data['t_1']), np.array(data['t_2'])
# Angle of rotation in radians
phi = np.array(data['phi'])

img_h = img_w = 512
cam_h = cam_w = 15
f = 70

# Convert to float32 for better performance
p3d, faces, vcolors = np.array(np.array(p3d), dtype=np.float32), np.array(np.array(faces), dtype=np.int32),\
                      np.array(np.array(vcolors), dtype=np.float32)

start_time = time.time()

img = render_object(p3d, faces, vcolors, cam_h, cam_w, img_h, img_w, f, cv, ck, cup)
img = np.clip(img, 0, 1)
plt.imsave("0.jpg", img, origin='lower')

A = np.array([0, 0, 0])
p3d = rotate_translate(p3d, 0, u, A, t1)
img = render_object(p3d, faces, vcolors, cam_h, cam_w, img_h, img_w, f, cv, ck, cup)
img = np.clip(img, 0, 1)
plt.imsave("1.jpg", img, origin='lower')

p3d = rotate_translate(p3d, phi, u, A, np.array([0, 0, 0]))
img = render_object(p3d, faces, vcolors, cam_h, cam_w, img_h, img_w, f, cv, ck, cup)
img = np.clip(img, 0, 1)
plt.imsave("2.jpg", img, origin='lower')

p3d = rotate_translate(p3d, 0, u, A, t2)
img = render_object(p3d, faces, vcolors, cam_h, cam_w, img_h, img_w, f, cv, ck, cup)
img = np.clip(img, 0, 1)
plt.imsave("3.jpg", img, origin='lower')

print("Objects rendered in", time.time() - start_time)
