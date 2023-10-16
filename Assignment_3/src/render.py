from gourauds import *
from projection import *
from phong import *


def rasterize(p2d, rows, cols, H, W):
    # Depicts the coordinates of the Camera system with plane dimensions of H*W onto pixel positions of an image with
    # dimensions of rows*cols. The axis of the camera passes through the center of the orthogonal H*W, while the image
    # indexing starts from down to up and from left to right.
    # - p2d: 2*N numpy array with the 2D coordinates after projection from p3d
    # - rows: number of rows in the image n2d
    # - cols: number of columns in the image n2d
    # - H, W: height and width of the camera plane

    # Create an empty n2d array of size rows*cols
    n2d = np.zeros((p2d.shape[1], 2))

    # Calculate the scaling factors for mapping p2d coordinates to pixel positions
    width = cols / W
    height = rows / H

    # Iterate over each projected 2D coordinate and map it to the corresponding pixel position
    for i in range(p2d.shape[1]):
        n2d[i, 0] = np.around((p2d[0, i] + H / 2) * height + 0.5)
        n2d[i, 1] = np.around((-p2d[1, i] + W / 2) * width + 0.5)

    return n2d


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts,
                  vert_colors, faces, mat, lights, light_amb, lighting):
    # renders an object made of a specific material, placed in a scene with light sources and a camera; calculates how
    # light is reflected onto the object and its final color at each point.
    # - shader: string {"gouraud", "phong"} deciding the coloring function
    # - focal: the distance of the projection from the centre of the camera measured in the units used by the camera
    # coordinate system
    # - eye: 3 × 1 vector with the coordinates of the centre of the camera
    # - lookat: 3 × 1 vector with the coordinates of the camera target point
    # - up:  3 × 1 unit "up" vector of the camera
    # - bg_color: 3 × 1 vector with the colour components of the background
    # - M, N: height, width of the generated image in pixels
    # - H, W: physical height and width of the camera lens in the units used by the camera coordinate system
    # - verts: 3 × N_v matrix with the coordinates of the vertices of the object
    # - vert_colors: 3 × N_v matrix with the colour components of each vertex of the object
    # - faces: 3*N_t (number of triangles) matrix describing the triangles; the k-th column of faces contains
    # the serial numbers of the vertices of the k-th triangle of the object, 1 ≤ k ≤ NT;
    # the order of juxtaposition of the vertices marks by the right-handed screw rule the
    # direction of the normal vector and therefore also in which direction is the outer side of the object
    # mat: object of type PhongMaterial
    # - lights: list of objects of type PointLight
    # - light_amb: 3 × 1 vector with the components of the ambient radiation intensity in the interval [0, 1]
    # - img: the image with the rendered object

    assert shader in ["gouraud", "phong"]

    # Calculate normals for each vertex of each triangle
    normals = calculate_normals(verts, faces.T)

    # Project vertices onto the camera plane
    verts_projected, depth = camera_looking_at(focal, eye, lookat, up, verts)

    # Rasterize the projected vertices
    verts2d = rasterize(verts_projected, M, N, H, W).astype(int)

    # Initialize image
    image_shape = (M, N, 3)
    img = np.full(image_shape, bg_color)

    # Average depth of every triangle
    depth_order = np.mean(depth[faces], axis=1)

    # Sort triangles by depth
    sorted_triangles = np.flip(np.argsort(depth_order))

    for triangle in sorted_triangles:
        triangle_vertices_indices = faces[triangle]
        triangle_verts2d = verts2d[triangle_vertices_indices].T
        triangle_vcolors = vert_colors[triangle_vertices_indices].T
        bcoords = np.mean(verts[:, triangle_vertices_indices], axis=0).T

        if shader == "gouraud":
            img = shade_gouraud(triangle_verts2d, normals[:, triangle_vertices_indices], triangle_vcolors,
                                bcoords, eye, mat, lights, light_amb, img, lighting)
        elif shader == "phong":
            img = shade_phong(triangle_verts2d, normals[:, triangle_vertices_indices], triangle_vcolors,
                              bcoords, eye, mat, lights, light_amb, img, lighting)
    return img
