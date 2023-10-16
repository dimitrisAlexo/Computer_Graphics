from flats import *
from gourauds import *
from projection import *


def shade_triangle(canvas, vertices, vcolors, shade_t):
    # calls for the respective triangle shading function depending on
    # the shade_t = {"flat", "gouraud"} variable

    if shade_t == "flat":
        updatedcanvas = flats(canvas, vertices, vcolors)
    elif shade_t == "gouraud":
        updatedcanvas = gourauds(canvas, vertices, vcolors)
    else:
        raise ValueError("Invalid value for shade_t. Must be 'flat' or 'gouraud'.")

    return updatedcanvas


def render(verts2d, faces, vcolors, depth, shade_t):
    # renders the final image
    # - img: colored image of dimensions MxNx3 containing K colored triangles forming
    # a projection of a 3D image onto the 2D plane
    # - verts2d: Lx2 matrix containing the coordinates of L vertices of K triangles
    # - faces: Kx3 matrix containing the vertices of the K triangles as a reference to
    # the respective coordinates of verts2d
    # - vcolors: Lx3 matrix containing the color values (r, g, b) for each one of the L
    # triangle vertices
    # - depth: Lx1 matrix containing the depth of each vertex
    # - shade_t: string {"flat", "gouraud"} deciding the coloring function
    # - M, N: height and width of the canvas

    # check if shade_t is of accepted value
    assert shade_t in ["flat", "gouraud"]

    # set canvas dimensions
    M = N = 512

    # set white background
    img = np.ones((M, N, 3))

    # compute the average depth of each triangle
    triangle_depth = depth[faces].mean(axis=1)  # Kx1

    # sort the triangles by depth in descending order
    sorted_triangles = triangle_depth.argsort()[::-1].tolist()  # Kx1

    for triangle in sorted_triangles:
        indices = faces[triangle]
        triangle_vertices = np.array(verts2d[indices])
        triangle_vcolors = np.array(vcolors[indices])
        img = shade_triangle(img, triangle_vertices, triangle_vcolors, shade_t)
    return img


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


def render_object(p3d, faces, vcolors, H, W, rows, cols, f, cv, ck, cup):

    # Renders the 3D object onto the 2D plane.
    # - p3d: 3*N numpy array with the 3D coordinates of points in the WCS
    # - faces: Kx3 matrix containing the vertices of the K triangles as a reference to
    # the respective coordinates of verts2d
    # - vcolors: Lx3 matrix containing the color values (r, g, b) for each one of the L
    # triangle vertices
    # - H, W: height and width of the camera plane
    # - rows: number of rows in the image n2d
    # - cols: number of columns in the image n2d
    # - f: focal length of the camera
    # - cv: 3*1 numpy array with the 3D coordinates of the pinhole camera's center with respect to the WCS' s origin
    # - ck: 3*1 numpy array with the 3D coordinates of the target point K of the camera
    # - cup: the unit up-vector
    # - img: image with the rendered object

    p2d, depth = camera_looking_at(f, cv, ck, cup, p3d)
    n2d = rasterize(p2d, rows, cols, H, W).astype(int)

    img = render(n2d, faces, vcolors, depth, "gouraud")

    return img
