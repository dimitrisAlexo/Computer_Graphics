from flats import *
from gourauds import *


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
