import numpy as np


# define the Edge object
class Edge:
    def __init__(self, vertices, vcolors, normals, edge_slope, is_active):
        self.vertices = vertices
        self.vcolors = vcolors
        self.normals = normals
        self.edge_slope = edge_slope
        self.is_active = is_active
        self.y_min = min(vertices[0, 1], vertices[1, 1])
        self.y_max = max(vertices[0, 1], vertices[1, 1])


class PhongMaterial:
    # defines the properties of a material on the 3D surface in order to be displayed by the Phong shading.
    # k_a: ambient light coefficient (float)
    # k_d: diffuse reflection coefficient (float)
    # k_s: specular reflection coefficient (float)
    # n_phong: Phong constant (int)

    def __init__(self, k_a, k_d, k_s, n_phong):
        self.k_a = k_a
        self.k_d = k_d
        self.k_s = k_s
        self.n_phong = n_phong


class PointLight:
    # constructs a point light source
    # pos: 1*3 vector with the position of the source in the 3D space
    # intensity: 1*3 vector with the intensity for each color (rgb) as a float in [0, 1]

    def __init__(self, pos, intensity):
        self.pos = pos
        self.intensity = intensity


def calculate_normals(verts, faces):
    # calculates the normal vectors of the surface for each vertex of each triangle (we have N_t triangles)
    # - verts: 3*N_v matrix containing the coordinates of the vertices of the object
    # - faces: 3*N_t matrix describing the triangles; the k-th column of faces contains
    # the serial numbers of the vertices of the k-th triangle of the object, 1 ≤ k ≤ NT;
    # the order of juxtaposition of the vertices marks by the right-handed screw rule the
    # direction of the normal vector and therefore also in which direction is the outer side of the object.
    # - normals: 3*N_v matrix with normal vectors for each vertex

    num_verts = verts.shape[1]
    normals = np.zeros((3, num_verts))

    for i in range(faces.shape[1]):
        v1_idx = faces[0, i]
        v2_idx = faces[1, i]
        v3_idx = faces[2, i]

        v1 = verts[:, v1_idx]
        v2 = verts[:, v2_idx]
        v3 = verts[:, v3_idx]

        normal = np.cross(v2 - v1, v3 - v1)

        normals[:, v1_idx] += normal
        normals[:, v2_idx] += normal
        normals[:, v3_idx] += normal

    normals = normals / np.linalg.norm(normals, axis=0)

    return normals


def interpolate_vectors(p1, p2, V1, V2, xy, dim):
    # calculates the value V of a vector in coordinates p = (x,y) by interpolating
    # two vectors with values V1 and V2 with respective coordinates p1 = (x1,y1) and p2 = (x2,y2)
    # - V: return value after interpolation between V1 and V2
    # - p1,p2: 2D coordinates of the points in which V1, V2 correspond
    # - V1,V2: values in p1 and p2 points respectively
    # - xy: the x or y value of a point p depending on whether dim = 1 or dim = 2 respectively

    if dim == 1:
        if (p2[0] - p1[0]) == 0:
            return V1
        t = (xy - p1[0]) / (p2[0] - p1[0])
        V = V1 + t * np.subtract(V2, V1)
    elif dim == 2:
        if (p2[1] - p1[1]) == 0:
            return V1
        t = (xy - p1[1]) / (p2[1] - p1[1])
        V = V1 + t * np.subtract(V2, V1)
    else:
        raise ValueError("Invalid value for dim")
    return V


def slope(a, b):
    # calculates the slope of the line connecting the two points

    np.seterr(divide='ignore')
    if a[0] == b[0]:
        return np.inf
    else:
        return (b[1] - a[1]) / (b[0] - a[0])


def initialize_variables(edges, active_edges):
    # finds the initial values (when y = y_min) for the points of the vertices of the horizontal line, as
    # well as their colors
    # - edges: list of the edges of the triangle
    # - active_edges: list of the active edges of the triangle in that particular scanline

    if edges[active_edges[0]].vertices[0, 1] == edges[active_edges[0]].y_max:
        x1 = edges[active_edges[0]].vertices[1, 0]
        y1 = edges[active_edges[0]].vertices[1, 1]
        color1 = edges[active_edges[0]].vcolors[1, :]
    else:
        x1 = edges[active_edges[0]].vertices[0, 0]
        y1 = edges[active_edges[0]].vertices[0, 1]
        color1 = edges[active_edges[0]].vcolors[0, :]

    if edges[active_edges[1]].vertices[0, 1] == edges[active_edges[1]].y_max:
        x2 = edges[active_edges[1]].vertices[1, 0]
        y2 = edges[active_edges[1]].vertices[1, 1]
        color2 = edges[active_edges[1]].vcolors[1, :]
    else:
        x2 = edges[active_edges[1]].vertices[0, 0]
        y2 = edges[active_edges[1]].vertices[0, 1]
        color2 = edges[active_edges[1]].vcolors[0, :]

    return x1, y1, x2, y2, color1, color2


def update_active_edges(edges, active_edges, y):
    # updates the active edges after every iteration (scanning on the y-axis)
    # - edges: list of the edges of the triangle
    # - active_edges: list of the active edges of the triangle in that particular scanline
    # - y: current y-coordinate

    if y == edges[active_edges[0]].y_max:

        for i in range(len(edges)):

            if y == edges[i].y_min:
                active_edges[0] = i

    elif y == edges[active_edges[1]].y_max:

        for i in range(len(edges)):

            if y == edges[i].y_min:
                active_edges[1] = i

    return active_edges
