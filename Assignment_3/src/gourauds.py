from helpers import *
from lighting import *


def shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, X, lighting):
    # triangle filling function where the inner points of the triangle get the
    # RGB values that result from the linear interpolation of the RGB values of
    # its vertices (first vertically and then horizontally)
    # - X: MxNx3 image (perhaps) with pre-existing triangles
    # - vertices: 3x2 matrix containing in each row the 2D coordinates of one of the
    # triangle's vertices
    # - vcolors: 3x3 matrix containing in each row the color of one of the vertices in
    # RGB form and with values in the spectrum [0, 1]
    # - updatedcanvas: MxNx3 matrix containing for each point of the triangle (vertices
    # and inner points) the calculated RGB values as well as the pre-existing triangles
    # of the input canvas covering possible common colored points with the pre-existing
    # triangles

    for i in range(verts_c.shape[0]):
        verts_c[:, i] = light(bcoords, verts_n[:, i].T, verts_c[:, i].T, cam_pos, mat, lights, light_amb, lighting).T[0]

    vertices = verts_p.T
    vcolors = verts_c.T

    # initialize updatedcanvas as canvas
    updatedcanvas = X

    # check if there are fewer than 3 distinct vertices and a triangle cannot be formed
    if len(np.unique(vertices, axis=0)) < 3:
        return updatedcanvas

    # define the minimum and the maximum y of the triangle
    y_max = vertices[:, 1].max()
    y_min = vertices[:, 1].min()

    # initialize edge objects
    edgeAB = Edge(np.array([vertices[0, :], vertices[1, :]]), np.array([vcolors[0, :], vcolors[1, :]]), None,
                  slope(vertices[0, :], vertices[1, :]), False)
    edgeBC = Edge(np.array([vertices[1, :], vertices[2, :]]), np.array([vcolors[1, :], vcolors[2, :]]), None,
                  slope(vertices[1, :], vertices[2, :]), False)
    edgeCA = Edge(np.array([vertices[2, :], vertices[0, :]]), np.array([vcolors[2, :], vcolors[0, :]]), None,
                  slope(vertices[2, :], vertices[0, :]), False)

    # initialize parameters that will be used for the scanning
    active_edges = []
    edges = [edgeAB, edgeBC, edgeCA]
    horizontal_line = False

    # find the active edges list of the scanning line y == y_min
    for i in range(len(edges)):

        if y_min == edges[i].y_min:

            if edges[i].edge_slope != 0:
                active_edges.append(i)
                edges[i].is_active = True
            else:
                horizontal_line = True

    # check if there are not enough active edges
    if len(active_edges) < 2:
        return updatedcanvas

    # check the condition where y = y_min and do the appropriate filling
    if not horizontal_line:
        x1 = x2 = 0

        for i in range(len(vertices)):
            if vertices[i, 1] == y_min:
                x1 = vertices[i, 0]
                x2 = x1

    else:
        x1, y1, x2, y2, color1, color2 = initialize_variables(edges, active_edges)

    # filling algorith (first scan every row and then scan every column)
    for y in range(y_min + 1, y_max + 1):

        if edges[active_edges[0]].edge_slope != float('inf'):
            x1 = x1 + 1 / edges[active_edges[0]].edge_slope

        if edges[active_edges[1]].edge_slope != float('inf'):
            x2 = x2 + 1 / edges[active_edges[1]].edge_slope

        colorA = interpolate_vectors(edges[active_edges[0]].vertices[0],
                                     edges[active_edges[0]].vertices[1],
                                     edges[active_edges[0]].vcolors[0, :],
                                     edges[active_edges[0]].vcolors[1, :],
                                     y, 2)
        colorB = interpolate_vectors(edges[active_edges[1]].vertices[0],
                                     edges[active_edges[1]].vertices[1],
                                     edges[active_edges[1]].vcolors[0, :],
                                     edges[active_edges[1]].vcolors[1, :],
                                     y, 2)

        for x in range(int(min(x1, x2)), int(max(x1, x2)) + 1):
            updatedcanvas[y, int(round(x))] = np.clip(interpolate_vectors([x1, y], [x2, y],
                                                                          colorA, colorB,
                                                                          x, 1), 0, 1)
        if y == y_max:
            break

        active_edges = update_active_edges(edges, active_edges, y)

    return updatedcanvas
