from helpers import *
from lighting import *


def shade_phong(verts_p, verts_n, verts_c, bcoords, cam_pos, mat, lights, light_amb, X, lighting):
    vertices = verts_p.T
    vcolors = verts_c.T
    normals = verts_n.T

    # initialize updatedcanvas as canvas
    updatedcanvas = X

    # check if there are fewer than 3 distinct vertices and a triangle cannot be formed
    if len(np.unique(vertices, axis=0)) < 3:
        return updatedcanvas

    # define the minimum and the maximum y of the triangle
    y_max = vertices[:, 1].max()
    y_min = vertices[:, 1].min()

    # initialize edge objects
    edgeAB = Edge(np.array([vertices[0, :], vertices[1, :]]), np.array([vcolors[0, :], vcolors[1, :]]),
                  np.array([normals[0, :], normals[1, :]]), slope(vertices[0, :], vertices[1, :]), False)
    edgeBC = Edge(np.array([vertices[1, :], vertices[2, :]]), np.array([vcolors[1, :], vcolors[2, :]]),
                  np.array([normals[1, :], normals[2, :]]), slope(vertices[1, :], vertices[2, :]), False)
    edgeCA = Edge(np.array([vertices[2, :], vertices[0, :]]), np.array([vcolors[2, :], vcolors[0, :]]),
                  np.array([normals[2, :], normals[0, :]]), slope(vertices[2, :], vertices[0, :]), False)

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

        normalA = interpolate_vectors(edges[active_edges[0]].vertices[0],
                                      edges[active_edges[0]].vertices[1],
                                      edges[active_edges[0]].normals[0, :],
                                      edges[active_edges[0]].normals[1, :],
                                      y, 2)
        normalB = interpolate_vectors(edges[active_edges[1]].vertices[0],
                                      edges[active_edges[1]].vertices[1],
                                      edges[active_edges[1]].normals[0, :],
                                      edges[active_edges[1]].normals[1, :],
                                      y, 2)
        for x in range(int(min(x1, x2)), int(max(x1, x2)) + 1):
            interp_color = interpolate_vectors([x1, y], [x2, y], colorA, colorB, x, 1)
            interp_normal = interpolate_vectors([x1, y], [x2, y], normalA, normalB, x, 1)

            I = light(bcoords, interp_normal, interp_color, cam_pos, mat, lights, light_amb, lighting).T[0]

            updatedcanvas[y, int(round(x))] = np.clip(I, 0, 1)
        if y == y_max:
            break

        active_edges = update_active_edges(edges, active_edges, y)

    return updatedcanvas
