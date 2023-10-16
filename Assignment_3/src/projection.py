from transform import *


def pin_hole(f, cv, cx, cy, cz, p3d):
    # Calculates the projection of a 3D image onto the 2D plane.
    # - p3d: 3*N numpy array with the 3D coordinates of points in the WCS
    # - cv: 3*1 numpy array with the 3D coordinates of the pinhole camera's center with respect to the WCS' s origin
    # - cx, cy, cz: 3*1 numpy arrays with the 3D coordinates of the unit vectors x, y, z respectively of the camera
    # - f: focal length of the camera
    # - p2d: 2*N numpy array with the 2D coordinates after projection from p3d
    # - depth: 1*N numpy array with the depth of each point in p3d

    # Ensure that p3d is a numpy array
    p3d = np.array(p3d)

    # Transform p3d to camera coordinate system
    R = np.column_stack((cx, cy, cz))
    c0 = np.dot(R, cv)
    p3d_cam = change_coordinate_system(p3d, R, c0)

    # Calculate projection onto the image plane
    p2d = (f * p3d_cam[:2]) / p3d_cam[2]
    depth = -p3d_cam[2]

    return p2d, depth


def camera_looking_at(f, cv, ck, cup, p3d):

    # Produces the p2d projection of p3d points using the pin_hole() function, but takes into account the direction the
    # camera is looking at.
    # - f: focal length of the camera
    # - cv: 3*1 numpy array with the 3D coordinates of the pinhole camera's center with respect to the WCS' s origin
    # - ck: 3*1 numpy array with the 3D coordinates of the target point K of the camera
    # - cup: the unit up-vector
    # - p3d: 3*N numpy array with the 3D coordinates of points in the WCS
    # - p2d: 2*N numpy array with the 2D coordinates after projection from p3d
    # - depth: 1*N numpy array with the depth of each point in p3d

    # Compute the camera coordinate system
    cz = (cv - ck) / np.linalg.norm(cv - ck)
    cx = np.cross(cup, cz)
    cy = np.cross(cz, cx)

    # Normalize cx, cy, and cz
    cx /= np.linalg.norm(cx)
    cy /= np.linalg.norm(cy)
    cz /= np.linalg.norm(cz)

    # Call pin_hole function with the computed camera coordinate system
    p2d, depth = pin_hole(f, cv, cx, cy, cz, p3d)

    return p2d, depth
