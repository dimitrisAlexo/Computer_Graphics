import numpy as np


def rotmat(theta, u):
    # Calculates the rotation matrix R that corresponds to a clockwise rotation of an angle theta (in rads) around an
    # axis with direction that is given by the unit vector u.
    # - theta: the angle of rotation
    # - u: the unit vector indicating the direction of the axis of rotation
    # - R: the Rodrigues rotation matrix

    # Skew-symmetric matrix associated with the rotation axis
    K = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])

    # Rodrigues rotation matrix formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return R


def rotate_translate(cp, theta, u, A, t):
    # Transforms a point cp in R^3 (non-homogeneous) by rotating it at an angle of theta around an axis that passes
    # through A in R^3 and is parallel to the unit vector u. Then translates the point at a displacement vector t. All
    # coordinates refer to the same coordinate system (WCS). The function also works for cp and cp being 3*N arrays of
    # coordinates of points.
    # - cp: the point (or array of points) we want to rotate/translate
    # - theta: the angle of rotation
    # - u: the unit vector indicating the direction of the axis of rotation
    # - A: the point from which the axis of rotation passes through
    # - t: the displacement vector for translation of point cp
    # - cq: the rotated/translated point (or array of points)

    # Ensure that cp is a numpy array
    cp = np.array(cp)

    # Calculate the rotation matrix using rotmat function
    R = rotmat(theta, u)

    if cp.ndim == 1:
        # Single point case: cp is a 3x1 array
        v = cp - A
        v_rotated = np.dot(R, v)
        cq = v_rotated + A + t
    else:
        # Multiple points case: cp is a 3xN array
        v = cp - np.tile(A[:, np.newaxis], (1, cp.shape[1]))
        v_rotated = np.dot(R, v)
        cq = v_rotated + np.tile(A[:, np.newaxis], (1, cp.shape[1])) + np.tile(t[:, np.newaxis], (1, cp.shape[1]))

    return cq


def change_coordinate_system(cp, R, c0):
    # Changes the coordinate system for a point cp (3x1 array) or a 3*N array of coordinates of points.
    # - cp: 3*1 (or 3*N) array with the 3D coordinates in the old coordinate system
    # - dp: the coordinates of the point (or points) cp as regard to the new coordinate system with origin o+v0 (o is
    # the old origin), which is the outcome of a rotation R
    # - R: a rotation matrix
    # - c0: 3*1 vector with the coordinates of the v0 vector as regard to the old coordinate system

    # Ensure that cp is a numpy array
    cp = np.array(cp)

    if cp.ndim == 1:
        # Single point case: cp is a 3x1 array
        v = cp - c0
        dp = np.dot(R.T, v)
    else:
        # Multiple points case: cp is a 3xN array
        c0_reshaped = np.tile(c0[:, np.newaxis], (1, cp.shape[1]))
        v = cp - c0_reshaped
        dp = np.dot(R.T, v)

    return dp
