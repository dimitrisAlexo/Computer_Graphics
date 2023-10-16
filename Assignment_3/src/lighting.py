import numpy as np


def ambient(mat, light_amb):

    I = np.zeros((3, 1))  # Initialize the intensity vector as zeros

    # Ambient light calculation
    I += mat.k_a * light_amb

    return I


def diffusion(point, normal, vcolor, mat, lights):

    I = np.zeros((3, 1))  # Initialize the intensity vector as zeros

    for l in lights:
        # Diffuse reflection calculation
        L_d = l.pos.T - point
        L_d /= np.linalg.norm(L_d)  # Normalize the light direction vector

        I_d = mat.k_d * max(np.dot(L_d.T[0], normal.T[0]), 0)
        I += I_d * l.intensity.T * vcolor

    return I


def specular(point, normal, vcolor, cam_pos, mat, lights):

    I = np.zeros((3, 1))  # Initialize the intensity vector as zeros

    for l in lights:
        L_d = l.pos.T - point
        L_d /= np.linalg.norm(L_d)  # Normalize the light direction vector

        # Specular reflection calculation
        V_s = cam_pos - point
        V_s /= np.linalg.norm(V_s)  # Normalize the view direction vector

        R = 2 * np.dot(L_d.T[0], normal.T[0]) * normal - L_d
        R /= np.linalg.norm(R)  # Normalize the reflection direction vector

        I_s = mat.k_s * np.dot(V_s.T[0], R.T[0]) ** mat.n_phong
        I += I_s * l.intensity.T * vcolor

    return I


def light(point, normal, vcolor, cam_pos, mat, lights, light_amb, lighting):
    # calculates the lighting of a single point of a PhongMaterial surface due to ambient light, diffuse reflection and
    # specular reflection
    # - point: 3*1 with the 3D coordinates of the point of the surface
    # - normal: 3*1 vector with the coordinates of the normal vector of the surface at the aforementioned point
    #  (perpendicular to the surface with outwards directions - direction to the observer)
    # - vcolor: 3*1 vector with each color (rgb) as a float in [0, 1]
    # - cam_pos: 3*1 vector with the 3D coordinates of the camera (observer)
    # - mat: object of type PhongMaterial
    # - lights: list of objects of type PointLight
    # - light_amb: 3 × 1 vector with the components of the ambient radiation intensity in the interval [0, 1]
    # - lighting: string {"ambient", "diffusion", "specular", "full"}
    # - I: return value; 3*1 vector of intensity for each color (rgb) that reflects from the aforementioned point
    # according to the Phong model

    point = np.array([point]).T
    normal = np.array([normal]).T
    vcolor = np.array([vcolor]).T
    cam_pos = np.array([cam_pos]).T

    I = np.zeros((3, 1))  # Initialize the intensity vector as zeros

    assert lighting in ["ambient", "diffusion", "specular", "full"]

    if lighting == "ambient":
        I = ambient(mat, light_amb)
    elif lighting == "diffusion":
        I = diffusion(point, normal, vcolor, mat, lights)
    elif lighting == "specular":
        I = specular(point, normal, vcolor, cam_pos, mat, lights)
    elif lighting == "full":
        I = ambient(mat, light_amb) + diffusion(point, normal, vcolor, mat, lights) + \
            specular(point, normal, vcolor, cam_pos, mat, lights)

    return I
