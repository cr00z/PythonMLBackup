import tkinter as tk
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Tuple
import utils


@dataclass
class Sphere:
    center: np.ndarray
    radius: int
    color: np.ndarray
    specular: int
    reflective: float


@dataclass
class Light:
    AMBIENT: ClassVar[int] = 0
    POINT: ClassVar[int] = 1
    DIRECTIONAL: ClassVar[int] = 2

    ltype: int
    intensity: float
    position: np.ndarray = 0


# constants
canvas_width = 200
canvas_height = 200
viewport_size = 1
projection_plane_z = 1
background_color = np.array((0, 0, 32))
recursion_depth = 3
small_value = 0.001

# frame1_1
# camera_position = np.array((0, 0, 0))
# camera_rotation = np.identity(3)

# frame1_2
camera_position = np.array([3, 0, 1])
camera_rotation = np.array([[0.7071, 0, -0.7071],
                            [0,      1,       0],
                            [0.7071, 0,  0.7071]])

spheres = [
    Sphere(np.array((0, -1, 3)), 1, np.array((255, 0, 0)), 500, 0.2),
    Sphere(np.array((2,  0, 4)), 1, np.array((0, 0, 255)), 500, 0.3),
    Sphere(np.array((-2, 0, 4)), 1, np.array((0, 255, 0)), 10,  0.4),
    # ground
    Sphere(np.array((0, -5001, 0)), 5000, np.array((255, 255, 0)), 1000, 0.5),
]

lights = [
    Light(Light.AMBIENT, 0.2),
    Light(Light.POINT, 0.6, np.array((2, 1, 0))),
    Light(Light.DIRECTIONAL, 0.2, np.array((1, 4, 4))),
]


def canvas_to_viewport(cx: int, cy: int) -> np.ndarray:
    """Convert canvas coordinates to scene (viewport) coordinates"""
    return np.array((
        cx * viewport_size / canvas_width,
        cy * viewport_size / canvas_height,
        projection_plane_z,
    ))


def intersect_ray_sphere(
        origin: np.ndarray, direction: np.ndarray, sphere: Sphere
) -> Tuple[float, float]:
    """Check intersect for direction ray and sphere"""
    oc = origin - sphere.center
    k1 = direction.dot(direction)
    k2 = 2 * oc.dot(direction)
    k3 = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = k2 * k2 - 4 * k1 * k3
    if discriminant < 0:
        return np.inf, np.inf
    t1 = (-k2 + np.sqrt(discriminant)) / (2 * k1)
    t2 = (-k2 - np.sqrt(discriminant)) / (2 * k1)
    return t1, t2


def vector_length(vector: np.ndarray) -> float:
    return np.sqrt(vector.dot(vector))


def closest_intersection(
        origin: np.ndarray, direction: np.ndarray, min_t: float, max_t: float
) -> Tuple[Sphere, int]:
    """Calculate the first intersection of the ray from a given point with the
    object """
    closest_t = np.inf
    closest_sphere = None
    for sphere in spheres:
        ts = intersect_ray_sphere(origin, direction, sphere)
        if (ts[0] < closest_t) and (min_t < ts[0] < max_t):
            closest_t = ts[0]
            closest_sphere = sphere
        if (ts[1] < closest_t) and (min_t < ts[1] < max_t):
            closest_t = ts[1]
            closest_sphere = sphere
    return closest_sphere, closest_t


def reflect_ray(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Returns a vector reflected relative to the normal"""
    return 2 * normal * normal.dot(vector) - vector


def compute_lighting(
        point: np.ndarray, normal: np.ndarray, view: np.ndarray, specular: int
) -> float:
    intensity = 0
    for light in lights:
        if light.ltype == Light.AMBIENT:
            intensity += light.intensity
        else:
            if light.ltype == Light.POINT:
                vec_l = light.position - point
                t_max = 1
            else:
                # Light.DIRECTIONAL
                vec_l = light.position
                t_max = np.inf

            # shadows
            shadow_sphere, shadow_t = closest_intersection(
                point, vec_l, small_value, t_max
            )
            if shadow_sphere is not None:
                continue

            # diffuse reflection
            n_dot_l = normal.dot(vec_l)
            if n_dot_l > 0:
                intensity += light.intensity * n_dot_l / \
                             (vector_length(normal) * vector_length(vec_l))
            # specular reflection
            if specular != -1:
                vec_r = reflect_ray(vec_l, normal)
                r_dot_v = vec_r.dot(view)
                if r_dot_v > 0:
                    intensity += light.intensity * pow(
                        r_dot_v / (vector_length(vec_r) * vector_length(view)),
                        specular
                    )
    return intensity


def trace_ray(
        origin: np.ndarray,
        direction: np.ndarray,
        min_t: int,
        max_t: int,
        recursion_depth: int
) -> np.ndarray:
    """Check intersect for all spheres"""
    closest_sphere, closest_t = closest_intersection(
        origin, direction, min_t, max_t
    )

    if closest_sphere is None:
        return background_color

    # compute lighting
    point = origin + closest_t * direction
    perpendicular = (point - closest_sphere.center)
    normal = perpendicular / vector_length(perpendicular)
    lighting = compute_lighting(
        point, normal, -direction, closest_sphere.specular
    )
    local_color = closest_sphere.color * lighting

    # if the recursion limit is reached or the object is not reflective,
    # then finished
    reflective = closest_sphere.reflective
    if (recursion_depth == 0) or (reflective <= 0):
        return local_color

    # compute reflection
    reflect_vector = reflect_ray(-direction, normal)
    reflected_color = trace_ray(
        point, reflect_vector, small_value, np.inf, recursion_depth - 1
    )

    return local_color * (1 - reflective) + reflected_color * reflective


if __name__ == '__main__':
    # init
    root = tk.Tk()
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()
    buffer = np.zeros((canvas_width, canvas_height, 3), dtype=np.uint8)

    # main loop
    for x in range(-canvas_width // 2, canvas_width // 2):
        for y in range(-canvas_height // 2, canvas_height // 2):
            direction = canvas_to_viewport(x, y)
            direction = camera_rotation.dot(direction)
            color = trace_ray(
                camera_position, direction, 1, np.inf, recursion_depth
            )
            buffer = utils.put_pixel(buffer, color.clip(0, 255), x, y)

    # draw buffer
    img = utils.photo_image(buffer)
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    root.mainloop()
