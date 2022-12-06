import tkinter as tk
import numpy as np
import utils
from dataclasses import dataclass
from typing import List


# point
@dataclass
class Pt:
    x: float
    y: float
    h: float = 1.0


@dataclass
class Vertex:
    x: float
    y: float
    z: float


@dataclass
class Triangle:
    v0_idx: int
    v1_idx: int
    v2_idx: int
    color: np.ndarray


@dataclass
class Model:
    vertexes: List[Vertex]
    triangles: List[Triangle]


# constants
canvas_width = 600
canvas_height = 600
viewport_size = 1
projection_plane_z = 1
# colors
red = np.array([255, 0, 0])
green = np.array([0, 255, 0])
blue = np.array([0, 0, 255])
yellow = np.array([255, 255, 0])
purple = np.array([255, 0, 255])
cyan = np.array([0, 255, 255])


def interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return np.array([d0])
    coef_a = (d1 - d0) / (i1 - i0)
    items = np.full((i1 - i0 + 1), coef_a)
    items[0] = d0
    values = np.cumsum(items)
    return values


def draw_line(buffer, p0, p1, color):
    if abs(p1.x - p0.x) > abs(p1.y - p0.y):
        if p0.x > p1.x:
            p0, p1 = p1, p0
        ys = interpolate(p0.x, p0.y, p1.x, p1.y)
        for x in range(p0.x, p1.x + 1):
            utils.put_pixel(buffer, color, x, round(ys[x - p0.x]))
    else:
        if p0.y > p1.y:
            p0, p1 = p1, p0
        xs = interpolate(p0.y, p0.x, p1.y, p1.x)
        for y in range(p0.y, p1.y + 1):
            utils.put_pixel(buffer, color, round(xs[y - p0.y]), y)


def draw_wireframe_triangle(buffer, p0, p1, p2, color):
    draw_line(buffer, p0, p1, color)
    draw_line(buffer, p1, p2, color)
    draw_line(buffer, p2, p0, color)


def draw_filled_triangle(buffer, p0, p1, p2, color):
    if p1.y < p0.y:
        p1, p0 = p0, p1
    if p2.y < p0.y:
        p2, p0 = p0, p2
    if p2.y < p1.y:
        p2, p1 = p1, p2

    x_left = interpolate(p0.y, p0.x, p2.y, p2.x)
    x_right = np.concatenate([
        interpolate(p0.y, p0.x, p1.y, p1.x)[:-1],
        interpolate(p1.y, p1.x, p2.y, p2.x)
    ])
    m = x_right.shape[0] // 2
    if x_left[m] > x_right[m]:
        x_left, x_right = x_right, x_left

    for y in range(p0.y, p2.y + 1):
        for x in range(round(x_left[y - p0.y]), round(x_right[y - p0.y]) + 1):
            utils.put_pixel(buffer, color, x, y)


def draw_shaded_triangle(buffer, p0, p1, p2, color):
    if p1.y < p0.y:
        p1, p0 = p0, p1
    if p2.y < p0.y:
        p2, p0 = p0, p2
    if p2.y < p1.y:
        p2, p1 = p1, p2

    x_left = interpolate(p0.y, p0.x, p2.y, p2.x)
    x_right = np.concatenate([
        interpolate(p0.y, p0.x, p1.y, p1.x)[:-1],
        interpolate(p1.y, p1.x, p2.y, p2.x)
    ])
    h_left = interpolate(p0.y, p0.h, p2.y, p2.h)
    h_right = np.concatenate([
        interpolate(p0.y, p0.h, p1.y, p1.h)[:-1],
        interpolate(p1.y, p1.h, p2.y, p2.h)
    ])
    m = x_right.shape[0] // 2
    if x_left[m] > x_right[m]:
        x_left, x_right = x_right, x_left
        h_left, h_right = h_right, h_left

    for y in range(p0.y, p2.y + 1):
        x_l = round(x_left[y - p0.y])
        x_r = round(x_right[y - p0.y])
        h_segment = interpolate(x_l, h_left[y - p0.y], x_r, h_right[y - p0.y])
        for x in range(x_l, x_r + 1):
            utils.put_pixel(buffer, color * h_segment[x - x_l], x, y)


def viewport_to_canvas(p: Pt) -> Pt:
    """Convert canvas coordinates to scene (viewport) coordinates"""
    return Pt(
        round(p.x * canvas_width / viewport_size),
        round(p.y * canvas_height / viewport_size)
    )


def project_vertex(v: Vertex) -> Pt:
    return viewport_to_canvas(Pt(
        v.x * projection_plane_z / v.z,
        v.y * projection_plane_z / v.z
    ))


def move_object(model, move):
    for v in model.vertexes:
        v.x += move[0]
        v.y += move[1]
        v.z += move[2]


def render_object(model):
    projected = []
    for v in model.vertexes:
        projected.append(project_vertex(v))
    for t in model.triangles:
        draw_wireframe_triangle(
            buffer,
            projected[t.v0_idx],
            projected[t.v1_idx],
            projected[t.v2_idx],
            t.color
        )


# scenes
def scene1():
    p0, p1, p2 = Pt(-200, -250, 0.3), Pt(200, 50, 0.1), Pt(20, 250, 1.0),
    draw_filled_triangle(buffer, p0, p1, p2, np.array([0, 255, 0]))
    p0.y += 100
    p1.y += 100
    p2.y += 100
    draw_shaded_triangle(buffer, p0, p1, p2, np.array([0, 255, 0]))
    p0.y += 100
    p1.y += 100
    p2.y += 100
    draw_wireframe_triangle(buffer, p0, p1, p2, np.array([255, 0, 0]))


def scene2():
    # cube
    vaf = Vertex(-2, -0.5, 5)
    vbf = Vertex(-2, 0.5, 5)
    vcf = Vertex(-1, 0.5, 5)
    vdf = Vertex(-1, -0.5, 5)

    vab = Vertex(-2, -0.5, 6)
    vbb = Vertex(-2, 0.5, 6)
    vcb = Vertex(-1, 0.5, 6)
    vdb = Vertex(-1, -0.5, 6)

    draw_line(buffer, project_vertex(vaf), project_vertex(vbf), blue)
    draw_line(buffer, project_vertex(vbf), project_vertex(vcf), blue)
    draw_line(buffer, project_vertex(vcf), project_vertex(vdf), blue)
    draw_line(buffer, project_vertex(vdf), project_vertex(vaf), blue)

    draw_line(buffer, project_vertex(vab), project_vertex(vbb), red)
    draw_line(buffer, project_vertex(vbb), project_vertex(vcb), red)
    draw_line(buffer, project_vertex(vcb), project_vertex(vdb), red)
    draw_line(buffer, project_vertex(vdb), project_vertex(vab), red)

    draw_line(buffer, project_vertex(vaf), project_vertex(vab), green)
    draw_line(buffer, project_vertex(vbf), project_vertex(vbb), green)
    draw_line(buffer, project_vertex(vcf), project_vertex(vcb), green)
    draw_line(buffer, project_vertex(vdf), project_vertex(vdb), green)


def scene3():
    cube = Model(
        # vertexes
        [
            Vertex(1, 1, 1),
            Vertex(-1, 1, 1),
            Vertex(-1, -1, 1),
            Vertex(1, -1, 1),
            Vertex(1, 1, -1),
            Vertex(-1, 1, -1),
            Vertex(-1, -1, -1),
            Vertex(1, -1, -1),
        ],
        # triangles
        [
            Triangle(0, 1, 2, red),
            Triangle(0, 2, 3, red),
            Triangle(4, 0, 3, green),
            Triangle(4, 3, 7, green),
            Triangle(5, 4, 7, blue),
            Triangle(5, 7, 6, blue),
            Triangle(1, 5, 6, yellow),
            Triangle(1, 6, 2, yellow),
            Triangle(4, 5, 1, purple),
            Triangle(4, 1, 0, purple),
            Triangle(2, 6, 7, cyan),
            Triangle(2, 7, 3, cyan),
        ]
    )
    move_object(cube, [-1.5, 0, 7])
    render_object(cube)


if __name__ == '__main__':
    # init
    root = tk.Tk()
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()
    buffer = np.zeros((canvas_width, canvas_height, 3), dtype=np.uint8)

    scene3()

    # draw buffer
    img = utils.photo_image(buffer)
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    root.mainloop()
