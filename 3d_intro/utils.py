import tkinter as tk
import numpy as np


def put_pixel(
        buffer: np.ndarray, color: np.ndarray, x: int, y: int
) -> np.ndarray:
    """Put pixel into buffer"""
    height, width = buffer.shape[:2]
    sx = width // 2 + x
    sy = height // 2 - y - 1
    if (sx >= 0) and (sx < width) and (sy >= 0) and (sy < height):
        buffer[sy, sx] = color
    return buffer


def photo_image(buffer: np.ndarray) -> tk.PhotoImage:
    """Convert into PPM format
    https://en.wikipedia.org/wiki/Netpbm#Description
    """
    height, width = buffer.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    data = ppm_header + buffer.tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')