"""Medidas del contorno visible; no estima la parte oculta de una pieza.

La homografia convierte pixeles al plano de una referencia rectangular. Para
medidas en centimetros, la camara debe permanecer fija y la superficie medida
debe coincidir con ese plano. Cambios de perspectiva, resolucion, altura de las
piezas o distorsion de lente pueden introducir error. No es una calibracion 3D.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def make_calibration(points, width_cm: float, height_cm: float,
                     image_size: tuple[int, int], source: str | None = None) -> dict:
    """Crea pixels->cm con esquinas TL, TR, BR, BL y tamano (ancho, alto)."""
    width, height = map(int, image_size)
    if width < 2 or height < 2:
        raise ValueError("Resolucion de calibracion invalida")
    if not np.isfinite([width_cm, height_cm]).all() or min(width_cm, height_cm) <= 0:
        raise ValueError("Las dimensiones de referencia deben ser positivas")
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Se requieren cuatro esquinas finitas TL, TR, BR, BL")
    if ((pts[:, 0] < 0) | (pts[:, 0] >= width) |
            (pts[:, 1] < 0) | (pts[:, 1] >= height)).any():
        raise ValueError("Las esquinas deben estar dentro de la imagen")
    contour = pts.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 4:
        raise ValueError("Referencia degenerada o esquinas en orden incorrecto")
    destination = np.array([[0, 0], [width_cm, 0],
                            [width_cm, height_cm], [0, height_cm]], np.float32)
    matrix = cv2.getPerspectiveTransform(pts, destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("No se pudo obtener una homografia invertible")
    return {
        "schema_version": 1,
        "units": "cm",
        "image_size": [width, height],
        "reference_size_cm": [float(width_cm), float(height_cm)],
        "reference_points_px": pts.tolist(),
        "pixel_to_cm": matrix.tolist(),
        "source": source,
        "assumptions": "Camara fija, misma resolucion y mismo plano de referencia; contorno visible.",
    }


def validate_calibration(calibration: dict, image_size: tuple[int, int] | None = None) -> np.ndarray:
    if calibration.get("schema_version") != 1 or calibration.get("units") != "cm":
        raise ValueError("Formato de calibracion incompatible")
    size = calibration.get("image_size")
    if (not isinstance(size, list) or len(size) != 2 or
            any(not isinstance(v, int) or v < 2 for v in size)):
        raise ValueError("La calibracion no contiene una resolucion valida")
    if image_size is not None and tuple(size) != tuple(image_size):
        raise ValueError(f"Resolucion {tuple(image_size)} distinta de la calibracion {tuple(size)}; recalibre")
    matrix = np.asarray(calibration.get("pixel_to_cm"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Homografia invalida")
    return matrix


def load_calibration(path: str | Path, image_size: tuple[int, int] | None = None) -> dict:
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_calibration(calibration, image_size)
    return calibration


def measure_mask(mask: np.ndarray, calibration: dict | None = None) -> dict:
    """Rectangulo orientado minimo de todos los fragmentos visibles de una instancia."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("La mascara debe ser bidimensional")
    binary = (mask != 0).astype(np.uint8)
    height, width = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("La mascara esta vacia")
    points_px = np.concatenate(contours).astype(np.float32)
    if len(points_px) < 3 or cv2.contourArea(cv2.convexHull(points_px)) <= 0:
        raise ValueError("Mascara demasiado pequena o degenerada para medir")
    points = points_px
    matrix = None
    if calibration is not None:
        matrix = validate_calibration(calibration, (width, height))
        denominators = points_px[:, 0, :] @ matrix[2, :2] + matrix[2, 2]
        if np.any(np.abs(denominators) < 1e-8):
            raise ValueError("Contorno cerca del horizonte de la homografia; recalibre")
        points = cv2.perspectiveTransform(points_px, matrix)
        if not np.isfinite(points).all():
            raise ValueError("Transformacion de medidas no finita")
    rectangle = cv2.minAreaRect(points)
    sides = sorted(map(float, rectangle[1]), reverse=True)
    box = cv2.boxPoints(rectangle).reshape(-1, 1, 2)
    if matrix is not None:
        box = cv2.perspectiveTransform(box, np.linalg.inv(matrix))
    center = points_px[:, 0, :].mean(axis=0)
    truncated = bool(binary[0].any() or binary[-1].any() or
                     binary[:, 0].any() or binary[:, -1].any())
    return {
        "length": sides[0], "width": sides[1],
        "unit": "cm" if matrix is not None else "px",
        "box_px": box[:, 0, :].tolist(),
        "center_px": center.tolist(),
        "area_px": int(binary.sum()),
        "truncated": truncated, "visible_only": True,
    }
