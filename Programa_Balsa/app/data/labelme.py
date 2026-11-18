"""Conversión de LabelMe a instancias, sin confundir clase con ID de pieza.

Clases semánticas: 0=mesa (stuff), 1=balsa (thing).
Mapa de instancias: 0=mesa, 1..254=piezas, 255=sin anotación/ignorar.
Los polígonos de un mismo group_id forman una pieza. Un group_id nulo no
agrupa: cada polígono representa una pieza diferente.
"""
from __future__ import annotations

import hashlib
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABELS = {0: "mesa", 1: "balsa"}
IGNORE_INDEX = 255
EPS = 1e-8


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (abs(_cross(a, b, p)) <= EPS
            and min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS)


def _segments_intersect(a, b, c, d, *, proper=False):
    x, y, z, t = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    crossing = (((x > EPS and y < -EPS) or (x < -EPS and y > EPS))
                and ((z > EPS and t < -EPS) or (z < -EPS and t > EPS)))
    if crossing or proper:
        return crossing
    return any((_on_segment(a, b, c), _on_segment(a, b, d),
                _on_segment(c, d, a), _on_segment(c, d, b)))


def _strictly_inside(point, polygon):
    """Prueba par/impar: un punto de borde no pertenece al interior."""
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if _on_segment(a, b, point):
            return False
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < x:
                inside = not inside
    return inside


def _interior_edge_samples(polygon):
    """Muestras interiores próximas a bordes para detectar límites coincidentes."""
    signed_area = sum(a[0] * polygon[(i + 1) % len(polygon)][1]
                      - polygon[(i + 1) % len(polygon)][0] * a[1]
                      for i, a in enumerate(polygon))
    sign = 1 if signed_area > 0 else -1
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        delta = min(1e-5, length * 1e-3)
        p = ((a[0] + b[0]) / 2 - sign * dy / length * delta,
             (a[1] + b[1]) / 2 + sign * dx / length * delta)
        if _strictly_inside(p, polygon):
            yield p


def _positive_area_overlap(a, b):
    """No considera solapamiento el contacto por una arista o un vértice."""
    if (min(max(p[0] for p in a), max(p[0] for p in b))
            - max(min(p[0] for p in a), min(p[0] for p in b)) <= EPS
        or min(max(p[1] for p in a), max(p[1] for p in b))
            - max(min(p[1] for p in a), min(p[1] for p in b)) <= EPS):
        return False
    for i, p in enumerate(a):
        q = a[(i + 1) % len(a)]
        for j, r in enumerate(b):
            if _segments_intersect(p, q, r, b[(j + 1) % len(b)], proper=True):
                return True
    return (any(_strictly_inside(p, b) for p in a)
            or any(_strictly_inside(p, a) for p in b)
            or any(_strictly_inside(p, b) for p in _interior_edge_samples(a))
            or any(_strictly_inside(p, a) for p in _interior_edge_samples(b)))


def _validated_polygon(shape, width, height, index):
    if shape.get("shape_type", "polygon") != "polygon":
        raise ValueError(f"Forma {index}: solo se admiten polígonos, no {shape.get('shape_type')!r}.")
    raw = shape.get("points", [])
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"Forma {index}: un polígono requiere al menos tres puntos.")
    points = []
    for point in raw:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in point)):
            raise ValueError(f"Forma {index}: coordenadas inválidas.")
        x, y = map(float, point)
        if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
            raise ValueError(f"Forma {index}: coordenadas fuera de la imagen ({width}x{height}).")
        points.append((x, y))
    if points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError(f"Forma {index}: vértices repetidos o insuficientes.")
    area = abs(sum(a[0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * a[1]
                   for i, a in enumerate(points))) / 2
    if area <= EPS:
        raise ValueError(f"Forma {index}: área nula.")
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            if _segments_intersect(a, b, points[j], points[(j + 1) % len(points)]):
                raise ValueError(f"Forma {index}: polígono con autointersección.")
    return points


def _rasterize(points, size):
    if isinstance(points, np.ndarray):
        return points
    mask = Image.new("1", size)
    ImageDraw.Draw(mask).polygon(points, fill=1)
    return np.asarray(mask, dtype=bool)


def _validated_mask(shape, width, height, index):
    """Formato máscara nativo LabelMe: bbox inclusivo + PNG base64 recortado."""
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError(f"Forma {index}: una máscara necesita dos esquinas de bbox")
    coords = np.asarray(points, dtype=np.float64)
    if coords.shape != (2, 2) or not np.isfinite(coords).all() or not np.equal(coords, np.rint(coords)).all():
        raise ValueError(f"Forma {index}: bbox de máscara inválido")
    (x0, y0), (x1, y1) = coords.astype(int)
    if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
        raise ValueError(f"Forma {index}: bbox fuera de imagen")
    try:
        raw = base64.b64decode(shape.get("mask", ""), validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format != "PNG" or opened.size != (x1-x0+1, y1-y0+1):
                raise ValueError("Dimensiones o formato de máscara incompatible")
            patch = np.asarray(opened.convert("L"))
    except (ValueError, OSError, TypeError) as error:
        raise ValueError(f"Forma {index}: PNG de máscara inválido") from error
    if not np.isin(patch, [0, 1, 255]).all():
        raise ValueError(f"Forma {index}: la máscara debe ser binaria")
    result = np.zeros((height, width), dtype=bool)
    result[y0:y1+1, x0:x1+1] = patch > 0
    if not result.any():
        raise ValueError(f"Forma {index}: máscara vacía")
    return result


def load_labelme(image_path: str | Path, annotation_path: str | Path) -> dict:
    """Lee una imagen y sus máscaras sin alterar la imagen ni el JSON original."""
    image_path, annotation_path = Path(image_path).resolve(), Path(annotation_path).resolve()
    raw = annotation_path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
        raise ValueError(f"{annotation_path.name}: no es un JSON LabelMe con shapes.")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    if (document.get("imageWidth"), document.get("imageHeight")) != (width, height):
        raise ValueError(f"{annotation_path.name}: dimensiones del JSON distintas de la imagen.")
    declared = str(document.get("imagePath", "")).replace("\\", "/").rsplit("/", 1)[-1]
    if declared and declared.casefold() != image_path.name.casefold():
        raise ValueError(f"{annotation_path.name}: imagePath no corresponde a {image_path.name}.")
    mesa, pieces, piece_keys = [], [], {}
    automatic = []
    for index, shape in enumerate(document["shapes"]):
        if not isinstance(shape, dict) or shape.get("label") not in ("mesa", "balsa"):
            raise ValueError(f"{annotation_path.name}: etiqueta desconocida en forma {index}.")
        points = (_validated_mask(shape, width, height, index)
                  if shape.get("shape_type") == "mask"
                  else _validated_polygon(shape, width, height, index))
        if shape["label"] == "mesa":
            mesa.append(points)
            continue
        group_id = shape.get("group_id")
        if group_id is not None and (isinstance(group_id, (dict, list, bool))
                                      or not isinstance(group_id, (str, int, float))
                                      or isinstance(group_id, float) and not math.isfinite(group_id)):
            raise ValueError(f"Forma {index}: group_id debe ser escalar o null.")
        key = ("individual", index) if group_id is None else ("group", json.dumps(group_id, ensure_ascii=False))
        if key not in piece_keys:
            piece_keys[key] = len(pieces)
            pieces.append([])
        pieces[piece_keys[key]].append(points)
        description = str(shape.get("description") or "")
        if "automátic" in description.casefold() or "automatic" in description.casefold():
            automatic.append(index)
    if not mesa:
        raise ValueError(f"{annotation_path.name}: falta la anotación stuff 'mesa'.")
    if len(pieces) > 254:
        raise ValueError("Se admiten como máximo 254 instancias por imagen; 255 se reserva para ignore.")
    for i, piece in enumerate(pieces):
        for other in pieces[i + 1:]:
            if any((bool((_rasterize(a, image.size) & _rasterize(b, image.size)).any())
                    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
                    else _positive_area_overlap(a, b)) for a in piece for b in other):
                raise ValueError(f"{annotation_path.name}: solapamiento entre instancias balsa distintas.")
    instance_map = np.full((height, width), IGNORE_INDEX, dtype=np.int64)
    for polygon in mesa:
        instance_map[_rasterize(polygon, image.size)] = 0
    id2semantic = {0: 0}
    boundary_conflicts = 0
    for instance_id, piece in enumerate(pieces, 1):
        mask = np.zeros((height, width), dtype=bool)
        for polygon in piece:
            mask |= _rasterize(polygon, image.size)
        if not mask.any():
            raise ValueError(f"{annotation_path.name}: una pieza no cubre ningún píxel.")
        shared_boundary = mask & (instance_map > 0) & (instance_map < IGNORE_INDEX)
        boundary_conflicts += int(shared_boundary.sum())
        # El primer ID conserva los píxeles exactos de borde compartidos. Las
        # regiones interiores nunca se solapan: se comprobó geométricamente.
        mask &= ~shared_boundary
        if not mask.any():
            raise ValueError(f"{annotation_path.name}: pieza sin píxeles propios tras resolver bordes.")
        instance_map[mask] = instance_id
        id2semantic[instance_id] = 1
    return {
        "image": image,
        "instance_map": instance_map,
        "id2semantic": id2semantic,
        "id": image_path.stem,
        "metadata": {
            "image_path": str(image_path), "annotation_path": str(annotation_path),
            "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "original_hw": [height, width], "num_instances": len(pieces),
            "automatic_shape_indices": automatic,
            "unannotated_pixels": int((instance_map == IGNORE_INDEX).sum()),
            "shared_boundary_pixels": boundary_conflicts,
        },
    }
