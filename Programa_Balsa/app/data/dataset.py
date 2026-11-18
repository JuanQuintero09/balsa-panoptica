"""Manifest portable, revisión trazable y Dataset sin dependencia de torch."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

from data.labelme import load_labelme, sha256_file

SPLITS = ("train", "val", "test")


def read_manifest(manifest_path: str | Path):
    path = Path(manifest_path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1 or not isinstance(document.get("records"), list):
        raise ValueError("Manifest incompatible: se requieren schema_version=1 y records.")
    # La ruta relativa permite mover toda la carpeta del proyecto sin editar el
    # manifest. Por defecto, data/manifest.json se refiere a su carpeta superior.
    root = (path.parent / document.get("project_root", "..")).resolve()
    return path, root, document


def _resolved_record(record, root):
    output = dict(record)
    for field in ("image", "annotation"):
        raw = record.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"Registro {record.get('id')}: falta {field}.")
        resolved = (root / raw).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Registro {record.get('id')}: no existe {resolved}.")
        output[field] = str(resolved)
    return output


def validate_manifest(document, root, *, check_image_hashes=True):
    """Comprueba todos los splits juntos para descubrir duplicados y fugas."""
    ids, images, annotations, image_hashes, groups = set(), set(), set(), {}, {}
    resolved = []
    for record in document["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("Cada registro necesita un id de texto no vacío.")
        item = _resolved_record(record, root)
        sample_id = item["id"]
        if sample_id in ids:
            raise ValueError(f"ID duplicado: {sample_id}.")
        ids.add(sample_id)
        image_key, annotation_key = (os.path.normcase(item[k]) for k in ("image", "annotation"))
        if image_key in images or annotation_key in annotations:
            raise ValueError(f"Ruta de imagen/anotación duplicada: {sample_id}.")
        images.add(image_key)
        annotations.add(annotation_key)
        if check_image_hashes:
            digest = sha256_file(item["image"])
            if digest in image_hashes:
                raise ValueError(f"Imágenes duplicadas por contenido: {image_hashes[digest]} y {sample_id}.")
            image_hashes[digest] = sample_id
        split, group = item.get("split"), item.get("group")
        if split is not None and split not in SPLITS:
            raise ValueError(f"{sample_id}: split debe ser train, val, test o null.")
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise ValueError(f"{sample_id}: group debe ser texto no vacío o null.")
        if split is not None and group is None:
            raise ValueError(f"{sample_id}: asigna group/toma antes de asignar split.")
        if group is not None and split is not None:
            if group in groups and groups[group] != split:
                raise ValueError(f"Grupo {group!r} compartido entre splits {groups[group]} y {split}.")
            groups[group] = split
        if not isinstance(item.get("reviewed", False), bool):
            raise ValueError(f"{sample_id}: reviewed debe ser booleano.")
        expected = item.get("annotation_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{sample_id}: falta annotation_sha256 válido.")
        current = sha256_file(item["annotation"])
        item["annotation_changed"] = current != expected
        item["reviewed"] = bool(item.get("reviewed")) and not item["annotation_changed"]
        item["current_annotation_sha256"] = current
        resolved.append(item)
    return resolved


def load_records(manifest_path: str | Path, split: str, require_reviewed: bool = True) -> list[dict]:
    """Devuelve rutas absolutas y rechaza revisiones obsoletas por cambios JSON."""
    if split not in SPLITS:
        raise ValueError(f"Split inválido: {split!r}.")
    _, root, document = read_manifest(manifest_path)
    records = validate_manifest(document, root)
    selected = [r for r in records if r.get("split") == split]
    if not selected:
        raise ValueError(f"No hay registros asignados a {split}; asigna tomas con review_data.py.")
    if require_reviewed:
        pending = [r["id"] for r in selected if not r["reviewed"]]
        if pending:
            raise ValueError(f"{split}: faltan revisión manual vigente en {len(pending)} imágenes: "
                             + ", ".join(pending[:8]) + ". Usa review_data.py después de revisarlas.")
    return selected


class BalsaDataset:
    def __init__(self, records: list[dict], max_size: int = 320, augment: bool = False, cache: bool = False):
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size debe ser un entero positivo.")
        self.records, self.max_size, self.augment = records, max_size, augment
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.cache is not None and index in self.cache:
            sample = dict(self.cache[index])
            sample["image"] = sample["image"].copy()
            sample["instance_map"] = sample["instance_map"].copy()
            return self._augment_sample(sample, record)
        sample = load_labelme(record["image"], record["annotation"])
        if record.get("reviewed") and sample["metadata"]["annotation_sha256"] != record.get("annotation_sha256"):
            raise ValueError(f"{record['id']}: la anotación cambió después de la revisión.")
        image, mask = sample["image"], sample["instance_map"]
        scale = min(1.0, self.max_size / max(image.size))
        target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        if target != image.size:
            image = image.resize(target, Image.Resampling.BILINEAR)
            mask = np.asarray(Image.fromarray(mask.astype(np.int32)).resize(target, Image.Resampling.NEAREST), dtype=np.int64)
        sample["image"], sample["instance_map"] = image, mask
        if self.cache is not None:
            cached = dict(sample)
            cached["instance_map"] = mask.copy()
            self.cache[index] = cached
        return self._augment_sample(sample, record)

    def _augment_sample(self, sample, record):
        image, mask = sample["image"], sample["instance_map"]
        if self.augment:
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), np.fliplr(mask)
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), np.flipud(mask)
        sample["image"] = image
        sample["instance_map"] = np.ascontiguousarray(mask, dtype=np.int64)
        present = set(np.unique(mask).tolist())
        sample["id2semantic"] = {key: value for key, value in sample["id2semantic"].items() if key in present}
        # Incluye siempre la definición stuff, incluso si una imagen queda
        # completamente cubierta por piezas tras recortar/redimensionar.
        sample["id2semantic"][0] = 0
        sample["id"] = record.get("id", sample["id"])
        sample["metadata"]["processed_hw"] = [image.height, image.width]
        return sample
