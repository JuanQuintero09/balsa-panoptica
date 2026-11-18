"""Transferencia desde el Mask2Former cartográfico aportado por el estudiante."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

LABELS = {0: "mesa", 1: "balsa"}
IGNORE = 255


def make_processor(checkpoint):
    from transformers import Mask2FormerImageProcessor
    processor = Mask2FormerImageProcessor.from_pretrained(
        str(checkpoint), local_files_only=True,
    )
    processor.do_resize = False  # Se conserva el aspecto y se añade padding explícito.
    processor.do_reduce_labels = False  # Mesa=0 es una clase válida.
    processor.ignore_index = IGNORE
    processor.num_labels = 2
    return processor


def make_model(checkpoint, task="panoptic", freeze_backbone=True, train_num_points=None):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    checkpoint = Path(checkpoint)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"No se encontró {checkpoint / 'model.safetensors'}")
    config = Mask2FormerConfig.from_pretrained(str(checkpoint), local_files_only=True)
    # Se reemplaza SOLO la cabeza de categorías: 5+no-object -> 2+no-object.
    config.num_labels = len(LABELS)
    config.id2label = LABELS.copy()
    config.label2id = {name: i for i, name in LABELS.items()}
    config.balsa_task = task
    config.balsa_thing_ids = [1]
    config.balsa_stuff_ids = [0]
    if train_num_points is not None:
        if train_num_points < 256:
            raise ValueError("train_num_points debe ser >=256")
        config.train_num_points = int(train_num_points)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(checkpoint), config=config, local_files_only=True,
        use_safetensors=True, ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for parameter in model.model.pixel_level_module.encoder.parameters():
            parameter.requires_grad_(False)
    return model, make_processor(checkpoint)


def pad_sample(image, instance_map=None, divisor=32):
    """Añade padding al borde derecho/inferior, sin deformar los objetos."""
    width, height = image.size
    padded_w = (width + divisor - 1) // divisor * divisor
    padded_h = (height + divisor - 1) // divisor * divisor
    result = Image.new("RGB", (padded_w, padded_h), (124, 116, 104))
    result.paste(image, (0, 0))
    if instance_map is None:
        return result, None
    if instance_map.shape != (height, width):
        raise ValueError("La imagen y el mapa de instancias tienen tamaños distintos")
    target = np.full((padded_h, padded_w), IGNORE, dtype=np.int64)
    target[:height, :width] = instance_map
    return result, target


def encode_batch(processor, batch, device, task="panoptic"):
    """Conserva una máscara por pieza, aun cuando comparten clase o se tocan."""
    import torch
    images, maps, mappings = [], [], []
    for sample in batch:
        target = sample["instance_map"].copy()
        mapping = dict(sample["id2semantic"])
        if task == "semantic":
            # Baseline adaptado: todas las piezas comparten UNA máscara de clase.
            target = np.where(target == IGNORE, IGNORE, (target > 0).astype(np.int64))
            mapping = {0: 0, 1: 1}
        elif task != "panoptic":
            raise ValueError(f"Tarea desconocida: {task}")
        image, target = pad_sample(sample["image"], target)
        images.append(image)
        maps.append(target)
        mappings.append(mapping)
    # HF 4.57.6 rellena con 255 las máscaras YA binarias al formar un batch.
    # Igualar tamaños antes del processor evita objetivos BCE/Dice con valor 255.
    common_w = max(image.width for image in images)
    common_h = max(image.height for image in images)
    for i, image in enumerate(images):
        canvas = Image.new("RGB", (common_w, common_h), (124, 116, 104))
        canvas.paste(image, (0, 0))
        target = np.full((common_h, common_w), IGNORE, dtype=np.int64)
        target[:image.height, :image.width] = maps[i]
        images[i], maps[i] = canvas, target
    encoded = processor(
        images=images, segmentation_maps=maps,
        instance_id_to_semantic_id=mappings, return_tensors="pt",
    )
    inputs = {"pixel_values": encoded["pixel_values"].to(device),
              "mask_labels": [m.to(device) for m in encoded["mask_labels"]],
              "class_labels": [c.to(device) for c in encoded["class_labels"]]}
    if "pixel_mask" in encoded:
        pixel_mask = encoded["pixel_mask"].clone()
        for i, sample in enumerate(batch):
            w, h = sample["image"].size
            pixel_mask[i, h:, :] = 0
            pixel_mask[i, :, w:] = 0
        inputs["pixel_mask"] = pixel_mask.to(device)
    if any(not torch.isfinite(x).all() for x in inputs["mask_labels"]):
        raise ValueError("Máscaras no finitas")
    if any(((x < 0) | (x > 1)).any() for x in inputs["mask_labels"]):
        raise ValueError("Las máscaras binarias contienen valores fuera de [0,1]")
    return inputs
