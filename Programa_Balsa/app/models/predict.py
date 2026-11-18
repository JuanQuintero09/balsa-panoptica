"""Inferencia con instancias, stuff fusionado y retorno a resolución original."""
from __future__ import annotations
from pathlib import Path
import json
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image

from .mask2former import LABELS, make_processor, pad_sample


def get_device(device="auto"):
    import torch
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA, pero PyTorch no detecta una GPU compatible")
    return result


def load_model(checkpoint, device="auto"):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    checkpoint = Path(checkpoint)
    run_path = checkpoint / "training_run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("run_state") in {"running", "cooldown", "failed"}:
            raise RuntimeError("El checkpoint aún no está listo para la demostración. Espera a que termine el entrenamiento y su evaluación.")
    config = Mask2FormerConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if config.num_labels != 2 or config.id2label != LABELS:
        raise ValueError(
            "Este checkpoint no tiene las etiquetas mesa=0 y balsa=1. "
            "Los pesos cartográficos originales requieren ajuste fino con train.py."
        )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=True,
    )
    device = get_device(device)
    model.to(device).eval()
    return model, make_processor(checkpoint), device


def predict_image(model, processor, image, device, max_size=320, threshold=0.5):
    import torch
    if max_size < 64 or not 0 <= threshold <= 1:
        raise ValueError("max_size debe ser >=64 y threshold estar en [0,1]")
    device = get_device(device)
    image = image.convert("RGB")
    original_size = image.size
    scale = min(1.0, max_size / max(image.size))
    small_size = tuple(max(1, round(v * scale)) for v in image.size)
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    padded, _ = pad_sample(small)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    encoded = processor(images=[padded], return_tensors="pt")
    pixel_mask = encoded.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask[:, small.height:, :] = 0
        pixel_mask[:, :, small.width:] = 0
    inputs = {k: v.to(device) for k, v in encoded.items() if k in ("pixel_values", "pixel_mask")}
    with torch.inference_mode():
        outputs = model(**inputs)
    task = getattr(model.config, "balsa_task", "panoptic")
    if task == "semantic":
        raw = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(padded.height, padded.width)],
        )[0].cpu().numpy().astype(np.int32)
        raw = raw[:small.height, :small.width]
        panoptic = np.asarray(Image.fromarray(raw).resize(original_size, Image.Resampling.NEAREST)).copy()
        semantic = panoptic.astype(np.uint8)
        segments = [{"id": i, "label_id": i, "score": 1.0}
                    for i in (0, 1) if np.any(panoptic == i)]
    else:
        # Evita una colisión de IDs en compute_segments de Transformers 4.57.6:
        # reutilizar un ID stuff antes de una cosa puede reiniciar su contador.
        # Las cosas van primero y las regiones fusionables al final.
        labels = outputs.class_queries_logits[0].argmax(-1)
        order = torch.cat([(labels == 1).nonzero().flatten(),
                           (labels == 0).nonzero().flatten(),
                           ((labels != 0) & (labels != 1)).nonzero().flatten()])
        ordered = SimpleNamespace(
            class_queries_logits=outputs.class_queries_logits[:, order],
            masks_queries_logits=outputs.masks_queries_logits[:, order],
        )
        result = processor.post_process_panoptic_segmentation(
            ordered, threshold=threshold, mask_threshold=0.5,
            overlap_mask_area_threshold=0.8, label_ids_to_fuse={0},
            target_sizes=[(padded.height, padded.width)],
        )[0]
        raw = result["segmentation"].cpu().numpy().astype(np.int32)
        raw = raw[:small.height, :small.width]
        raw = np.asarray(Image.fromarray(raw).resize(original_size, Image.Resampling.NEAREST))
        # HF puede devolver ceros sin segmento: no convertirlos en mesa inventada.
        panoptic = np.full(raw.shape, -1, dtype=np.int32)
        semantic = np.full(raw.shape, 255, dtype=np.uint8)
        segments = []
        seen = set()
        for segment in result["segments_info"]:
            sid, label = int(segment["id"]), int(segment["label_id"])
            if sid in seen:
                continue  # Las queries de mesa fusionadas comparten descriptor.
            present = raw == sid
            if not present.any():
                continue
            panoptic[present] = sid
            semantic[present] = label
            segments.append({"id": sid, "label_id": label, "score": float(segment["score"])})
            seen.add(sid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"panoptic": panoptic, "semantic": semantic, "segments": segments,
            "seconds": time.perf_counter() - start, "task": task,
            "instance_capable": task == "panoptic"}
