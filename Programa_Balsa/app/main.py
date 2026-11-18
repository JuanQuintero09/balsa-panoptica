"""Camara/video/imagen: segmentacion panoptica y medidas del contorno visible.

Los IDs se asignan por fotograma, sin seguimiento temporal. Los centimetros
requieren calibracion del mismo plano y la misma camara fija. Una pieza ocluida
o cortada por el borde solo permite medir su parte visible, no su tamano completo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.measure import load_calibration, measure_mask, validate_calibration


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [(50, 80, 255), (255, 170, 30), (70, 220, 120),
          (230, 70, 210), (70, 220, 240), (210, 140, 60)]


def caption(image: np.ndarray, text: str, position=(10, 25), scale=.55, color=(255, 255, 255)):
    """Texto con fondo para mantener contraste sobre la mesa y la madera."""
    h, w = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(int(position[0]), max(0, w - tw - 8)))
    y = max(th + 3, min(int(position[1]), h - baseline - 3))
    cv2.rectangle(image, (x, y - th - 3), (min(w - 1, x + tw + 7), y + baseline + 3), (20, 24, 30), -1)
    cv2.putText(image, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_prediction(frame: np.ndarray, prediction: dict, calibration: dict | None = None,
                      status_banner: str | None = None):
    """Dibuja a resolucion original y devuelve medidas auditables como JSON."""
    h, w = frame.shape[:2]
    panoptic = np.asarray(prediction["panoptic"])
    semantic = np.asarray(prediction["semantic"])
    if panoptic.shape != (h, w) or semantic.shape != (h, w):
        raise ValueError("La prediccion debe estar en la resolucion original antes de medir")
    if calibration is not None:
        validate_calibration(calibration, (w, h))
    result = frame.copy()
    mesa = semantic == 0
    result[mesa] = (.88 * result[mesa] + .12 * np.array([210, 100, 25])).astype(np.uint8)
    unknown = semantic == 255
    result[unknown] = (.7 * result[unknown] + .3 * np.array([160, 30, 160])).astype(np.uint8)
    measurements = []
    for index, segment in enumerate(s for s in prediction["segments"] if int(s["label_id"]) == 1):
        segment_id = int(segment["id"])
        mask = panoptic == segment_id
        if not mask.any():
            continue
        color = COLORS[index % len(COLORS)]
        result[mask] = (.70 * result[mask] + .30 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        row = {"segment_id": segment_id, "score": float(segment.get("score", 0.0))}
        try:
            measure = measure_mask(mask, calibration)
            row.update(measure)
            box = np.rint(measure["box_px"]).astype(np.int32)
            cv2.polylines(result, [box], True, color, 2, cv2.LINE_AA)
            text = f'#{segment_id} {measure["length"]:.1f} x {measure["width"]:.1f} {measure["unit"]}'
            if measure["truncated"]:
                text += " [borde]"
            center = measure["center_px"]
            caption(result, text, (int(center[0]) - 70, int(center[1])), scale=.5, color=color)
        except ValueError as exc:
            row["measurement_error"] = str(exc)
        measurements.append(row)
    caption(result, f'Mesa | Balsa: {len(measurements)} | IDs por fotograma', (10, 24), scale=.55)
    if status_banner:
        caption(result, status_banner, (10, 74), scale=.55, color=(0, 200, 255))
    caption(result, "Medidas del contorno visible" + (" | cm: plano calibrado" if calibration else " | px"),
            (10, h - 10), scale=.50)
    return result, measurements


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentacion panoptica de balsa con camara y medidas")
    parser.add_argument("--checkpoint", default="checkpoints/balsa_robusto_final")
    parser.add_argument("--source", default="0", help="Indice de camara, ruta de video o imagen")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-size", type=int, default=320, help="Borde largo de inferencia; medidas a resolucion original")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calibration", help="JSON generado por calibrate.py para esta camara fija y plano")
    parser.add_argument("--save", help="Ruta de imagen o video de salida; genera tambien un JSON/JSONL de medidas")
    parser.add_argument("--headless", action="store_true", help="Procesar sin abrir una ventana")
    parser.add_argument("--max-frames", type=int, default=0, help="0=todo el video; obligatorio >0 en camara headless")
    args = parser.parse_args()
    if args.max_size < 64 or not 0 <= args.threshold <= 1 or args.max_frames < 0 or args.threads < 1:
        parser.error("max-size >=64, threshold entre 0 y 1 y max-frames >=0")
    if args.headless and args.source.isdigit() and args.max_frames == 0:
        parser.error("Indique --max-frames >0 para una camara sin ventana")
    return args


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    # La carga diferida permite consultar --help sin cargar pesos ni acceder a la camara.
    from models.predict import load_model, predict_image
    model, processor, device = load_model(args.checkpoint, device=args.device)
    if getattr(model.config, "balsa_task", None) != "panoptic":
        raise ValueError("La demo necesita un checkpoint balsa_task=panoptic entrenado por instancias")
    training_status = getattr(model.config, "balsa_training_status", None)
    status_banner = None
    if training_status in {"smoke_test", "draft"}:
        status_banner = f'BORRADOR: {training_status} - no es un modelo final'
        print(status_banner)
    calibration = load_calibration(args.calibration) if args.calibration else None
    if calibration and args.source.isdigit() and str(calibration.get("source", "")).startswith("camera:"):
        if calibration["source"] != f"camera:{int(args.source)}":
            raise ValueError("La calibracion se obtuvo con otro indice de camara; recalibre esta fuente")
    save = Path(args.save) if args.save else None
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    if source_path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(source_path) as pil:
            rgb = pil.convert("RGB")
        frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        started = time.perf_counter()
        prediction = predict_image(model, processor, rgb, device, max_size=args.max_size, threshold=args.threshold)
        rendered, measurements = render_prediction(frame, prediction, calibration, status_banner)
        elapsed = time.perf_counter() - started
        inference = float(prediction["seconds"])
        report = {"source": str(source_path), "device": str(device), "training_status": training_status,
                  "resolution": [rgb.width, rgb.height], "inference_seconds": inference,
                  "processing_seconds": elapsed, "measurements": measurements,
                  "note": "Una imagen; no representa FPS sostenidos de una camara."}
        if save:
            if save.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("Una entrada de imagen requiere una salida de imagen en --save")
            if not cv2.imwrite(str(save), rendered):
                raise RuntimeError(f"No se pudo guardar {save}")
            save.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not args.headless:
            cv2.imshow("Balsa panoptica - tecla para cerrar", rendered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"No se pudo abrir la fuente de video: {args.source}")
    writer = None
    log_file = None
    n = 0
    inference_total = processing_total = 0.0
    previous_fps = None
    try:
        fps_source = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps_source) or fps_source <= 0:
            fps_source = 20.0
        if save:
            if save.suffix.lower() not in {".mp4", ".avi"}:
                raise ValueError("Una fuente de video requiere --save con extension .mp4 o .avi")
            log_file = save.with_suffix(".jsonl").open("w", encoding="utf-8")
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if calibration is not None:
                validate_calibration(calibration, (w, h))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction = predict_image(model, processor, image, device,
                                       max_size=args.max_size, threshold=args.threshold)
            rendered, measurements = render_prediction(frame, prediction, calibration, status_banner)
            inference_seconds = float(prediction["seconds"])
            inference_fps = 1.0 / max(inference_seconds, 1e-9)
            text = f'Inferencia: {inference_fps:.1f} FPS'
            if previous_fps is not None:
                text += f' | Ciclo previo: {previous_fps:.1f} FPS'
            caption(rendered, text, (10, 48), scale=.5)
            if writer is None and save:
                codec = "MJPG" if save.suffix.lower() == ".avi" else "mp4v"
                writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*codec), fps_source, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"No se pudo crear el video de salida: {save}")
            if writer is not None:
                writer.write(rendered)
            key = -1
            if not args.headless:
                cv2.imshow("Balsa panoptica - Q/ESC para salir", rendered)
                key = cv2.waitKey(1) & 0xff
            elapsed = time.perf_counter() - started
            previous_fps = 1.0 / max(elapsed, 1e-9)
            n += 1
            inference_total += inference_seconds
            processing_total += elapsed
            if log_file:
                log_file.write(json.dumps({"frame": n, "resolution": [w, h],
                    "inference_seconds": inference_seconds, "processing_seconds": elapsed,
                    "measurements": measurements}, ensure_ascii=False) + "\n")
            if key in {ord("q"), 27} or (args.max_frames and n >= args.max_frames):
                break
        if n == 0:
            raise RuntimeError("La fuente se abrio pero no entrego fotogramas")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if log_file:
            log_file.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({"frames": n, "device": str(device), "training_status": training_status,
                      "mean_inference_fps": n / max(inference_total, 1e-9),
                      "mean_processing_fps": n / max(processing_total, 1e-9),
                      "includes_first_frame_warmup": True,
                      "ids_are_temporal_tracks": False}, indent=2))


if __name__ == "__main__":
    main()
