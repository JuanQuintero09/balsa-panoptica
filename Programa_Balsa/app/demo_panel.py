"""Panel local de demostración. La cámara se abre solo por una acción explícita.

No requiere un framework web. Sirve únicamente en 127.0.0.1 y conserva una sola
imagen procesada; el entrenamiento y las métricas se ejecutan por separado.
"""
from __future__ import annotations

import argparse
import copy
import json
import select
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
STATIC = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}


class DemoState:
    """Un solo trabajador y un único fotograma compartido, sin colas de vídeo."""

    def __init__(self, checkpoint: Path, manifest: Path, device: str, max_size: int,
                 threshold: float, photo_interval: float = 2.0):
        self.checkpoint, self.manifest = checkpoint, manifest
        self.device_name, self.max_size, self.threshold = device, max_size, threshold
        self.photo_interval = photo_interval
        self.condition = threading.Condition()
        self.closed = False
        self.generation = 0
        self.requested = None
        self.jpeg = None
        self.sequence = 0
        self.model = self.processor = self.device = None
        self.records = None
        self.last_viewed = time.monotonic()
        self.info = {
            "phase": "idle", "mode": None, "source_label": "Sin fuente activa",
            "message": "Elige una fuente para comenzar.", "frame_id": None,
            "frames": 0, "measurements": [], "resolution": None,
            "inference_fps": None, "processing_fps": None,
            "inference_ms": None, "processing_ms": None, "device": None,
            "error": None, "ids_are_tracks": False,
        }
        self.worker = threading.Thread(target=self._work, name="balsa-inference", daemon=True)
        self.worker.start()

    def status(self):
        with self.condition:
            self.last_viewed = time.monotonic()
            result = copy.deepcopy(self.info)
            result["source_run"] = self.generation
            result["frame_sequence"] = self.sequence if self.jpeg is not None else None
            return result

    def start(self, mode, camera_index=0):
        if mode not in ("photos", "camera"):
            raise ValueError("La fuente debe ser fotografías o cámara.")
        if isinstance(camera_index, bool) or not isinstance(camera_index, int) or not 0 <= camera_index <= 5:
            raise ValueError("El índice de cámara debe estar entre 0 y 5.")
        with self.condition:
            self.generation += 1
            self.requested = {"mode": mode, "camera_index": camera_index}
            self.jpeg = None
            self.last_viewed = time.monotonic()
            label = "Secuencia de fotografías de prueba" if mode == "photos" else f"Cámara {camera_index}"
            self.info.update(phase="starting", mode=mode, source_label=label,
                             message="Preparando la demostración…", error=None, measurements=[],
                             frames=0, frame_id=None, inference_fps=None, processing_fps=None,
                             inference_ms=None, processing_ms=None, resolution=None)
            self.condition.notify_all()

    def stop(self, message="Demostración detenida."):
        with self.condition:
            self.generation += 1
            self.requested = None
            self.info.update(phase="idle", mode=None, source_label="Demostración detenida",
                             message=message, measurements=[], inference_fps=None,
                             processing_fps=None, inference_ms=None, processing_ms=None,
                             error=None, frames=0, resolution=None)
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.generation += 1
            self.requested = None
            self.condition.notify_all()
        self.worker.join(timeout=10)

    def _load(self):
        if self.model is not None:
            return
        import torch
        from models.predict import load_model
        torch.set_num_threads(4)
        model, processor, device = load_model(self.checkpoint, self.device_name)
        if getattr(model.config, "balsa_task", None) != "panoptic":
            raise ValueError("La demostración requiere el modelo panóptico de balsa.")
        if getattr(model.config, "balsa_training_status", None) != "trained":
            raise ValueError("La demostración requiere pesos entrenados; no se acepta un smoke_test.")
        self.model, self.processor, self.device = model, processor, device

    def _test_records(self):
        if self.records is None:
            from data.dataset import load_records
            records = load_records(self.manifest, "test", require_reviewed=True)
            # Comenzar por escenas variadas sin incorporar fotos ajenas al test.
            preferred = ["balsa_0072", "balsa_0109", "balsa_0095", "balsa_0185", "balsa_0228"]
            rank = {sid: i for i, sid in enumerate(preferred)}
            self.records = sorted(records, key=lambda r: (rank.get(r["id"], len(rank)), r["id"]))
        return self.records

    def _work(self):
        import cv2
        import numpy as np
        from PIL import Image
        from main import caption, render_prediction
        from models.predict import predict_image

        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.requested is not None)
                if self.closed:
                    return
                request, generation = dict(self.requested), self.generation
            capture = None
            try:
                self._load()
                with self.condition:
                    if generation != self.generation or self.closed:
                        continue
                mode = request["mode"]
                records = self._test_records() if mode == "photos" else None
                if mode == "camera":
                    # Único punto que abre una cámara: solo se alcanza tras POST explícito.
                    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                    capture = cv2.VideoCapture(request["camera_index"], backend)
                    if not capture.isOpened():
                        raise RuntimeError("No se pudo abrir la cámara seleccionada. Revisa su conexión y que otra aplicación no la esté usando.")
                position = frames = 0
                while True:
                    with self.condition:
                        if self.closed or generation != self.generation:
                            break
                        if mode == "camera" and time.monotonic() - self.last_viewed > 15:
                            self.stop("La cámara se detuvo al cerrar o desconectar el panel.")
                            break
                    started = time.perf_counter()
                    if mode == "photos":
                        record = records[position % len(records)]
                        with Image.open(record["image"]) as opened:
                            rgb = opened.convert("RGB")
                        frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                        frame_id = record["id"]
                    else:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError("La cámara dejó de entregar imágenes. Comprueba su conexión.")
                        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_id = f"Fotograma {frames + 1}"
                    prediction = predict_image(self.model, self.processor, rgb, self.device,
                                               max_size=self.max_size, threshold=self.threshold)
                    rendered, measurements = render_prediction(frame, prediction)
                    if mode == "photos":
                        caption(rendered, "Secuencia de fotografias de prueba - no captura de camara",
                                (10, 49), scale=.53)
                    ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not ok:
                        raise RuntimeError("No se pudo preparar la imagen para el panel.")
                    duration = time.perf_counter() - started
                    inference = float(prediction["seconds"])
                    colors = ["#ff5032", "#1eaaff", "#78dc46", "#d246e6", "#f0dc46", "#3c8cd2"]
                    for i, measurement in enumerate(measurements):
                        measurement["color"] = colors[i % len(colors)]
                    frames += 1
                    with self.condition:
                        if generation != self.generation or self.closed:
                            break
                        self.jpeg = encoded.tobytes()
                        self.sequence += 1
                        self.info.update(phase="running", frames=frames, frame_id=frame_id,
                                         message="Procesando fotografías de prueba." if mode == "photos" else "Cámara en funcionamiento.",
                                         measurements=measurements, resolution=[rgb.width, rgb.height],
                                         inference_fps=1/max(inference, 1e-9), processing_fps=1/max(duration, 1e-9),
                                         inference_ms=inference*1000, processing_ms=duration*1000,
                                         device=str(self.device), error=None)
                        self.condition.notify_all()
                    if mode == "photos":
                        position += 1
                        with self.condition:
                            self.condition.wait_for(lambda: self.closed or generation != self.generation,
                                                    timeout=self.photo_interval)
            except Exception as error:
                traceback.print_exc()
                with self.condition:
                    if generation == self.generation:
                        self.requested = None
                        self.info.update(phase="error", mode=None, error=str(error),
                                         source_label="Fuente detenida por un error",
                                         message=str(error), measurements=[], frames=0,
                                         frame_id=None, resolution=None, inference_fps=None,
                                         processing_fps=None, inference_ms=None, processing_ms=None)
                        self.condition.notify_all()
            finally:
                if capture is not None:
                    capture.release()


def metric_rows():
    rows = []
    for filename, name, instance_capable in (
        ("semantic_extended_test.json", "Mask2Former semántico V1", False),
        ("v1_extended_test.json", "Mask2Former panóptico V1", True),
        ("robust_extended_test.json", "Mask2Former robusto V2", True),
    ):
        path = ROOT / "results" / filename
        row = {"model": name, "available": False, "instance_capable": instance_capable}
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("definitive") and report.get("evaluation_split", "test") == "test":
                semantic = report["semantic"]
                negatives = [r for r in report["per_image"] if r["ground_truth_instances"] == 0]
                false_positive_images = sum(r["semantic"]["per_class"]["balsa"]["fp"] > 0 for r in negatives)
                row.update(available=True, image_count=semantic["image_count"],
                           positive_images=semantic["image_count"] - len(negatives),
                           negative_images=len(negatives), negative_images_with_fp=false_positive_images,
                           balsa_iou=semantic["balsa_iou"], balsa_f1=semantic["balsa_f1"],
                           balsa_precision=semantic["balsa_precision"],
                           map=report["instances"].get("map") if instance_capable else None)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        rows.append(row)
    return {"rows": rows}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 8

    def __init__(self, address, state):
        super().__init__(address, Handler)
        self.state = state
        self.stream_slots = threading.BoundedSemaphore(4)
        self.request_slots = threading.BoundedSemaphore(16)

    def process_request(self, request, client_address):
        if not self.request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "BalsaDemo/1.0"
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def _local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in hosts:
            self._json(403, {"error": "El panel admite únicamente conexiones locales."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in hosts}:
            self._json(403, {"error": "La acción debe iniciarse desde este panel."})
            return False
        return True

    def _headers(self, status, kind, length=None):
        if status >= 400:
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()

    def _json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(payload))
        self.wfile.write(payload)

    def do_GET(self):
        if not self._local_request():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/status":
            self._json(200, self.server.state.status())
        elif path == "/api/metrics":
            self._json(200, metric_rows())
        elif path == "/api/frame.jpg":
            requested = parse_qs(parsed.query, keep_blank_values=True).get("sequence")
            try:
                sequence = int(requested[0]) if requested is not None and len(requested) == 1 else None
                if requested is not None and (sequence is None or sequence < 0):
                    raise ValueError
            except (ValueError, TypeError):
                self._json(400, {"error": "Secuencia de imagen inválida."})
                return
            with self.server.state.condition:
                jpeg = self.server.state.jpeg
                current_sequence = self.server.state.sequence
            if jpeg is None:
                self._json(404, {"error": "Todavía no hay una imagen procesada."})
            elif sequence is not None and sequence != current_sequence:
                self._json(409, {"error": "La imagen cambió. Consulta su estado actualizado."})
            else:
                self._headers(200, "image/jpeg", len(jpeg))
                self.wfile.write(jpeg)
        elif path == "/stream.mjpg":
            self._stream()
        elif path in STATIC:
            file = WEB / STATIC[path]
            try:
                payload = file.read_bytes()
            except OSError:
                self._json(404, {"error": "No se encontró el recurso del panel."})
                return
            kind = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}.get(file.suffix, "application/octet-stream")
            if kind.startswith(("text/", "application/javascript")):
                kind += "; charset=utf-8"
            self._headers(200, kind, len(payload))
            self.wfile.write(payload)
        else:
            self._json(404, {"error": "Ruta no disponible."})

    def do_POST(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path not in ("/api/start", "/api/stop"):
            self._json(404, {"error": "Acción no disponible."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 1024:
                raise ValueError("Solicitud demasiado grande.")
            if length and self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise ValueError("La solicitud debe usar JSON.")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Solicitud inválida.")
            if path == "/api/start":
                self.server.state.start(body.get("mode"), body.get("camera_index", 0))
            else:
                self.server.state.stop()
            self._json(200, self.server.state.status())
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})

    def _stream(self):
        if not self.server.stream_slots.acquire(blocking=False):
            self._json(429, {"error": "Ya hay suficientes vistas abiertas. Cierra una pestaña."})
            return
        self.close_connection = True
        try:
            self._headers(200, "multipart/x-mixed-replace; boundary=frame")
            last = -1
            state = self.server.state
            while not state.closed:
                with state.condition:
                    state.condition.wait_for(lambda: state.closed or (state.jpeg is not None and state.sequence != last), timeout=2)
                    if state.closed:
                        break
                    if state.jpeg is None or state.sequence == last:
                        # Descubrir desconexiones aun sin frames nuevos evita
                        # acumular streams inactivos al cerrar pestañas.
                        if select.select([self.connection], [], [], 0)[0]:
                            if not self.connection.recv(1, socket.MSG_PEEK):
                                break
                        continue
                    jpeg, last = state.jpeg, state.sequence
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            self.server.stream_slots.release()

    def log_message(self, format, *args):
        if self.path.startswith("/api/status"):
            return
        super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default="checkpoints/balsa_robusto_final")
    parser.add_argument("--manifest", default="data/manifest_robusto.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--photo-interval", type=float, default=2.0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.max_size < 64 or not 0 <= args.threshold <= 1 or args.photo_interval < .2:
        parser.error("Puerto 1024..65535, max-size>=64, threshold0..1 y photo-interval>=0.2.")
    state = DemoState((ROOT / args.checkpoint).resolve(), (ROOT / args.manifest).resolve(),
                      args.device, args.max_size, args.threshold, args.photo_interval)
    try:
        server = LocalServer((args.host, args.port), state)
    except OSError as error:
        state.close()
        raise SystemExit(f"No se pudo abrir el panel local: {error}. Prueba otro --port.") from error
    url = f"http://127.0.0.1:{args.port}"
    print(f"Panel de balsa: {url}\nLa cámara está apagada. Pulsa Ctrl+C en esta consola para cerrar.", flush=True)
    if not args.no_browser:
        threading.Timer(.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print("Cerrando la demostración…", flush=True)
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
