import os
import time
import sys
import threading
import numpy as np
from pathlib import Path
import random
import io

os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA, PATHS

try:
    import Jetson.GPIO as GPIO
    ON_JETSON = True
except ImportError:
    print("WARNUNG: Jetson.GPIO nicht verfügbar – Simulationsmodus aktiv.")
    ON_JETSON = False

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

import cv2
from PIL import Image

# ── Konfiguration ─────────────────────────────────────────────────────────────
MODEL_KEY        = 'baseline_fp32'
LED_ON_DURATION  = 3.0
NUM_THREADS      = 2

# Stream: auf True setzen für Livestream im Browser
# Lokal:      http://192.168.178.150:5000
# SSH-Tunnel: ssh -L 5000:localhost:5000 amelie@192.168.178.150
ENABLE_STREAM    = True
STREAM_PORT      = 5000

# Bildquelle: True = Testbilder aus Datensatz, False = Kamera live
USE_TEST_IMAGES  = False

AVAILABLE_MODELS = {
    'baseline_fp32':     'baseline_fp32.tflite',
    'baseline_fp16':     'baseline_fp16.tflite',
    'baseline_int8':     'baseline_int8.tflite',
    'augmentation_fp32': 'augmentation_fp32.tflite',
    'augmentation_fp16': 'augmentation_fp16.tflite',
    'augmentation_int8': 'augmentation_int8.tflite',
}

# ── RGB-LED Pin-Belegung (BOARD-Schema) ───────────────────────────────────────
PIN_LED_R = 15   # Rot  → Klasse C
PIN_LED_G = 12   # Grün → Klasse A
PIN_LED_B = 18   # Blau → Klasse B

LED_PINS    = {0: PIN_LED_G, 1: PIN_LED_B, 2: PIN_LED_R}
CLASS_NAMES = {0: 'A', 1: 'B', 2: 'C'}

# ── Globaler Frame-Buffer für den Stream ──────────────────────────────────────
latest_frame = None
frame_lock   = threading.Lock()


# ── GPIO ──────────────────────────────────────────────────────────────────────

def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    for pin in LED_PINS.values():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


def all_leds_off():
    if ON_JETSON:
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)


def signal_result(class_index: int):
    all_leds_off()
    if ON_JETSON:
        GPIO.output(LED_PINS[class_index], GPIO.HIGH)
        time.sleep(LED_ON_DURATION)
        GPIO.output(LED_PINS[class_index], GPIO.LOW)
    else:
        color = {0: 'GRÜN', 1: 'BLAU', 2: 'ROT'}[class_index]
        print(f"  [SIM] LED leuchtet {color} für {LED_ON_DURATION}s.")
        time.sleep(LED_ON_DURATION)


# ── Livestream ────────────────────────────────────────────────────────────────

def start_stream_server():
    try:
        from flask import Flask, Response
    except ImportError:
        print("FEHLER: Flask nicht installiert – pip install flask")
        return

    app = Flask(__name__)

    def generate_frames():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG', quality=85)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.getvalue() +
                b'\r\n'
            )
            time.sleep(0.04)

    @app.route('/')
    def index():
        return f"""
        <html>
        <head>
            <title>Palettenklassifikation – Live</title>
            <style>
                body {{ background: #111; color: #eee; font-family: sans-serif;
                        display: flex; flex-direction: column; align-items: center;
                        padding: 2rem; }}
                img  {{ border: 2px solid #444; border-radius: 8px;
                        width: 640px; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Palette Classification – Live</h1>
            <img src="/stream" />
            <p>Modell: {AVAILABLE_MODELS[MODEL_KEY]} | Threads: {NUM_THREADS}</p>
        </body>
        </html>
        """

    @app.route('/stream')
    def stream():
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=STREAM_PORT, debug=False, use_reloader=False),
        daemon=True
    )
    thread.start()
    print(f"Stream aktiv → http://192.168.178.150:{STREAM_PORT}")
    print(f"SSH-Tunnel:   ssh -L {STREAM_PORT}:localhost:{STREAM_PORT} amelie@192.168.178.150")


def start_camera_stream_thread(cap):
    """
    Hintergrund-Thread: liest kontinuierlich Frames von der Kamera
    und schreibt sie in latest_frame für den Stream.
    Läuft unabhängig vom Enter-Trigger — der Browser zeigt immer
    das aktuelle Kamerabild, auch zwischen Klassifikationen.
    """
    global latest_frame

    def loop():
        while True:
            try:
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb).resize((640, 480))
                    with frame_lock:
                        latest_frame = pil_image
                time.sleep(0.04)  # ~25 FPS
            except Exception:
                break

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


# ── Kamera ────────────────────────────────────────────────────────────────────

def open_camera():
    """Öffnet GStreamer-Pipeline (Raspberry Pi Kamera), Fallback auf /dev/video0."""
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Kamera bereit – Modus: GStreamer")
        return cap, 'gstreamer'

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print("Kamera bereit – Modus: v4l2")
        return cap, 'v4l2'

    raise RuntimeError("Keine Kamera verfügbar.")


def capture_image(cap, mode: str):
    if USE_TEST_IMAGES:
        all_images = list(PATHS['dataset_test'].rglob('*.jpg'))
        if not all_images:
            raise FileNotFoundError("Keine Testbilder gefunden.")
        img_path   = random.choice(all_images)
        true_label = img_path.parent.name[0]
        print(f"  Testbild: {img_path.name}  (Klasse: {true_label})")
        pil_stream  = Image.open(img_path).resize((640, 480))
        pil_classify = pil_stream.resize(DATA['img_size'])
    else:
        if mode == 'v4l2':
            for _ in range(10):
                cap.read()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Kein Bild von der Kamera erhalten.")
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_stream   = Image.fromarray(frame_rgb).resize((640, 480))
        pil_classify = pil_stream.resize(DATA['img_size'])

    return pil_stream, pil_classify


# ── Modell & Inferenz ─────────────────────────────────────────────────────────

def load_interpreter():
    if MODEL_KEY not in AVAILABLE_MODELS:
        raise ValueError(f"Unbekanntes Modell: '{MODEL_KEY}'")

    model_path = PATHS['models'] / AVAILABLE_MODELS[MODEL_KEY]
    if not model_path.exists():
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {model_path}")

    interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    print(f"Modell geladen: {AVAILABLE_MODELS[MODEL_KEY]}")
    return interpreter


def classify(interpreter, pil_image) -> tuple:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Rohpixelwerte [0, 255] – Normalisierung übernimmt das Modell intern
    img_input = np.expand_dims(np.array(pil_image, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output_detail = output_details[0]
    raw_output    = interpreter.get_tensor(output_detail['index'])[0]

    # INT8-Dequantisierung falls nötig
    if output_detail['dtype'] == np.int8:
        scale, zero_point = output_detail['quantization']
        probabilities = (raw_output.astype(np.float32) - zero_point) * scale
    else:
        probabilities = raw_output

    return int(np.argmax(probabilities)), float(np.max(probabilities))


# ── Hauptschleife ─────────────────────────────────────────────────────────────

def run_demo():
    global latest_frame
    interpreter = load_interpreter()

    if ON_JETSON:
        setup_gpio()

    cap, camera_mode = open_camera()

    if ENABLE_STREAM:
        start_stream_server()

    print("\n" + "=" * 50)
    print("  Palettenklassifikation – Demo")
    print(f"  Modell:  {AVAILABLE_MODELS[MODEL_KEY]}")
    print(f"  Quelle:  {'Testbilder' if USE_TEST_IMAGES else 'Kamera (' + camera_mode + ')'}")
    print(f"  Stream:  {'aktiv → Port ' + str(STREAM_PORT) if ENABLE_STREAM else 'deaktiviert'}")
    print("  Enter drücken = Klassifikation starten | Strg+C = Beenden")
    print("=" * 50 + "\n")

    try:
        while True:
            input("  Drücke Enter > ")
            print("── Klassifikation ──────────────────────────────")

            pil_stream, pil_classify = capture_image(cap, camera_mode)

            # Stream mit großem Bild updaten (640×480)
            if ENABLE_STREAM:
                with frame_lock:
                    latest_frame = pil_stream

            t0 = time.perf_counter()
            class_index, confidence = classify(interpreter, pil_classify)
            duration_ms = (time.perf_counter() - t0) * 1000

            predicted = CLASS_NAMES[class_index]
            print(f"  Ergebnis:  Klasse {predicted}  ({confidence*100:.1f}% Konfidenz)")
            print(f"  Inferenz:  {duration_ms:.1f} ms\n")

            signal_result(class_index)

    except KeyboardInterrupt:
        print("\nDemo beendet.")
    finally:
        cap.release()
        all_leds_off()
        if ON_JETSON:
            GPIO.cleanup()


if __name__ == '__main__':
    run_demo()