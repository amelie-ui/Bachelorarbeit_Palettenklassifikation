import os
import time
import sys
import threading
import numpy as np
from pathlib import Path

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
MODEL_KEY       = 'baseline_fp32'
LED_ON_DURATION = 3.0
NUM_THREADS     = 2

# Auf True setzen für Livestream im Browser
# Lokal:      http://<jetson-ip>:5000
# SSH-Tunnel: ssh -L 5000:localhost:5000 amelie@192.168.178.150
#             → dann http://localhost:5000 im Browser
ENABLE_STREAM   = False
STREAM_PORT     = 5000

AVAILABLE_MODELS = {
    'baseline_fp32':     'baseline_fp32.tflite',
    'baseline_fp16':     'baseline_fp16.tflite',
    'baseline_int8':     'baseline_int8.tflite',
    'augmentation_fp32': 'augmentation_fp32.tflite',
    'augmentation_fp16': 'augmentation_fp16.tflite',
    'augmentation_int8': 'augmentation_int8.tflite',
}

# ── RGB-LED Pin-Belegung (BOARD-Schema) ───────────────────────────────────────
PIN_LED_R = 15
PIN_LED_G = 12
PIN_LED_B = 18

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
        print(f"  [SIM] RGB-LED leuchtet {color} für {LED_ON_DURATION}s.")
        time.sleep(LED_ON_DURATION)


# ── Livestream ────────────────────────────────────────────────────────────────

def start_stream_server():
    try:
        from flask import Flask, Response
    except ImportError:
        print("FEHLER: Flask nicht installiert. Bitte: pip install flask")
        return

    app = Flask(__name__)

    def generate_frames():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            import io
            buf = io.BytesIO()
            frame.save(buf, format='JPEG', quality=85)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.getvalue() +
                b'\r\n'
            )
            time.sleep(0.04)  # ~25 FPS

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
                img  {{ border: 2px solid #444; border-radius: 8px; max-width: 640px; }}
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
    print(f"SSH-Tunnel:    ssh -L {STREAM_PORT}:localhost:{STREAM_PORT} amelie@192.168.178.150")
    print(f"Dann im Browser: http://localhost:{STREAM_PORT}")


# ── Kamera ────────────────────────────────────────────────────────────────────

def capture_image(cap):
    """Liest 10 Frames und verwirft sie – nimmt dann den aktuellen Frame."""
    for _ in range(10):
        cap.read()  # echte read() statt grab() – leert den Buffer zuverlässig

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Kein Bild von der Kamera erhalten.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize(DATA['img_size'])
    pil_image.save('/home/amelie/DasWirdGut_IchKannDas/jetson/last_capture.jpg')
    return pil_image


# ── Modell & Inferenz ─────────────────────────────────────────────────────────

def load_interpreter():
    if MODEL_KEY not in AVAILABLE_MODELS:
        raise ValueError(f"Unbekanntes Modell: '{MODEL_KEY}'")

    model_path = PATHS['models'] / AVAILABLE_MODELS[MODEL_KEY]
    if not model_path.exists():
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {model_path}")

    interpreter = tflite.Interpreter(
        model_path=str(model_path),
        num_threads=NUM_THREADS
    )
    interpreter.allocate_tensors()
    print(f"Modell geladen: {AVAILABLE_MODELS[MODEL_KEY]}")
    return interpreter


def classify(interpreter, pil_image) -> tuple:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    img_input   = np.expand_dims(image_array, axis=0)

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

    class_index = int(np.argmax(probabilities))
    confidence  = float(np.max(probabilities))
    return class_index, confidence


# ── Hauptschleife ─────────────────────────────────────────────────────────────

def run_demo():
    global latest_frame

    interpreter = load_interpreter()

    if ON_JETSON:
        setup_gpio()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera konnte nicht geöffnet werden (/dev/video0).")
    print("Kamera bereit.")

    if ENABLE_STREAM:
        start_stream_server()

    print("\n" + "=" * 50)
    print("  Palettenklassifikation – Demo")
    print(f"  Modell:  {AVAILABLE_MODELS[MODEL_KEY]}")
    print(f"  Threads: {NUM_THREADS}")
    print(f"  Stream:  {'aktiv auf Port ' + str(STREAM_PORT) if ENABLE_STREAM else 'deaktiviert'}")
    print("  Enter drücken = neue Klassifikation starten.")
    print("  Beenden mit Strg+C")
    print("=" * 50 + "\n")

    try:
        while True:
            input("  Drücke Enter für nächste Klassifikation > ")
            print("── Klassifikation ──────────────────────────────")

            pil_image = capture_image(cap)

            # Frame in Stream-Buffer schreiben
            if ENABLE_STREAM:
                with frame_lock:
                    latest_frame = pil_image

            t0 = time.perf_counter()
            class_index, confidence = classify(interpreter, pil_image)
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