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
    print("[DEBUG] GPIO initialisiert – Pins:", list(LED_PINS.values()))


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
        print("FEHLER: Flask nicht installiert.")
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
    print(f"[DEBUG] Stream aktiv → http://192.168.178.150:{STREAM_PORT}")


# ── Kamera ────────────────────────────────────────────────────────────────────

def open_camera():
    """
    Versucht zuerst GStreamer (Raspberry Pi Kamera), dann Fallback auf /dev/video0.
    Gibt das cap-Objekt und den verwendeten Modus zurück.
    """
    gstreamer_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=224, height=224, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )

    print("[DEBUG] Versuche GStreamer-Pipeline (Raspberry Pi Kamera)...")
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        print("[DEBUG] GStreamer-Pipeline erfolgreich geöffnet.")
        return cap, 'gstreamer'

    print("[DEBUG] GStreamer fehlgeschlagen – Fallback auf /dev/video0 (USB-Kamera).")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        print("[DEBUG] /dev/video0 erfolgreich geöffnet.")
        return cap, 'v4l2'

    raise RuntimeError("Keine Kamera verfügbar – weder GStreamer noch /dev/video0.")


def capture_image(cap, mode: str):
    """
    Nimmt ein aktuelles Bild auf.
    Bei v4l2: Buffer leeren via mehrfaches read().
    Bei GStreamer: direkt lesen, Buffer wird intern verwaltet.
    """
    if mode == 'v4l2':
        # Buffer leeren: 10 Frames verwerfen
        for _ in range(10):
            cap.read()

    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Kein Bild von der Kamera erhalten.")

    # Debug: Rohbild-Statistiken ausgeben
    print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"[DEBUG] Pixelwerte – min: {frame.min()}, max: {frame.max()}, mean: {frame.mean():.1f}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize(DATA['img_size'])

    # Bild speichern für visuelle Kontrolle
    save_path = '/home/amelie/DasWirdGut_IchKannDas/jetson/last_capture.jpg'
    pil_image.save(save_path)
    print(f"[DEBUG] Bild gespeichert: {save_path}")

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

    # Debug: Input/Output-Details ausgeben
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[DEBUG] Modell geladen: {AVAILABLE_MODELS[MODEL_KEY]}")
    print(f"[DEBUG] Input  – shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    print(f"[DEBUG] Output – shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")

    return interpreter


def classify(interpreter, pil_image) -> tuple:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_array = np.array(pil_image, dtype=np.float32)
    img_input   = np.expand_dims(image_array, axis=0)

    print(f"[DEBUG] Input-Array – min: {img_input.min():.3f}, max: {img_input.max():.3f}, mean: {img_input.mean():.3f}")

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output_detail = output_details[0]
    raw_output    = interpreter.get_tensor(output_detail['index'])[0]

    print(f"[DEBUG] Rohes Modell-Output: {raw_output}")

    # INT8-Dequantisierung falls nötig
    if output_detail['dtype'] == np.int8:
        scale, zero_point = output_detail['quantization']
        probabilities = (raw_output.astype(np.float32) - zero_point) * scale
        print(f"[DEBUG] Dequantisiert (scale={scale}, zp={zero_point}): {probabilities}")
    else:
        probabilities = raw_output

    print(f"[DEBUG] Wahrscheinlichkeiten – A: {probabilities[0]:.3f}, B: {probabilities[1]:.3f}, C: {probabilities[2]:.3f}")

    class_index = int(np.argmax(probabilities))
    confidence  = float(np.max(probabilities))
    return class_index, confidence


# ── Hauptschleife ─────────────────────────────────────────────────────────────

def run_demo():
    global latest_frame

    interpreter = load_interpreter()

    if ON_JETSON:
        setup_gpio()

    cap, camera_mode = open_camera()
    print(f"[DEBUG] Kamera bereit – Modus: {camera_mode}")

    if ENABLE_STREAM:
        start_stream_server()

    print("\n" + "=" * 50)
    print("  Palettenklassifikation – Demo")
    print(f"  Modell:  {AVAILABLE_MODELS[MODEL_KEY]}")
    print(f"  Threads: {NUM_THREADS}")
    print(f"  Kamera:  {camera_mode}")
    print(f"  Stream:  {'aktiv auf Port ' + str(STREAM_PORT) if ENABLE_STREAM else 'deaktiviert'}")
    print("  Enter drücken = neue Klassifikation starten.")
    print("  Beenden mit Strg+C")
    print("=" * 50 + "\n")

    try:
        while True:
            input("  Drücke Enter für nächste Klassifikation > ")
            print("── Klassifikation ──────────────────────────────")

            pil_image = capture_image(cap, camera_mode)

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