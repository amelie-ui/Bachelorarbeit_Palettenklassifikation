import os
import time
import sys
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
# Modell hier anpassen – einfach den Schlüssel ändern:
# 'baseline_fp32', 'baseline_fp16', 'baseline_int8',
# 'augmentation_fp32', 'augmentation_fp16', 'augmentation_int8'
MODEL_KEY = 'baseline_int8'

# Wie viele Sekunden die LED nach der Klassifikation leuchten soll
LED_ON_DURATION = 3.0

# Anzahl CPU-Threads für TFLite-Inferenz (1, 2 oder 4)
NUM_THREADS = 2

# Stream deaktiviert – auf True setzen um Flask-Livestream zu aktivieren
ENABLE_STREAM = False

# Verfügbare Modelle
AVAILABLE_MODELS = {
    'baseline_fp32':     'baseline_fp32.tflite',
    'baseline_fp16':     'baseline_fp16.tflite',
    'baseline_int8':     'baseline_int8.tflite',
    'augmentation_fp32': 'augmentation_fp32.tflite',
    'augmentation_fp16': 'augmentation_fp16.tflite',
    'augmentation_int8': 'augmentation_int8.tflite',
}

# ── RGB-LED Pin-Belegung (BOARD-Schema = physische Pinnummern) ────────────────
PIN_LED_R = 15   # physischer Pin 15 → Rot   (Klasse C)
PIN_LED_G = 12   # physischer Pin 12 → Grün  (Klasse A)
PIN_LED_B = 18   # physischer Pin 18 → Blau  (Klasse B)

# Mapping: Klassenindex → GPIO-Pin
# Index 0 = Klasse A (grün), 1 = Klasse B (blau), 2 = Klasse C (rot)
LED_PINS    = {0: PIN_LED_G, 1: PIN_LED_B, 2: PIN_LED_R}
CLASS_NAMES = {0: 'A', 1: 'B', 2: 'C'}


# ── GPIO ──────────────────────────────────────────────────────────────────────

def setup_gpio():
    """Initialisiert die drei LED-Pins im BOARD-Schema."""
    GPIO.setmode(GPIO.BOARD)
    for pin in LED_PINS.values():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


def all_leds_off():
    """Schaltet alle drei Farbkanäle aus."""
    if ON_JETSON:
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)


def signal_result(class_index: int):
    """
    Leuchtet die zur Klasse gehörige Farbe für LED_ON_DURATION Sekunden.
    Schaltet vorher alle anderen Farben aus.
    """
    all_leds_off()
    if ON_JETSON:
        GPIO.output(LED_PINS[class_index], GPIO.HIGH)
        time.sleep(LED_ON_DURATION)
        GPIO.output(LED_PINS[class_index], GPIO.LOW)
    else:
        color = {0: 'GRÜN', 1: 'BLAU', 2: 'ROT'}[class_index]
        print(f"  [SIM] RGB-LED leuchtet {color} für {LED_ON_DURATION}s.")
        time.sleep(LED_ON_DURATION)


# ── Kamera ────────────────────────────────────────────────────────────────────

def capture_image():
    """
    Nimmt ein einzelnes Bild von der Kamera auf (/dev/video0).
    Gibt ein PIL-Image zurück, das direkt für die Inferenz verwendet wird.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Kamera konnte nicht geöffnet werden (/dev/video0).")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Kein Bild von der Kamera erhalten.")

    # OpenCV liefert BGR → PIL erwartet RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Auf Modell-Eingabegröße skalieren (224×224)
    pil_image = pil_image.resize(DATA['img_size'])
    return pil_image


# ── Modell & Inferenz ─────────────────────────────────────────────────────────

def load_interpreter():
    """Lädt den TFLite-Interpreter anhand von MODEL_KEY."""
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
    """Einzelne TFLite-Inferenz – gibt Klassenindex und Konfidenz zurück."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Normalisierung auf [0, 1] und Batch-Dimension hinzufügen
    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    img_input   = np.expand_dims(image_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
    class_index   = int(np.argmax(probabilities))
    confidence    = float(np.max(probabilities))
    return class_index, confidence


# ── Hauptschleife ─────────────────────────────────────────────────────────────

def run_demo():
    interpreter = load_interpreter()

    if ON_JETSON:
        setup_gpio()

    print("\n" + "=" * 50)
    print("  Palettenklassifikation – Demo")
    print(f"  Modell:  {AVAILABLE_MODELS[MODEL_KEY]}")
    print(f"  Threads: {NUM_THREADS}")
    print("  Enter drücken = neue Klassifikation starten.")
    print("  Beenden mit Strg+C")
    print("=" * 50 + "\n")

    try:
        while True:
            input("  Drücke Enter für nächste Klassifikation > ")
            print("── Klassifikation ──────────────────────────────")

            # Bild von Kamera aufnehmen
            pil_image = capture_image()

            # Klassifikation
            t0 = time.perf_counter()
            class_index, confidence = classify(interpreter, pil_image)
            duration_ms = (time.perf_counter() - t0) * 1000

            predicted = CLASS_NAMES[class_index]

            print(f"  Ergebnis:  Klasse {predicted}  ({confidence*100:.1f}% Konfidenz)")
            print(f"  Inferenz:  {duration_ms:.1f} ms\n")

            # LED-Feedback
            signal_result(class_index)

    except KeyboardInterrupt:
        print("\nDemo beendet.")
    finally:
        all_leds_off()
        if ON_JETSON:
            GPIO.cleanup()


if __name__ == '__main__':
    run_demo()