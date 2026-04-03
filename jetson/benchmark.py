import os
import time
import json
import numpy as np
import sys
import psutil
from pathlib import Path

# Umgebungsvariable für Jetson Nano Stabilität
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# Pfad-Setup
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA, PATHS

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def get_process_ram_mb():
    """Prozessspezifischer RAM-Verbrauch via psutil (RSS)."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def load_test_images(n_images: int = 30):
    """
    Lädt Bilder gleichmäßig verteilt aus den Klassenordnern.
    Verhindert den '[:1]' Fehler, der nur 3 Bilder gesamt geladen hat.
    """
    from PIL import Image
    images = []
    test_path = PATHS['dataset_test']

    # Klassenordner identifizieren (A, B, C)
    classes = [d for d in sorted(test_path.iterdir()) if d.is_dir()]
    if not classes:
        print("FEHLER: Keine Klassenordner in dataset_test gefunden!")
        return []

    # Bilder pro Klasse berechnen für gleichmäßige Verteilung
    imgs_per_class = n_images // len(classes)

    for class_dir in classes:
        all_imgs = sorted(list(class_dir.glob('*.jpg')))
        # Nimm imgs_per_class Bilder aus diesem Ordner
        for img_path in all_imgs[:imgs_per_class]:
            img = Image.open(img_path).resize(DATA['img_size'])
            # Normalisierung auf [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)

    print(f'Erfolgreich geladen: {len(images)} Testbilder ({len(classes)} Klassen).')
    return images


def preprocess(image):
    """Erweitert Dimensionen für den TFLite-Input [1, 224, 224, 3]."""
    return np.expand_dims(image.astype(np.float32), axis=0)


def benchmark_model(tflite_path: str, images: list, n_warmup: int = 5):
    """Benchmarkt ein TFLite-Modell auf dem Jetson Nano."""

    # Interpreter Setup
    interpreter = tflite.Interpreter(
        model_path=str(tflite_path),
        num_threads=1  # Stabilisiert die CPU-Last auf dem Nano
    )

    # Kleiner Puffer für den RAM-Load
    time.sleep(0.2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']

    # ── Warmup (Nicht gemessen) ───────────────────────────
    print(f'  Warmup ({n_warmup} Durchläufe)...')
    for i in range(n_warmup):
        img = preprocess(images[i % len(images)])
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()

    # ── Messung ───────────────────────────────────────────
    print(f'  Messe {len(images)} Bilder...')

    # CPU-Messung resetten
    psutil.cpu_percent(percpu=True, interval=None)
    inference_times = []

    for image in images:
        img = preprocess(image)
        interpreter.set_tensor(input_index, img)

        start = time.perf_counter()
        interpreter.invoke()
        duration_ms = (time.perf_counter() - start) * 1000
        inference_times.append(duration_ms)

    # Metriken erfassen
    ram_peak = get_process_ram_mb()
    cpu_cores = psutil.cpu_percent(percpu=True, interval=None)
    cpu_max = max(cpu_cores)  # Höchste Last auf einem Kern

    avg_ms = float(np.mean(inference_times))

    return {
        'model': os.path.basename(tflite_path),
        'avg_ms': round(avg_ms, 2),
        'std_ms': round(float(np.std(inference_times)), 2),
        'min_ms': round(float(np.min(inference_times)), 2),
        'max_ms': round(float(np.max(inference_times)), 2),
        'fps': round(1000 / avg_ms, 2),
        'ram_peak_mb': ram_peak,
        'cpu_percent': cpu_max,
        'model_size_mb': round(os.path.getsize(tflite_path) / (1024 * 1024), 2),
    }


def run_benchmark():
    # Hier kannst du die Anzahl für die Thesis auf z.B. 30 setzen
    images = load_test_images(n_images=30)
    if not images: return

    results = []
    models = [
        'baseline_fp32.tflite', 'baseline_fp16.tflite', 'baseline_int8.tflite',
        'augmentation_fp32.tflite', 'augmentation_fp16.tflite', 'augmentation_int8.tflite',
    ]

    for model_file in models:
        tflite_path = PATHS['models'] / model_file

        if not tflite_path.exists():
            print(f'Übersprungen: {model_file} nicht gefunden.')
            continue

        print(f'\n── {model_file} ──────────────────────────────')
        result = benchmark_model(str(tflite_path), images)
        results.append(result)

        print(f"  Ø {result['avg_ms']} ms | FPS: {result['fps']} | RAM: {result['ram_peak_mb']}MB")

    # Ergebnisse speichern
    output_path = PATHS['metrics'] / 'jetson_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Zusammenfassung printen
    print('\n' + '─' * 85 + '\n Zusammenfassung\n' + '─' * 85)
    print(f"{'Modell':<30} {'Ø ms':>8} {'±ms':>6} {'FPS':>6} {'Größe':>8} {'RAM':>8} {'CPU':>6}")
    print('-' * 85)
    for r in results:
        print(
            f"{r['model']:<30} {r['avg_ms']:>8.1f} {r['std_ms']:>6.2f} {r['fps']:>6.1f} "
            f"{r['model_size_mb']:>6.1f}MB {r['ram_peak_mb']:>6.1f}MB {r['cpu_percent']:>5.1f}%")


if __name__ == '__main__':
    run_benchmark()