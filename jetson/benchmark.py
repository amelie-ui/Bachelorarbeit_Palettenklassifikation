# jetson/benchmark.py
import time
import json
import numpy as np
import os
import sys
import psutil

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from config import DATA, PATHS

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def get_process_ram_mb():
    """Prozessspezifischer RAM-Verbrauch via psutil (RSS)."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def load_test_images(n_images: int = 20):
    """
    Lädt n_images Testbilder als numpy-Arrays.
    Kein TF-Dataset auf Jetson – direktes Laden via Pillow.
    """
    from PIL import Image

    images = []
    test_path = PATHS['dataset_test']

    for class_dir in sorted(test_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob('*.jpg'))[:n_images // 3]:
            img = Image.open(img_path).resize(DATA['img_size'])
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)

    print(f'Geladene Testbilder: {len(images)}')
    return images


def preprocess(image):
    """
    Preprocessing für alle Varianten (FP32, FP16, INT8).
    INT8-Eingabetensor ist FLOAT32 — preprocess_input ist im Modell eingebaut.
    """
    return np.expand_dims(image.astype(np.float32), axis=0)


def benchmark_model(tflite_path: str, images: list, n_warmup: int = 5):
    """
    Benchmarkt ein TFLite-Modell auf dem Jetson Nano.

    Args:
        tflite_path: Pfad zur .tflite-Datei
        images:      Liste von numpy-Arrays (224, 224, 3)
        n_warmup:    Anzahl Warmup-Durchläufe (werden nicht gemessen)

    Returns:
        dict mit Metriken
    """
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ── Warmup ────────────────────────────────────────────
    print(f'  Warmup ({n_warmup} Durchläufe)...')
    for i in range(n_warmup):
        img = preprocess(images[i % len(images)])
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

    # ── Messung ───────────────────────────────────────────
    print(f'  Messe {len(images)} Bilder...')

    ram_before = get_process_ram_mb()

    # CPU-Messung initialisieren (erster Aufruf gibt 0.0 zurück → ignorieren)
    # CPU-Messung initialisieren
    psutil.cpu_percent(percpu=True, interval=None)

    inference_times = []

    for image in images:
        img = preprocess(image)
        interpreter.set_tensor(input_details[0]['index'], img)

        start = time.perf_counter()
        interpreter.invoke()
        duration_ms = (time.perf_counter() - start) * 1000

        inference_times.append(duration_ms)

    ram_peak = get_process_ram_mb()
    cpu_cores = psutil.cpu_percent(percpu=True, interval=None)
    cpu_percent = max(cpu_cores)  # höchst belasteter Kern

    # ── Metriken berechnen ────────────────────────────────
    avg_ms        = float(np.mean(inference_times))
    std_ms        = float(np.std(inference_times))
    min_ms        = float(np.min(inference_times))
    max_ms        = float(np.max(inference_times))
    fps           = round(1000 / avg_ms, 2)
    ram_peak_mb   = round(ram_peak, 2)
    model_size_mb = round(os.path.getsize(tflite_path) / (1024 * 1024), 2)

    return {
        'model':          os.path.basename(tflite_path),
        'avg_ms':         round(avg_ms, 2),
        'std_ms':         round(std_ms, 2),
        'min_ms':         round(min_ms, 2),
        'max_ms':         round(max_ms, 2),
        'fps':            fps,
        'ram_peak_mb':    ram_peak_mb,
        'cpu_percent': cpu_percent,  # Max über alle Kerne
        'cpu_cores':   cpu_cores,    # Optional: alle Kerne für JSON
        'model_size_mb':  model_size_mb,
    }


def run_benchmark():
    images = load_test_images(n_images=20)
    results = []

    models = [
        'baseline_fp32.tflite',
        'baseline_fp16.tflite',
        'baseline_int8.tflite',
        'augmentation_fp32.tflite',
        'augmentation_fp16.tflite',
        'augmentation_int8.tflite',
    ]

    for model_file in models:
        tflite_path = str(PATHS['models'] / model_file)

        if not __import__('pathlib').Path(tflite_path).exists():
            print(f'Übersprungen (nicht gefunden): {model_file}')
            continue

        print(f'\n── {model_file} ──────────────────────────────')
        result = benchmark_model(tflite_path, images)
        results.append(result)

        print(f"  Ø Inferenzzeit: {result['avg_ms']} ms (±{result['std_ms']} ms)")
        print(f"  Min/Max:        {result['min_ms']} / {result['max_ms']} ms")
        print(f"  FPS:            {result['fps']}")
        print(f"  Modellgröße:    {result['model_size_mb']} MB")
        print(f"  RAM Peak:       {result['ram_peak_mb']} MB")
        print(f"  CPU-Auslastung: {result['cpu_percent']} %")

    # ── Ergebnisse speichern ──────────────────────────────
    output_path = PATHS['metrics'] / 'jetson_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Benchmark gespeichert: {output_path}')

    # ── Vergleichstabelle ─────────────────────────────────
    print('\n── Zusammenfassung ───────────────────────────────────────────────')
    print(f"{'Modell':<35} {'Ø ms':>6} {'FPS':>6} {'MB':>6} {'RAM':>8} {'CPU':>6}")
    print('-' * 72)
    for r in results:
        print(f"{r['model']:<35} "
              f"{r['avg_ms']:>6.1f} "
              f"{r['fps']:>6.1f} "
              f"{r['model_size_mb']:>6.2f} "
              f"{r['ram_peak_mb']:>7.1f}MB "
              f"{r['cpu_percent']:>5.1f}%")


if __name__ == '__main__':
    run_benchmark()