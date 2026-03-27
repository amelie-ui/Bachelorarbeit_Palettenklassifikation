# jetson/benchmark.py
import time
import json
import numpy as np
import os# jetson/benchmark.py
import time
import json
import numpy as np
import os
import sys

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from config import DATA, PATHS

# TFLite Runtime auf Jetson (kein volles TensorFlow)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# RAM-Messung auf Jetson via /proc/meminfo
def get_ram_usage_mb():
    with open('/proc/meminfo') as f:
        lines = f.readlines()
    mem = {}
    for line in lines:
        parts = line.split()
        mem[parts[0].rstrip(':')] = int(parts[1])
    used = (mem['MemTotal'] - mem['MemAvailable']) / 1024
    return round(used, 2)


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


def preprocess(image, input_dtype, input_scale, input_zero_point):
    """Preprocessing je nach dtype – identisch zu evaluate_tflite.py"""
    if input_dtype == np.int8:
        # MobileNetV2: [0,255] → [-1,1] → INT8
        image = image / 127.5 - 1.0
        image = np.round(image / input_scale + input_zero_point).astype(np.int8)
    else:
        image = image.astype(np.float32)

    return np.expand_dims(image, axis=0)


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

    input_dtype      = input_details[0]['dtype']
    input_scale      = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    # ── Warmup ────────────────────────────────────────────
    # Erste Durchläufe sind durch JIT/Caching langsamer
    print(f'  Warmup ({n_warmup} Durchläufe)...')
    for i in range(n_warmup):
        img = preprocess(images[i % len(images)],
                         input_dtype, input_scale, input_zero_point)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

    # ── Messung ───────────────────────────────────────────
    print(f'  Messe {len(images)} Bilder...')
    ram_before = get_ram_usage_mb()
    inference_times = []

    for image in images:
        img = preprocess(image, input_dtype, input_scale, input_zero_point)

        interpreter.set_tensor(input_details[0]['index'], img)

        start = time.perf_counter()
        interpreter.invoke()
        duration_ms = (time.perf_counter() - start) * 1000

        inference_times.append(duration_ms)

    ram_after = get_ram_usage_mb()

    # ── Metriken berechnen ────────────────────────────────
    avg_ms  = float(np.mean(inference_times))
    std_ms  = float(np.std(inference_times))
    min_ms  = float(np.min(inference_times))
    max_ms  = float(np.max(inference_times))
    fps     = round(1000 / avg_ms, 2)
    ram_delta = round(ram_after - ram_before, 2)
    model_size_mb = round(os.path.getsize(tflite_path) / (1024 * 1024), 2)

    return {
        'model':          os.path.basename(tflite_path),
        'avg_ms':         round(avg_ms, 2),
        'std_ms':         round(std_ms, 2),
        'min_ms':         round(min_ms, 2),
        'max_ms':         round(max_ms, 2),
        'fps':            fps,
        'ram_delta_mb':   ram_delta,
        'ram_after_mb':   ram_after,
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

        print(f"  Ø Inferenzzeit: {result['avg_ms']} ms "
              f"(±{result['std_ms']} ms)")
        print(f"  Min/Max:        {result['min_ms']} / {result['max_ms']} ms")
        print(f"  FPS:            {result['fps']}")
        print(f"  Modellgröße:    {result['model_size_mb']} MB")
        print(f"  RAM-Delta:      {result['ram_delta_mb']} MB")

    # ── Ergebnisse speichern ──────────────────────────────
    output_path = PATHS['metrics'] / 'jetson_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Benchmark gespeichert: {output_path}')

    # ── Vergleichstabelle ─────────────────────────────────
    print('\n── Zusammenfassung ───────────────────────────────────────────')
    print(f"{'Modell':<35} {'Ø ms':>6} {'FPS':>6} {'MB':>6} {'RAM Δ':>8}")
    print('-' * 65)
    for r in results:
        print(f"{r['model']:<35} "
              f"{r['avg_ms']:>6.1f} "
              f"{r['fps']:>6.1f} "
              f"{r['model_size_mb']:>6.2f} "
              f"{r['ram_delta_mb']:>7.1f}MB")


if __name__ == '__main__':
    run_benchmark()
import sys

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from config import DATA, PATHS

# TFLite Runtime auf Jetson (kein volles TensorFlow)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# RAM-Messung auf Jetson via /proc/meminfo
def get_ram_usage_mb():
    with open('/proc/meminfo') as f:
        lines = f.readlines()
    mem = {}
    for line in lines:
        parts = line.split()
        mem[parts[0].rstrip(':')] = int(parts[1])
    used = (mem['MemTotal'] - mem['MemAvailable']) / 1024
    return round(used, 2)


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


def preprocess(image, input_dtype, input_scale, input_zero_point):
    """Preprocessing je nach dtype – identisch zu evaluate_tflite.py"""
    if input_dtype == np.int8:
        # MobileNetV2: [0,255] → [-1,1] → INT8
        image = image / 127.5 - 1.0
        image = np.round(image / input_scale + input_zero_point).astype(np.int8)
    else:
        image = image.astype(np.float32)

    return np.expand_dims(image, axis=0)


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

    input_dtype      = input_details[0]['dtype']
    input_scale      = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    # ── Warmup ────────────────────────────────────────────
    # Erste Durchläufe sind durch JIT/Caching langsamer
    print(f'  Warmup ({n_warmup} Durchläufe)...')
    for i in range(n_warmup):
        img = preprocess(images[i % len(images)],
                         input_dtype, input_scale, input_zero_point)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

    # ── Messung ───────────────────────────────────────────
    print(f'  Messe {len(images)} Bilder...')
    ram_before = get_ram_usage_mb()
    inference_times = []

    for image in images:
        img = preprocess(image, input_dtype, input_scale, input_zero_point)

        interpreter.set_tensor(input_details[0]['index'], img)

        start = time.perf_counter()
        interpreter.invoke()
        duration_ms = (time.perf_counter() - start) * 1000

        inference_times.append(duration_ms)

    ram_after = get_ram_usage_mb()

    # ── Metriken berechnen ────────────────────────────────
    avg_ms  = float(np.mean(inference_times))
    std_ms  = float(np.std(inference_times))
    min_ms  = float(np.min(inference_times))
    max_ms  = float(np.max(inference_times))
    fps     = round(1000 / avg_ms, 2)
    ram_delta = round(ram_after - ram_before, 2)
    model_size_mb = round(os.path.getsize(tflite_path) / (1024 * 1024), 2)

    return {
        'model':          os.path.basename(tflite_path),
        'avg_ms':         round(avg_ms, 2),
        'std_ms':         round(std_ms, 2),
        'min_ms':         round(min_ms, 2),
        'max_ms':         round(max_ms, 2),
        'fps':            fps,
        'ram_delta_mb':   ram_delta,
        'ram_after_mb':   ram_after,
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

        print(f"  Ø Inferenzzeit: {result['avg_ms']} ms "
              f"(±{result['std_ms']} ms)")
        print(f"  Min/Max:        {result['min_ms']} / {result['max_ms']} ms")
        print(f"  FPS:            {result['fps']}")
        print(f"  Modellgröße:    {result['model_size_mb']} MB")
        print(f"  RAM-Delta:      {result['ram_delta_mb']} MB")

    # ── Ergebnisse speichern ──────────────────────────────
    output_path = PATHS['metrics'] / 'jetson_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Benchmark gespeichert: {output_path}')

    # ── Vergleichstabelle ─────────────────────────────────
    print('\n── Zusammenfassung ───────────────────────────────────────────')
    print(f"{'Modell':<35} {'Ø ms':>6} {'FPS':>6} {'MB':>6} {'RAM Δ':>8}")
    print('-' * 65)
    for r in results:
        print(f"{r['model']:<35} "
              f"{r['avg_ms']:>6.1f} "
              f"{r['fps']:>6.1f} "
              f"{r['model_size_mb']:>6.2f} "
              f"{r['ram_delta_mb']:>7.1f}MB")


if __name__ == '__main__':
    run_benchmark()