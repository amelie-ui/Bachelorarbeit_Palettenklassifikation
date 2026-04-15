# Jetson Nano TFLite Setup

## Quick Start

```bash
# 1. Install Google Coral TFLite (ARM-optimized)
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl

# 2. Remove newer numpy to use system-wide numpy (1.13.3)
pip uninstall numpy

# 3. Install remaining dependencies
pip install -r requirements.txt
pip install -r requirements-jetson.txt

# 4. Verify
python -c "import tflite_runtime.interpreter; print('✓ TFLite ready')"
```

## Why This Setup?

**Problem:** Standard TFLite wheels don't compile on Jetson Nano's ARM architecture, causing "Illegal instruction" errors when numpy gets updated.

**Solution:** Google Coral provides an ARMv8-optimized wheel that works with Jetson's system-wide numpy (1.13.3).

**Important:** Do NOT run `pip install tflite-runtime` — use the exact Google Coral wheel URL above.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Illegal instruction` | Check: `pip list \| grep numpy` (should be empty) |
| `ModuleNotFoundError: tflite_runtime` | Re-run step 1 with exact URL |
| TFLite import fails | Verify numpy: `python -c "import numpy; print(numpy.__version__)"` |

