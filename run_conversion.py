# run_conversion.py
from conversion.convert_fp32 import convert_fp32
from conversion.convert_fp16 import convert_fp16
from conversion.convert_int8 import convert_int8


def run_conversion():

    for model_name in ['baseline', 'augmentation']:
        print(f'\n═══ Konvertierung: {model_name} ════════════════')

        print('\n── FP32 ─────────────────────────────────────')
        convert_fp32(model_name)

        print('\n── FP16 ─────────────────────────────────────')
        convert_fp16(model_name)

        print('\n── INT8 ─────────────────────────────────────')
        convert_int8(model_name)

    print('\n✓ Konvertierung abgeschlossen.')
    print('  Gespeichert in: models/')


if __name__ == '__main__':
    run_conversion()