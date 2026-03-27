import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

PATHS = {
    'dataset':      ROOT / 'data' / 'dataset',
    'dataset_test': ROOT / 'data' / 'dataset_test',
    'models':       ROOT / 'models',
    'outputs':      ROOT / 'outputs',
    'plots':        ROOT / 'outputs' / 'plots',
    'metrics':      ROOT / 'outputs' / 'metrics',
    'grad_cam':     ROOT / 'outputs' / 'grad_cam',
}

# Die einzelnen Dictionarys sollen vielleicht noch weg
DATA = {
    'img_size':          (224, 224),
    'batch_size':        32,
    'validation_split':  0.2,
    'seed':              77,
    'classes':       ['A_PALLET', 'B_PALLET', 'C_PALLET'],
}

MODEL = {
    'backbone': 'MobileNetV2',
    'alpha':    1.0,
    'dropout':  0.2,
}

TRAINING = {
        'learning_rate': 1e-2,
        'epochs':        50,
        'patience':      3,
        'optimizer':     'Adam'

    """phase2': {
        'learning_rate': 1e-5,
        'epochs':        10,
        'patience':      3,
        'frozen_layers': 100,
    },
    """
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)