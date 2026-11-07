"""Exercise real CLI model discovery without TensorFlow, data, or inference."""
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'predict_local.py'


class ModelDiscoveryTests(unittest.TestCase):
    def exercise(self, put_in_downloads):
        spec = importlib.util.spec_from_file_location('predictor_under_test', SOURCE)
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        calls = []

        def load_model(path):
            calls.append(Path(path).resolve())
            raise RuntimeError('Offline test stops at model loading; no model is read')

        tf = ModuleType('tensorflow')
        tf.keras = SimpleNamespace(models=SimpleNamespace(load_model=load_model))
        matplotlib, pyplot = ModuleType('matplotlib'), ModuleType('matplotlib.pyplot')
        matplotlib.pyplot = pyplot
        providers = {'tensorflow': tf, 'cv2': ModuleType('cv2'), 'numpy': ModuleType('numpy'), 'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}
        with tempfile.TemporaryDirectory(prefix='deepfake-model-discovery-') as folder:
            root = Path(folder)
            home = root / 'synthetic-user'
            (home / 'Downloads').mkdir(parents=True)
            cwd = root / 'working'
            cwd.mkdir()
            model = (home / 'Downloads' if put_in_downloads else cwd) / 'deepfake_detector.h5'
            model.touch()  # Placeholder never read or deserialized by the fake loader.
            previous = Path.cwd()
            try:
                os.chdir(cwd)
                with patch.dict(sys.modules, providers), patch('os.path.expanduser', return_value=str(home)), redirect_stdout(StringIO()):
                    predictor.main()
                self.assertEqual([model.resolve()], calls)
            finally:
                os.chdir(previous)

    def test_uses_the_current_users_downloads_directory(self):
        self.exercise(put_in_downloads=True)

    def test_preserves_current_directory_fallback(self):
        self.exercise(put_in_downloads=False)


if __name__ == '__main__':
    unittest.main()
