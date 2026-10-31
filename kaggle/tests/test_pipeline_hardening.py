"""Dependency-free regressions for TFRecord failure paths (TensorFlow is stubbed)."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'ml_training' / 'prepare_dataset.py'


class ImageSource:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        if self.path.startswith('bad'):
            raise OSError('corrupt image')
        return self

    def __exit__(self, *_):
        return False

    def convert(self, mode):
        return self

    def resize(self, size, method):
        return self

    def tobytes(self):
        return self.path.encode()


class Writer:
    instances = []

    def __init__(self, path):
        self.path = path
        self.records = []
        self.closed = False
        self.instances.append(self)

    def write(self, data):
        self.records.append(data)

    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        Writer.instances = []
        image = types.SimpleNamespace(open=ImageSource,
            Resampling=types.SimpleNamespace(BILINEAR=1))
        pil = types.ModuleType('PIL')
        pil.Image = image
        tf = types.ModuleType('tensorflow')
        tf.io = types.SimpleNamespace(TFRecordWriter=Writer,
            gfile=types.SimpleNamespace(glob=lambda pattern: []))
        with patch.dict(sys.modules, {'PIL': pil, 'tensorflow': tf}):
            spec = importlib.util.spec_from_file_location('dataset_under_test', MODULE)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

    def test_rejects_invalid_shard_size_and_missing_files(self):
        with self.assertRaises(ValueError):
            self.module.process_and_write_tfrecords([], 'unused', shard_size=0)
        with self.assertRaises(FileNotFoundError):
            self.module.load_tfrecord_dataset('missing_*.tfrecord')

    def test_skips_corrupt_pairs_without_empty_shards_and_closes_writers(self):
        self.module.serialize_example = lambda rgb, alpha: rgb + alpha
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'train')
            self.module.process_and_write_tfrecords(
                [('bad0', 'a'), ('rgb1', 'a'), ('bad2', 'a'),
                 ('rgb3', 'a'), ('rgb4', 'a')], prefix, shard_size=2)
            self.assertEqual([len(w.records) for w in Writer.instances], [2, 1])
            self.assertEqual([w.path for w in Writer.instances],
                             [prefix + '_000.tfrecord', prefix + '_001.tfrecord'])
            self.assertTrue(all(w.closed for w in Writer.instances))

    def test_all_corrupt_fails_instead_of_publishing_empty_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No usable'):
                self.module.process_and_write_tfrecords(
                    [('bad0', 'a')], str(Path(directory) / 'train'))
            self.assertEqual(Writer.instances, [])


if __name__ == '__main__':
    unittest.main()
