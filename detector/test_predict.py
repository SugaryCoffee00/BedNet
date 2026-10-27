import copy
from pathlib import Path
import tempfile
import unittest
import predict

class DisplayTests(unittest.TestCase):
    def test_threshold_inclusive(self):
        p = {'boxes': [[0,0,1,1]]*3, 'classes': [1,2,3], 'scores': [.599999,.60,.9]}
        self.assertEqual(predict.visible_indices(p), [1,2])

    def test_only_top_bed(self):
        p = {'boxes': [[0,0,1,1]]*3, 'classes': [0,0,1], 'scores': [.7,.9,.8]}
        self.assertEqual(predict.visible_indices(p), [1,2])

    def test_tied_beds_keep_first(self):
        p = {'boxes': [[0,0,1,1]]*2, 'classes': [0,0], 'scores': [.9,.9]}
        self.assertEqual(predict.visible_indices(p), [0])

    def test_low_bed_does_not_hide_other_objects(self):
        p = {'boxes': [[0,0,1,1]]*2, 'classes': [0,1], 'scores': [.5,.9]}
        self.assertEqual(predict.visible_indices(p), [1])

    def test_filter_does_not_mutate_predictions_or_gate(self):
        p = {'boxes': [[0,0,1,1]]*2, 'classes': [0,1], 'scores': [.9,.4]}
        before = copy.deepcopy(p)
        gate = predict.adapter.policy_gate(p)
        self.assertEqual(gate['decision'], 'BLOCK')
        self.assertEqual(predict.visible_indices(p), [0])
        self.assertEqual(p, before)
        self.assertEqual(predict.adapter.policy_gate(p), gate)

    def test_invalid_thresholds(self):
        for value in (-.1,1.1,float('nan'),float('inf'),True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict.visible_indices({'boxes': [], 'classes': [], 'scores': []}, value)

    def test_empty_stays_empty(self):
        self.assertEqual(predict.visible_indices({'boxes': [], 'classes': [], 'scores': []}), [])

class OutputPathTests(unittest.TestCase):
    def test_shipped_files_rejected(self):
        for name in ('v92_inference_adapter.py', 'checkpoint_manifest.json', 'input_contract.json', 'output_contract.json'):
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                predict.validate_output_paths(predict.adapter.ROOT / name, None, 'image.jpg', 'weights.pt')

    def test_output_collision_rejected(self):
        with tempfile.TemporaryDirectory(dir=predict.adapter.ROOT) as tmp:
            path = Path(tmp) / 'new.json'
            with self.assertRaises(ValueError):
                predict.validate_output_paths(path, path, 'image.jpg', 'weights.pt')

    def test_input_collision_rejected(self):
        with tempfile.TemporaryDirectory(dir=predict.adapter.ROOT) as tmp:
            path = Path(tmp) / 'input.png'
            with self.assertRaises(ValueError):
                predict.validate_output_paths(path, None, path, 'weights.pt')
            with self.assertRaises(ValueError):
                predict.validate_output_paths(path, None, 'image.jpg', path)

    def test_existing_result_preserved(self):
        with tempfile.TemporaryDirectory(dir=predict.adapter.ROOT) as tmp:
            path = Path(tmp) / 'prior.json'
            path.write_text('prior result')
            with self.assertRaises(FileExistsError):
                predict.validate_output_paths(path, None, 'image.jpg', 'weights.pt')
            self.assertEqual(path.read_text(), 'prior result')

    def test_distinct_fresh_outputs_allowed(self):
        with tempfile.TemporaryDirectory(dir=predict.adapter.ROOT) as tmp:
            out, rendered = Path(tmp) / 'new.json', Path(tmp) / 'new.png'
            self.assertEqual(predict.validate_output_paths(out, rendered, 'image.jpg', 'weights.pt'), [out, rendered])

if __name__ == '__main__':
    unittest.main(verbosity=2)
