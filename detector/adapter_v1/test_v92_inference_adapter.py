import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import v92_inference_adapter as a


def pred(*records):
    return {'boxes': [list(r[0]) for r in records], 'classes': [r[1] for r in records], 'scores': [r[2] for r in records]}


BED = ([0., 0., .5, .5], 0, .9)
PRINT = ([.1, .1, .3, .3], 1, .8)


class AdapterTests(unittest.TestCase):
    def test_exact_manifest_accepts(self):
        self.assertEqual(a.validate_manifest(copy.deepcopy(a.EXPECTED_MANIFEST)), a.EXPECTED_MANIFEST)

    def test_profile_missing_or_guessed_rejected(self):
        for value in (None, 'guess', 'torchvision_internal_imagenet_v1', ''):
            m = copy.deepcopy(a.EXPECTED_MANIFEST)
            m['input_profile'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.validate_manifest(m)
        m.pop('input_profile')
        with self.assertRaises(ValueError):
            a.validate_manifest(m)

    def test_manifest_hash_arch_head_class_offset_rejected(self):
        mutations = {'checkpoint_sha256': '0' * 64, 'architecture': 'resnext101',
                     'num_classes_including_background': 6, 'class_names': a.CLASSES[::-1],
                     'label_offset': 0, 'helper_sha256': '0' * 64, 'policy_version': 'guess'}
        for key, val in mutations.items():
            m = copy.deepcopy(a.EXPECTED_MANIFEST)
            m[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_manifest(m)

    def test_manifest_geometry_rejected(self):
        for key, value in {'size': 800, 'aspect_preserved': False, 'long_side_cap_factor': 2., 'internal_min': [640], 'internal_max': 1024, 'PIL_resample': 'BILINEAR'}.items():
            m = copy.deepcopy(a.EXPECTED_MANIFEST)
            m['external_resize'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_manifest(m)

    def test_manifest_norms_rejected(self):
        for where in ('external_normalization', 'internal_normalization'):
            for key in ('mean', 'std'):
                m = copy.deepcopy(a.EXPECTED_MANIFEST)
                m[where][key] = [0, 0, 0]
                with self.subTest(where=where, key=key), self.assertRaises(ValueError):
                    a.validate_manifest(m)

    def test_manifest_no_extra_fields_or_authority(self):
        for key, val in {'profile_override': a.PROFILE, 'release_authority': True, 'research_only': False, 'label_offset': True}.items():
            m = copy.deepcopy(a.EXPECTED_MANIFEST)
            m[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_manifest(m)

    def test_real_hash_binding(self):
        self.assertEqual(a.sha256_file(a.PARENT / 'detection_input_contract.py'), a.HELPER_SHA)
        self.assertEqual(a.validated_helper().LEGACY, a.PROFILE)

    def test_actual_bad_checkpoint_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
            a.validate_binding(__file__, a.EXPECTED_MANIFEST)

    def test_helper_bad_bytes_rejected(self):
        checkpoint = a.PARENT / 'bednet-detector-f44.pt'
        real_hash = a.sha256_file
        with patch.object(a, 'sha256_file', side_effect=lambda p: a.CHECKPOINT_SHA if p == checkpoint else real_hash(p)):
            with self.assertRaisesRegex(ValueError, 'helper SHA256'):
                a.validate_binding(checkpoint, a.EXPECTED_MANIFEST, __file__)

    def test_manifest_validates_before_file_io(self):
        with self.assertRaises(ValueError):
            a.validate_binding('does-not-exist', {})

    def test_helper_internal_transform_rejected(self):
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        helper = a.validated_helper()
        class M:
            pass
        m = M()
        for mean, std, min_size, max_size in [([0,0,0], a.STD, 800, 1333), (a.MEAN, [1,1,1], 800, 1333), (a.MEAN, a.STD, 640, 1333), (a.MEAN, a.STD, 800, 1024)]:
            m.transform = GeneralizedRCNNTransform(min_size, max_size, mean, std)
            with self.assertRaises(ValueError):
                helper.assert_internal_transform(m)

    def test_model_labels_shift_exactly_once(self):
        p = a.convert_model_output({'boxes': torch.tensor([[0., 0., 50., 25.]] * 6), 'labels': torch.arange(1, 7), 'scores': torch.full((6,), .5)}, (100, 50))
        self.assertEqual(p['classes'], list(range(6)))
        self.assertEqual(p['boxes'], [[0., 0., .5, .5]] * 6)

    def test_checkpoint_background_and_bad_offsets_rejected(self):
        for label in (0, -1, 7, 100):
            with self.subTest(label=label), self.assertRaises(ValueError):
                a.convert_model_output({'boxes': torch.tensor([[0., 0., 1., 1.]]), 'labels': torch.tensor([label]), 'scores': torch.tensor([.5])}, (100, 100))

    def test_wrong_dtypes_rejected(self):
        for key, tensor in [('boxes', torch.zeros((1, 4), dtype=torch.float64)), ('labels', torch.ones(1, dtype=torch.float32)), ('scores', torch.ones(1, dtype=torch.float16))]:
            output = {'boxes': torch.tensor([[0., 0., 1., 1.]]), 'labels': torch.tensor([1]), 'scores': torch.tensor([.5])}
            output[key] = tensor
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.convert_model_output(output, (100, 100))

    def test_invalid_scores_rejected(self):
        for value in (-.1, 1.01, float('nan'), float('inf'), True, '0.9'):
            with self.subTest(score=value), self.assertRaises(ValueError):
                a.policy_gate(pred((BED[0], 0, value)))

    def test_invalid_boxes_rejected(self):
        for box in ([0, 0, 0, .5], [.4, 0, .2, .5], [0, .5, .5, .1], [-.1, 0, .5, .5], [0, 0, 1.1, .5], [0, 0, float('nan'), .5], [0, 0, float('inf'), .5], [0, 0, .5], [False, 0, .5, .5]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                a.policy_gate(pred((box, 0, .9)))

    def test_invalid_class_ids_rejected(self):
        for c in (-1, 6, .0, True, '0'):
            with self.subTest(cls=c), self.assertRaises(ValueError):
                a.policy_gate(pred((BED[0], c, .9)))

    def test_bad_array_shapes_rejected(self):
        for p in ({'boxes': [], 'classes': [0], 'scores': []}, {'boxes': [], 'classes': [], 'scores': (),}, {'boxes': [], 'classes': [], 'scores': [], 'decision': 'clear'}):
            with self.assertRaises(ValueError):
                a.policy_gate(p)

    def test_top_one_bed_only_for_gate(self):
        p = pred(BED, ([.5,.5,1.,1.], 0, .95), PRINT)
        original = copy.deepcopy(p)
        gate = a.policy_gate(p)
        self.assertEqual(gate['selected_bed_index'], 1)
        self.assertEqual(gate['decision'], 'CLEAR_CANDIDATE')
        self.assertEqual(p, original)  # all beds/predictions preserved

    def test_bed_threshold_inclusive(self):
        self.assertEqual(a.policy_gate(pred((BED[0], 0, .30)))['decision'], 'CLEAR_CANDIDATE')
        self.assertEqual(a.policy_gate(pred((BED[0], 0, .299999)))['decision'], 'UNCERTAIN')

    def test_bed_tie_uses_first(self):
        self.assertEqual(a.policy_gate(pred(BED, BED))['selected_bed_index'], 0)

    def test_scrap_policy_nonblocking(self):
        self.assertEqual(a.policy_gate(pred(BED, (PRINT[0], 3, 1.)))['decision'], 'CLEAR_CANDIDATE')

    def test_global_alert_even_without_bed(self):
        for cls in (4, 5):
            gate = a.policy_gate(pred(([.8,.8,1.,1.], cls, .20)))
            self.assertEqual(gate['decision'], 'ALERT')
            self.assertIsNone(gate['bed'])

    def test_global_alert_off_bed_and_priority(self):
        self.assertEqual(a.policy_gate(pred(BED, PRINT, ([.8,.8,1.,1.], 4, .2)))['decision'], 'ALERT')

    def test_print_foreign_block_at_threshold(self):
        for cls in (1, 2):
            self.assertEqual(a.policy_gate(pred(BED, (PRINT[0], cls, .20)))['decision'], 'BLOCK')
            self.assertEqual(a.policy_gate(pred(BED, (PRINT[0], cls, .199999)))['decision'], 'CLEAR_CANDIDATE')

    def test_off_bed_objects_nonblocking(self):
        for cls in (1, 2):
            self.assertEqual(a.policy_gate(pred(BED, ([.6,.6,1.,1.], cls, .99)))['decision'], 'CLEAR_CANDIDATE')

    def test_strict_object_area_overlap_not_iou(self):
        # Bed [0,0,.15,1] intersects 15% of full-frame object: equality does NOT block.
        bed = ([0.,0.,.15,1.],0,.9)
        self.assertEqual(a.policy_gate(pred(bed, ([0.,0.,1.,1.],1,.9)))['decision'], 'CLEAR_CANDIDATE')
        bed = ([0.,0.,.150001,1.],0,.9)
        self.assertEqual(a.policy_gate(pred(bed, ([0.,0.,1.,1.],1,.9)))['decision'], 'BLOCK')

    def test_missing_bed_uncertain(self):
        for p in (pred(), pred(PRINT), pred((PRINT[0], 3, .9))):
            self.assertEqual(a.policy_gate(p)['decision'], 'UNCERTAIN')

    def test_bad_metadata_rejected(self):
        for gt in ({'scope': 'partial', 'score_block': .8}, {'scope': 'partial', 'decision': 'CLEAR'}, {'scope': 'partial', 'upstream_classifier_prediction': {}}, {'human_verdict': 'release_clear'}, {'scope': 'partial', 'provenance': {'x': float('nan')}}, []):
            with self.subTest(gt=gt), self.assertRaises((ValueError, TypeError)):
                a.validate_ground_truth(gt)

    def test_gt_scope_preserved_not_reinterpreted(self):
        gt = {'scope': 'spaghetti/failure-only negatives, not all-class negatives', 'positive_annotations': [], 'human_verdict': 'not_assessed'}
        saved = a.validate_ground_truth(gt)
        self.assertEqual(saved, gt)
        self.assertIsNot(saved, gt)

    def test_all_decisions_no_release_authority(self):
        helper = a.validated_helper()
        im = Image.new('RGB', (32, 16), 'red')
        x, resized = helper.prepare(im, a.PROFILE)
        for p in (pred(), pred(BED), pred(BED, PRINT), pred((PRINT[0], 5, .9))):
            r = a.build_result('fixture', im, x, resized, p, a.CHECKPOINT_SHA)
            self.assertEqual(r['schema'], a.SCHEMA)
            self.assertIs(r['release_authority'], False)
            self.assertIs(r['research_gate']['release_authority'], False)
            self.assertTrue(r['research_only'])
            for k in ('decision', 'score_block', 'predicted_gate_target', 'upstream_classifier_prediction'):
                self.assertNotIn(k, r)
            self.assertEqual(r['detector_prediction'], p)

    def test_no_source_metadata_blind_merge(self):
        helper = a.validated_helper()
        image = Image.new('RGB', (32,16), 'red')
        x, resized = helper.prepare(image, a.PROFILE)
        truth = {'scope': 'unit-test partial annotations', 'positive_annotations': [], 'human_verdict': 'not_assessed'}
        result = a.build_result('fixture', image, x, resized, pred(BED), a.CHECKPOINT_SHA, truth)
        self.assertEqual(result['ground_truth'], truth)
        self.assertNotIn('score_block', result)
        self.assertEqual(result['research_gate'], a.policy_gate(pred(BED)))

    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as tmp:
            path = Path(tmp) / 'bad.json'
            for raw in ('{"label_offset":1,"label_offset":0}', '{"score":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    a.read_json(path)

    def test_write_scope_rejected(self):
        with self.assertRaises(ValueError):
            a.output_path(a.PARENT / 'results.json')

    def test_no_cli_profile_or_threshold_override(self):
        for arg in ('--profile', '--bed-threshold', '--occ-threshold'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                a.main(['--checkpoint','x','--manifest','y','--image','z','--out',str(a.ROOT/'x.json'),arg,'.2'])

    def test_cli_requires_all_four_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            a.main([])

    def test_thermal_stop_before_work(self):
        with patch.object(a.ThermalGuard, 'temperature', return_value=75.):
            with self.assertRaisesRegex(RuntimeError, 'Thermal stop'):
                with a.ThermalGuard():
                    self.fail('Must not reach work')

    def test_output_schema_and_authority_rejections(self):
        helper = a.validated_helper()
        im = Image.new('RGB', (32,16), 'red')
        x, resized = helper.prepare(im, a.PROFILE)
        valid = a.build_result('fixture', im, x, resized, pred(BED), a.CHECKPOINT_SHA)
        self.assertEqual(a.validate_result(valid), valid)
        for key, value in [('decision','CLEAR'),('score_block',.3),('predicted_gate_target','release'),('release_authority',True),('research_only',False)]:
            r = copy.deepcopy(valid)
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_result(r)
        r = copy.deepcopy(valid)
        r['research_gate']['decision'] = 'BLOCK'
        with self.assertRaises(ValueError):
            a.validate_result(r)
        r = copy.deepcopy(valid)
        r['preprocessing']['input_dimensions'] = [640,640]
        with self.assertRaises(ValueError):
            a.validate_result(r)

    def test_output_schema_required_keys(self):
        contract = a.read_json(a.ROOT / 'output_contract.json')
        required = {'schema', 'observation', 'checkpoint', 'preprocessing', 'detector_prediction', 'research_gate', 'release_authority', 'research_only'}
        self.assertEqual(set(contract['required']), required)
        self.assertFalse(contract['additionalProperties'])
        self.assertEqual(contract['properties']['release_authority'], {'const': False})
        self.assertEqual(contract['properties']['schema'], {'const': a.SCHEMA})


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
