"""Frozen v92/f44 research adapter. Never authorizes physical release.

API: Adapter(checkpoint=..., manifest=..., device='cpu').predict(image=...,
     ground_truth=None, render=None) -> bednet-detector-inference-v1 dict.
No profile inference, thresholds overrides, training, downloads or classifier merge.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
VERSION = 'v92-f44-adapter-1.0.0'
SCHEMA = 'bednet-detector-inference-v1'
CHECKPOINT_SHA = 'd4c97d7aa6112754cf1e994d9df41ee5175c753c862c06c5befaa603a5fe3ab5'
HELPER_SHA = 'e0c2b531d601ace3f9d78499a7a971e880aa225cc870efd686aec3cac1a45948'
PROFILE = 'legacy_external_plus_internal_imagenet_v1'
CLASSES = ['bed', 'print', 'foreign', 'scrap', 'spaghetti', 'failure']
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
POLICY = 'v92-f44-research-policy-v1'
EXPECTED_MANIFEST = {
    'schema': 'bednet-checkpoint-specific-manifest-v1',
    'checkpoint_sha256': CHECKPOINT_SHA,
    'architecture': 'torchvision.fasterrcnn_resnet50_fpn_v2',
    'num_classes_including_background': 7,
    'class_names': CLASSES,
    'label_offset': 1,
    'input_profile': PROFILE,
    'external_resize': {'size': 640, 'aspect_preserved': True,
        'long_side_cap_factor': 1.6, 'PIL_resample': 'same default as trainer',
        'internal_min': [800], 'internal_max': 1333},
    'external_normalization': {'mean': MEAN, 'std': STD},
    'internal_normalization': {'mean': MEAN, 'std': STD},
    'helper_sha256': HELPER_SHA,
    'policy_version': POLICY,
    'research_only': True,
    'release_authority': False,
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rgb_sha256(image):
    """Historical pixel convention: decoded RGB bytes + str((width,height))."""
    image = image.convert('RGB')
    return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()


def tensor_sha256(tensor):
    """SHA256 over contiguous CHW float32 bytes, before internal transform."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_json(path):
    def duplicate_reject(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('Duplicate JSON field: ' + k)
            d[k] = v
        return d
    return json.loads(Path(path).read_text(), object_pairs_hook=duplicate_reject,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + v)))


def validate_manifest(manifest):
    # Canonical JSON comparison distinguishes bool from int and int from float;
    # no missing/default/extra fields, profile guessing or override route.
    if not isinstance(manifest, dict) or json.dumps(manifest, sort_keys=True, allow_nan=False) != json.dumps(EXPECTED_MANIFEST, sort_keys=True):
        raise ValueError('Manifest must exactly match this frozen checkpoint/profile/class/geometry/helper/policy contract')
    return copy.deepcopy(manifest)


def validate_binding(checkpoint, manifest, helper_path=None):
    manifest = validate_manifest(manifest)
    if sha256_file(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Unsupported checkpoint SHA256')
    helper_path = PARENT / 'detection_input_contract.py' if helper_path is None else Path(helper_path)
    if sha256_file(helper_path) != HELPER_SHA:
        raise ValueError('Input helper SHA256 mismatch; import forbidden')
    return manifest


def validated_helper():
    path = PARENT / 'detection_input_contract.py'
    if sha256_file(path) != HELPER_SHA:
        raise ValueError('Input helper SHA256 mismatch; import forbidden')
    # Compile the exact hashed bytes without writing parent __pycache__.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('Helper changed during read')
    import types
    module = types.ModuleType('_v92_validated_input_helper')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def validate_predictions(prediction):
    if not isinstance(prediction, dict) or set(prediction) != {'boxes', 'classes', 'scores'}:
        raise ValueError('Expected explicit boxes/classes/scores prediction namespace only')
    boxes, classes, scores = (prediction[k] for k in ('boxes', 'classes', 'scores'))
    if not all(isinstance(v, list) for v in (boxes, classes, scores)) or len(boxes) != len(classes) or len(boxes) != len(scores):
        raise ValueError('Prediction arrays must be aligned lists')
    for box, cls, score in zip(boxes, classes, scores):
        if type(cls) is not int or cls not in range(6):
            raise ValueError('Invalid zero-based detector class')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid detector score')
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box):
            raise ValueError('Invalid normalized detector box')
        if not box[0] < box[2] or not box[1] < box[3]:
            raise ValueError('Degenerate/reversed detector box')
    return copy.deepcopy(prediction)


def convert_model_output(output, input_dimensions):
    import torch
    if not isinstance(output, dict) or set(output) != {'boxes', 'labels', 'scores'}:
        raise ValueError('Unexpected model output fields')
    b, l, s = (output[k] for k in ('boxes', 'labels', 'scores'))
    if b.dtype != torch.float32 or s.dtype != torch.float32 or l.dtype != torch.int64:
        raise ValueError('Expected FP32 boxes/scores and int64 checkpoint labels')
    if b.ndim != 2 or b.shape[1] != 4 or l.ndim != 1 or s.ndim != 1 or len(b) != len(l) or len(b) != len(s):
        raise ValueError('Invalid model output shape')
    labels = l.detach().cpu().tolist()
    if any(type(c) is not int or c not in range(1, 7) for c in labels):
        raise ValueError('Unexpected checkpoint label offset/range')
    w, h = input_dimensions
    boxes = [[v / [w, h, w, h][i] for i, v in enumerate(box)] for box in b.detach().cpu().tolist()]
    return validate_predictions({'boxes': boxes, 'classes': [c - 1 for c in labels], 'scores': s.detach().cpu().tolist()})


def policy_gate(prediction):
    p = validate_predictions(prediction)
    beds = [i for i, (c, s) in enumerate(zip(p['classes'], p['scores'])) if c == 0 and s >= .30]
    selected = max(beds, key=lambda i: p['scores'][i]) if beds else None
    alerts = [i for i, (c, s) in enumerate(zip(p['classes'], p['scores'])) if c in (4, 5) and s >= .20]
    blockers = []
    overlaps = {}
    if selected is not None:
        bed = p['boxes'][selected]
        for i, (box, cls, score) in enumerate(zip(p['boxes'], p['classes'], p['scores'])):
            if cls not in (1, 2) or score < .20:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            intersect = max(0., min(box[2], bed[2]) - max(box[0], bed[0])) * max(0., min(box[3], bed[3]) - max(box[1], bed[1]))
            overlap = intersect / area
            overlaps[str(i)] = overlap
            if overlap > .15:
                blockers.append(i)
    decision = 'ALERT' if alerts else 'UNCERTAIN' if selected is None else 'BLOCK' if blockers else 'CLEAR_CANDIDATE'
    return {'policy_version': POLICY, 'decision': decision,
            'selected_bed_index': selected,
            'bed': None if selected is None else {'box': p['boxes'][selected], 'score': p['scores'][selected], 'class': 0},
            'blocking_prediction_indices': blockers, 'global_alert_prediction_indices': alerts,
            'object_area_overlap': overlaps, 'scrap_blocks': False,
            'limitations': 'Research policy only; no enrollment, visibility certification, or release permission.',
            'release_authority': False}


GT_KEYS = {'positive_annotations', 'positive_annotation_status', 'human_verdict',
           'occupancy_human_verdict', 'observability', 'provenance', 'graded_at',
           'cohort', 'split', 'training_eligible', 'scope', 'source_artifact', 'source_artifact_sha256',
           'gate_target', 'binary_release_label', 'classification_eligible', 'release_verification_eligible', 'visibility'}


def validate_ground_truth(gt):
    if gt is None:
        return None
    if not isinstance(gt, dict) or not set(gt) <= GT_KEYS:
        raise ValueError('Ground truth accepts only explicit scoped fields; classifier/source row merge forbidden')
    if not isinstance(gt.get('scope'), str) or not gt['scope']:
        raise ValueError('Explicit annotation scope required; missing boxes never imply absence')
    # Ensure serializable finite metadata; no synthetic grading or inferred targets.
    json.dumps(gt, allow_nan=False)
    return copy.deepcopy(gt)


def build_result(image_path, image, tensor, resized, prediction, loaded_sha, ground_truth=None):
    if loaded_sha != CHECKPOINT_SHA:
        raise ValueError('Cannot attribute output to an unsupported checkpoint')
    p = validate_predictions(prediction)
    gt = validate_ground_truth(ground_truth)
    result = {'schema': SCHEMA,
        'observation': {'path': str(Path(image_path).resolve()), 'pixel_sha256': rgb_sha256(image),
                        'pixel_hash_convention': 'sha256(RGB_bytes + str((width, height)).encode())',
                        'original_dimensions': list(image.size)},
        'checkpoint': {'loaded_sha256': loaded_sha, 'input_profile': PROFILE, 'adapter_version': VERSION},
        'preprocessing': {'input_dimensions': list(resized.size), 'input_tensor_sha256': tensor_sha256(tensor),
                          'tensor_dtype': 'float32', 'tensor_layout': 'CHW', 'tensor_hash_convention': 'sha256(contiguous_CHW_float32_bytes)',
                          'resized_rgb_pixel_sha256': rgb_sha256(resized)},
        'detector_prediction': p, 'research_gate': policy_gate(p),
        'research_only': True, 'release_authority': False}
    if gt is not None:
        result['ground_truth'] = gt
    return validate_result(result)


def validate_result(result):
    """Fail closed on namespace, authority, identity and gate disagreement."""
    import re
    required = {'schema', 'observation', 'checkpoint', 'preprocessing', 'detector_prediction',
                'research_gate', 'research_only', 'release_authority'}
    if not isinstance(result, dict) or not required <= set(result) or set(result) - required - {'ground_truth'}:
        raise ValueError('Unexpected output namespace or inherited classifier field')
    if result['schema'] != SCHEMA or result['research_only'] is not True or result['release_authority'] is not False:
        raise ValueError('Output must be research-only without release authority')
    obs, pre = result['observation'], result['preprocessing']
    if set(obs) != {'path', 'pixel_sha256', 'pixel_hash_convention', 'original_dimensions'}:
        raise ValueError('Invalid observation namespace')
    if not isinstance(obs['path'], str) or not Path(obs['path']).is_absolute():
        raise ValueError('Observation path must be absolute')
    if not isinstance(obs['original_dimensions'], list) or len(obs['original_dimensions']) != 2 or any(type(v) is not int or v <= 0 for v in obs['original_dimensions']):
        raise ValueError('Invalid original dimensions')
    if result['checkpoint'] != {'loaded_sha256': CHECKPOINT_SHA, 'input_profile': PROFILE, 'adapter_version': VERSION}:
        raise ValueError('Invalid output checkpoint identity')
    if set(pre) != {'input_dimensions','input_tensor_sha256','tensor_dtype','tensor_layout','tensor_hash_convention','resized_rgb_pixel_sha256'}:
        raise ValueError('Invalid preprocessing namespace')
    w, h = obs['original_dimensions']
    scale = min(640/min(w,h), 1024/max(w,h))
    if pre['input_dimensions'] != [int(round(w*scale)),int(round(h*scale))] or pre['tensor_dtype'] != 'float32' or pre['tensor_layout'] != 'CHW':
        raise ValueError('Output preprocessing geometry/dtype mismatch')
    if obs['pixel_hash_convention'] != 'sha256(RGB_bytes + str((width, height)).encode())' or pre['tensor_hash_convention'] != 'sha256(contiguous_CHW_float32_bytes)':
        raise ValueError('Unexpected output hash convention')
    for digest in (obs['pixel_sha256'], pre['input_tensor_sha256'], pre['resized_rgb_pixel_sha256']):
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid output pixel/tensor hash')
    prediction = validate_predictions(result['detector_prediction'])
    if result['research_gate'] != policy_gate(prediction):
        raise ValueError('Output gate disagrees with frozen detector policy')
    if 'ground_truth' in result:
        validate_ground_truth(result['ground_truth'])
    json.dumps(result, allow_nan=False)
    return copy.deepcopy(result)


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Research writes must remain under adapter_v1')
    return path


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(raw)
    temp.replace(path)


class ThermalGuard:
    """CUDA only: abort this process at >=75C or telemetry failure; no HW writes."""
    @staticmethod
    def temperature():
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True, timeout=5)
        temps = [float(line.strip()) for line in text.splitlines() if line.strip()]
        if not temps or any(not math.isfinite(t) for t in temps):
            raise RuntimeError('Invalid GPU thermal telemetry')
        return max(temps)

    def __enter__(self):
        self.stop = threading.Event()
        self.peak = self.temperature()
        if self.peak >= 75:
            raise RuntimeError('Thermal stop >=75C before CUDA inference')
        def watch():
            while not self.stop.wait(.25):
                try:
                    self.peak = max(self.peak, self.temperature())
                    if self.peak < 75:
                        continue
                    message = 'Thermal stop >=75C'
                except Exception as e:
                    message = 'Thermal telemetry failed: ' + str(e)
                print(message + '; research process aborted without hardware changes', file=sys.stderr, flush=True)
                os._exit(75)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


class Adapter:
    def __init__(self, *, checkpoint, manifest, device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Device must be cpu or cuda')
        self.checkpoint = Path(checkpoint).resolve()
        self.manifest = validate_binding(self.checkpoint, read_json(manifest) if isinstance(manifest, (str, Path)) else manifest)
        self.helper = validated_helper()
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent fallback')
        self.device = device
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=7)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if state['roi_heads.box_predictor.cls_score.weight'].shape[0] != 7 or state['roi_heads.box_predictor.bbox_pred.weight'].shape[0] != 28:
            raise ValueError('Unexpected checkpoint predictor head')
        self.model.load_state_dict(state, strict=True)
        if sha256_file(self.checkpoint) != CHECKPOINT_SHA:
            raise ValueError('Checkpoint changed while loading')
        self.loaded_sha = CHECKPOINT_SHA
        self.model.float().eval()
        self.helper.assert_internal_transform(self.model)
        # Move during guarded predict, not an unmonitored CUDA allocation.

    def predict(self, *, image, ground_truth=None, render=None):
        import contextlib
        import torch
        from PIL import Image
        gt = validate_ground_truth(ground_truth)
        if render is not None:
            output_path(render)
        with Image.open(image) as src:
            rgb = src.convert('RGB')
        if min(rgb.size) <= 0:
            raise ValueError('Invalid image dimensions')
        x, resized = self.helper.prepare(rgb, PROFILE, 640)
        if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError('Invalid input tensor')
        self.helper.assert_internal_transform(self.model)
        if self.model.training or any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('Inference requires eval mode and FP32 parameters')
        guard = ThermalGuard() if self.device == 'cuda' else contextlib.nullcontext()
        with guard, torch.inference_mode():
            self.model.to(self.device)
            output = self.model([x.to(self.device)])[0]
            if self.device == 'cuda':
                torch.cuda.synchronize()
            prediction = convert_model_output(output, resized.size)
        result = build_result(image, rgb, x, resized, prediction, self.loaded_sha, gt)
        if render is not None:
            render_result(resized, result, render)
        return result


def render_result(resized_rgb, result, path):
    from PIL import ImageDraw
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = resized_rgb.copy()
    draw = ImageDraw.Draw(im)
    p = result['detector_prediction']
    selected = result['research_gate']['selected_bed_index']
    for i, (box, cls, score) in enumerate(zip(p['boxes'], p['classes'], p['scores'])):
        # All non-bed model predictions preserved/displayed; top1 qualifying bed only.
        if cls == 0 and i != selected:
            continue
        b = [v * [im.width, im.height, im.width, im.height][k] for k, v in enumerate(box)]
        color = ['lime', 'red', 'orange', 'cyan', 'magenta', 'yellow'][cls]
        draw.rectangle(b, outline=color, width=2)
        draw.text((b[0], b[1]), f'{CLASSES[cls]} {score:.4f}', fill=color, stroke_width=1, stroke_fill='black')
    draw.rectangle((0, 0, im.width, 27), fill='black')
    draw.text((4, 3), 'RESEARCH ONLY - NO RELEASE AUTHORITY | ' + result['research_gate']['decision'], fill='white')
    im.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('checkpoint', 'manifest', 'image', 'out'):
        ap.add_argument('--' + arg, required=True)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--render', help='Optional resized RGB detector overlay path under adapter_v1')
    args = ap.parse_args(argv)
    try:
        output_path(args.out)
        if args.render:
            output_path(args.render)
        adapter = Adapter(checkpoint=args.checkpoint, manifest=args.manifest, device=args.device)
        result = adapter.predict(image=args.image, render=args.render)
        write_json(args.out, result)
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
