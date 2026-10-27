"""Run the frozen research detector; filter visualization, not model outputs."""
import argparse
import math
from pathlib import Path
import sys

from adapter_v1 import v92_inference_adapter as adapter


def visible_indices(prediction, minimum_confidence=.60):
    if type(minimum_confidence) not in (int, float) or not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
        raise ValueError('Display confidence must be between zero and one')
    p = adapter.validate_predictions(prediction)
    indices = [i for i, score in enumerate(p['scores']) if score >= minimum_confidence]
    beds = [i for i in indices if p['classes'][i] == 0]
    selected = max(beds, key=lambda i: p['scores'][i]) if beds else None
    return [i for i in indices if p['classes'][i] != 0 or i == selected]


def render(image_path, result, destination, minimum_confidence=.60):
    from PIL import Image, ImageDraw
    adapter.validate_result(result)
    destination = adapter.output_path(destination)
    with Image.open(image_path) as source:
        rgb = source.convert('RGB')
    if adapter.rgb_sha256(rgb) != result['observation']['pixel_sha256']:
        raise ValueError('Image pixels changed after inference')
    image = adapter.validated_helper().resize_rgb(rgb, 640)
    draw = ImageDraw.Draw(image)
    prediction = result['detector_prediction']
    indices = visible_indices(prediction, minimum_confidence)
    colors = ['lime', 'red', 'orange', 'cyan', 'magenta', 'yellow']
    for i in indices:
        box = prediction['boxes'][i]
        cls, score = prediction['classes'][i], prediction['scores'][i]
        xy = [v * [image.width, image.height, image.width, image.height][j] for j, v in enumerate(box)]
        draw.rectangle(xy, outline=colors[cls], width=3)
        draw.text((xy[0], xy[1]), f'{adapter.CLASSES[cls]} {score:.1%}', fill=colors[cls], stroke_width=1, stroke_fill='black')
    draw.rectangle((0, 0, image.width, 26), fill='black')
    draw.text((4, 4), f'RESEARCH ONLY | display >= {minimum_confidence:.0%} | no release authority', fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return indices


def validate_output_paths(output, rendered, image, checkpoint):
    paths = [adapter.output_path(p) for p in (output, rendered) if p is not None]
    inputs = {Path(image).resolve(), Path(checkpoint).resolve()}
    if len(set(paths)) != len(paths) or any(p in inputs for p in paths):
        raise ValueError('Output paths must be distinct from each other and all inputs')
    for path in paths:
        if path.exists():
            raise FileExistsError('Output already exists; choose a fresh filename: ' + str(path))
    return paths


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', default=str(root / 'adapter_v1/result.json'))
    parser.add_argument('--render')
    parser.add_argument('--display-confidence', type=float, default=.60)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(argv)
    try:
        visible_indices({'boxes': [], 'classes': [], 'scores': []}, args.display_confidence)
        validate_output_paths(args.out, args.render, args.image, args.checkpoint)
        engine = adapter.Adapter(checkpoint=args.checkpoint, manifest=root / 'adapter_v1/checkpoint_manifest.json', device=args.device)
        result = engine.predict(image=args.image)
        if args.render:
            render(args.image, result, args.render, args.display_confidence)
        adapter.write_json(args.out, result)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
