# Frozen f44 detector — research model card

**Not approved for automatic printer release.** These are the existing six-class Faster R-CNN weights behind the displayed results, packaged with a verified inference adapter. No new training was performed for this release.

## Weights and use

Download `bednet-detector-f44.pt` from [the research prerelease](https://github.com/SugaryCoffee00/BedNet/releases/tag/detector-f44-research-v1.0.0). Install `requirements.txt`, then run from repository root:

```bash
python detector/predict.py --checkpoint detector/bednet-detector-f44.pt --image frame.jpg --out detector/adapter_v1/result.json --render detector/adapter_v1/result.png
```

`--device cpu` is the default. Optional CUDA uses FP32 and stops the research process at >=75°C or failed temperature telemetry without changing hardware settings. Outputs must remain under `detector/adapter_v1/`. The wrapper requires fresh, distinct output filenames and rejects collisions with input files; existing outputs or shipped code/contracts are never overwritten. No downloads occur during inference.

Visualization defaults to `--display-confidence 0.60`. All non-bed predictions meeting that threshold are shown; only the highest-scoring qualifying bed is drawn. **Only the display is filtered**: JSON retains every model prediction, and policy decisions are computed from the unfiltered predictions. No visible box does not certify an empty bed.

## Specification

| Property | Value |
|---|---|
| Architecture | Torchvision `fasterrcnn_resnet50_fpn_v2` |
| Parameters | 43,281,778 total; ROI predictor 35,875 |
| Weights | FP32 state dictionary; 173,483,050 bytes |
| Head | Seven outputs including background |
| Output classes | bed0, print1, foreign2, scrap3, spaghetti4, failure5 |
| External resize | Aspect-preserving; shortest-side target640, longest-side cap1024; historical PIL default |
| Internal resize | min800 / max1333 |
| Normalization | Legacy external AND internal ImageNet mean[.485,.456,.406],std[.229,.224,.225] |
| Export | PyTorch only; no detector ONNX export included |
| Latency | Not benchmarked for this publication |
| Runtime verified | torch2.5.1+cu124, torchvision0.20.1+cu124, numpy1.26.4, Pillow12.3.0 |
| License | Project CC-BY4.0; preserve attribution and third-party library notices |

SHA256: `d4c97d7aa6112754cf1e994d9df41ee5175c753c862c06c5befaa603a5fe3ab5`.

Historical training manually ImageNet-normalized before Faster R-CNN's internal normalization; an old external evaluator used raw0–1 and square resizing. The adapter is bound to this exact checkpoint and reproduces the preserved dataset input path. Do not remove normalization from these weights. Future fresh Torchvision training should normally use raw0–1 plus internal normalization once, with an explicit versioned input contract. The legacy ConvNeXt verifier is a different model and is not included here.

`adapter_v1/v92_inference_adapter.py` is unchanged from the verified research adapter. The helper SHA, checkpoint SHA, architecture, label offset, class map, geometry and research policy are strictly checked. Model labels1..6 are shifted once to output classes0..5. Duplicate/invalid JSON, nonfinite scores, bad boxes, wrong transforms and inherited classifier fields are rejected.

## Output and policy

Output schema: `bednet-detector-inference-v1`; namespaces: `observation`, `checkpoint`, `preprocessing`, `detector_prediction`, `research_gate`, optional explicitly scoped `ground_truth`. Every result is research-only with `release_authority=false`.

Frozen research policy: highest-score bed>=.30; print/foreign>=.20 and intersection/OBJECT-area>.15 yields BLOCK; scrap does not block; spaghetti/failure>=.20 yields whole-frame ALERT even without a bed; no qualifying bed yields UNCERTAIN; otherwise CLEAR_CANDIDATE. These are diagnostic semantics, not a qualified production policy or permission to release. The60% display setting changes none of them.

## Evidence and limitations

A normalization-only replay used115 previously available observations:67 current TRAIN-cohort,28 older validation,20 old R2 development.65/67 TRAIN observations occurred in this detector's historical training corpus. Both arms used identical aspect-preserving geometry; this was not a reproduction of the old evaluator's square geometry.

| Cohort | Source-listed positive-box matches: raw → compatible input |
|---|---|
| TRAIN |71/148 →132/148 |
| Older validation |5/60 →21/60 |
| Old R2 development |No comparable box ground truth |

Class-aware matching at score>=.5/IoU>=.5 measures recovery of listed positives. **Annotations are partial: this is not precision, exhaustive recall, absence certification or untouched-test generalization.** Gains are nonuniform; an older adequate scrap-only case changes CLEAR→ALERT. Bed localization and useful clear recognition remain weak on some views. Scores are not calibrated safety probabilities.

The original private research harness passed41 unit tests,230 saved gate comparisons and6 actual forwards, independently rerun with zero score/normalized-box discrepancy under an unchanged absolute tolerance1e-5. This establishes plumbing only. The public data-independent tests below omit private images, frame judgments and source-specific fixture utilities. Private regression data is not distributed. No TEST-audit rescoring or training was performed to publish this checkpoint.

Historical logs support lineage, but the original launch command/launch-time trainer hash was not recovered. No camera enrollment, drift or visibility qualification, image-freshness guarantee, operational safety certification or live printer integration is included.

## Public tests

```bash
python detector/detection_input_contract.py --self-test
python detector/adapter_v1/test_v92_inference_adapter.py
python detector/test_predict.py
```

Unit-test geometry examples are synthetic test fixtures, not human grades or model-performance claims. The release also includes aggregate publication verification and SHA256 checksums; private farm images and machine-specific inference outputs are not uploaded.

## Attribution

BedNet © SugaryCoffee — https://github.com/SugaryCoffee00/BedNet — CC-BY4.0. See the repository LICENSE. The root classifier files and [historical classifier documentation](../legacy/README.md) are separate legacy artifacts; Hugging Face is not modified by this GitHub release.
