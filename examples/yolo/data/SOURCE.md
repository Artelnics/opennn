# YOLO external data and pretrained weights

Review date: 2026-09-30. OpenNN bundles no YOLO training images, annotations,
pretrained weights, ONNX models or converted checkpoints. This directory holds
notices only. OpenNN's LGPL software licence does not license external assets.

## What the example actually uses

`yolo [experiment]` requires CUDA FP32. The default experiment is
`v3-pretrained`: Darknet53 with an anchor-based FPN and an external synthetic
dataset. `v3-scratch` skips pretrained backbone loading. `v8-pretrained`
selects the OpenNN v8-style network and can import a user-supplied
`yolov8s.onnx`; `v8-scratch` skips that import. Existing converted checkpoints
in the experiment directory can still be resumed in either mode: `scratch`
does not erase a previous checkpoint or its provenance.

The example searches for local pretrained files and otherwise trains from
scratch. It does not call `Yolo::load_pretrained_backbone`, the separate library
method that downloads Darknet weights. No assets are hosted by Artelnics for
this example. Keep downloads and generated checkpoints outside Git.

## Darknet weights

Joseph Redmon's original Darknet and Alexey Bochkovskiy's fork publish the
same custom **YOLO LICENSE, Version 2, July 29 2016**, declaring Darknet public
domain. `LICENSE_DARKNET.txt` preserves the complete original text, rather
than substituting MIT, Apache 2.0 or the Unlicense.

Original licence sources:

- [pjreddie/darknet at f6afaabcdf85f77e7aff2ec55c020c0e297c77f9](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/LICENSE).
- [AlexeyAB/darknet at 59596d7880f6504768df41d6daa586f5cb2b932f](https://github.com/AlexeyAB/darknet/blob/59596d7880f6504768df41d6daa586f5cb2b932f/LICENSE).

In an issue specifically about the licence of linked weights, repository
collaborator `cenit` identifies the official YOLOv4 release and confirms that
repository resources use the YOLO licence unless explicitly stated otherwise:
[upstream clarification, 2024-05-03](https://github.com/AlexeyAB/darknet/issues/8881#issuecomment-2093052353).
Applying this clarification to `yolov4.conv.137`, an asset in the same
repository's official release, is the basis for recording the YOLO licence
for that backbone. The clarification does not independently establish terms
for every file hosted on Joseph Redmon's separate website.

| File | Original download | Recorded status |
| --- | --- | --- |
| `yolov4.conv.137` | [AlexeyAB official release](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) | Custom YOLO public-domain declaration, supported by the release clarification above. |
| `darknet53.conv.74` | [Joseph Redmon](https://pjreddie.com/media/files/darknet53.conv.74) | Official Darknet model source; the project declares public domain, but a separate confirmation covering this externally hosted checkpoint was not located. Do not record it as independently cleared for redistribution. |
| `yolov3-tiny.weights` | [Joseph Redmon](https://pjreddie.com/media/files/yolov3-tiny.weights) | Same limitation as `darknet53.conv.74`. |

Credit: Joseph Redmon and Ali Farhadi, *YOLOv3: An Incremental Improvement*,
2018; Alexey Bochkovskiy, Chien-Yao Wang and Hong-Yuan Mark Liao,
*YOLOv4: Optimal Speed and Accuracy of Object Detection*, 2020.

The `yolov4.conv.137` download was verified on the review date:
170,038,676 bytes, SHA-256
`11c5286cfbffabc9a3059927401ab3480e9483ceafd3265779a0dc26c769bed0`.
This digest identifies the reviewed file; the GitHub asset metadata supplies
no digest. `LICENSE_DARKNET.txt`: 515 bytes, SHA-256
`2021f0b8d6426fbf9a2f0126f1e8a7d3b1d7027c871b9979fa2543b9557000c4`.

Keep the original licence and this provenance record beside redistributed
YOLOv4 weights or OpenNN conversions. Record the source file's digest,
conversion command, transferred layers and modifications with each conversion.
Do not describe third-party weights as licensed by OpenNN's LGPL.

## Optional Ultralytics YOLOv8 conversion

Ultralytics declares its package and trained models subject to **AGPL-3.0**,
with a separate commercial licence option:
[official licensing guidance](https://www.ultralytics.com/license).
AGPL is a free copyleft licence; it is not a permissive licence and is not
replaced by OpenNN's LGPL when weights are exported to ONNX or OpenNN binary
format. Being an open-source project alone does not establish compliance with
the corresponding-source and licence requirements for a derivative work.

`LICENSE_ULTRALYTICS.txt` is the complete unmodified
[upstream licence at 94b9dfcac7eba6a208092959f983c69c69da5635](https://github.com/ultralytics/ultralytics/blob/94b9dfcac7eba6a208092959f983c69c69da5635/LICENSE):
34,523 bytes, SHA-256
`0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
It documents the optional upstream asset; it does not relicense OpenNN.

`../yolov8s_to_nd.py` imports PyTorch and, when the requested checkpoint is
missing, imports Ultralytics to download `yolov8s.pt`. Loading a pickled
Ultralytics checkpoint can also require that package. The converter transfers
backbone, neck and selected detection-head tensors, changes tensor layout,
and initializes the target classification output for the requested classes.
The example can also import a supplied `yolov8s.onnx` and save converted
OpenNN weights and normalization states.

Redistributing such a conversion requires the applicable upstream copyright
and licence notices, notices of modifications, and the corresponding source
required by AGPL, including the conversion material. Network deployment of a
modified covered program also invokes AGPL's source-access provisions. Any
use under a separate commercial licence must follow that licence's actual
terms. Adding an AGPL text alone does not clear a complete deployment.
No Ultralytics checkpoint or converted output is bundled with OpenNN.

## External datasets

No local dataset was supplied or identified by a file manifest in this review.
These are requirements for the example's supported input paths, not clearance
of arbitrary files that happen to have matching directory names.

| Input | Current selection | Source and redistribution requirements |
| --- | --- | --- |
| External synthetic shapes | Default; `SYNTHETIC_YOLO_IMAGES` and `SYNTHETIC_YOLO_LABELS`, falling back to `synthetic_yolo/images` and `synthetic_yolo/labels` | Expected 416 x 416 images with circle/square/triangle labels. Creator, generator and licence are unrecorded. Obtain those records or use a reproducible generator owned by the project. The internal coloured-block generator in `main.cpp` is currently unused and does not establish provenance for these external files. |
| PASCAL VOC 2007/2012 | Experiment name contains `voc`; `VOC_ROOT` and optional `VOC12_ROOT` | [Original project](http://host.robots.ox.ac.uk/pascal/VOC/). No blanket redistribution grant was confirmed in this review. Identify the selected images, owners and terms, preserve the annotation/source records, and confirm rights before bundling a subset. |
| COCO | Disabled by `use_coco = false`; enabling it uses `COCO_IMAGES` and `COCO_LABELS` | [Official terms at 5e1c4da72464b1c6f068df0c02c91e3000ea62c4](https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/termsofuse.htm): annotations are CC BY 4.0; the consortium does not own the images. Preserve annotation attribution and an image-by-image author/source/licence manifest. Do not apply the annotation licence to all photographs. |
| Raccoon | Disabled by `use_raccoon = false`; `RACCOON_IMAGES` and `RACCOON_LABELS` | [Dat Tran's repository at 27f269ab4d4da6c38ebb16b4a95c5c912a9f1c49](https://github.com/experiencor/raccoon_dataset/tree/27f269ab4d4da6c38ebb16b4a95c5c912a9f1c49) has MIT terms, but its README says images came from Google and Pixabay. The repository licence does not establish permission from every photograph's owner. Record those permissions before redistributing images. |
| BCCD | Experiment name contains `bccd`; `BCCD_IMAGES` and `BCCD_LABELS` | [Shenggan's original BCCD dataset](https://github.com/Shenggan/BCCD_Dataset/tree/d272fb14cdff6e473fafeeeba32aba5f560e9e43) explicitly declares the dataset MIT. Preserve its full `LICENSE` (Copyright (c) 2017 shenggan), credits to cosmicad and akshaylamba, source revision, selected-image manifest and conversion details. The unprovided local `BCCD_yolo` derivative has not been compared with that source. |

## Remaining resolution

The repository itself distributes notices and source code, not the external
assets above. A self-contained default example still needs a reproducible,
licensed synthetic dataset and a documented choice of initial weights.
Use scratch initialization in a new checkpoint directory to avoid importing
unverified weights. Keep optional external inputs documented separately and
verify the actual files before publishing data or converted model artifacts.
