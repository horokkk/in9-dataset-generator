# IN-9 Dataset Generator with SAM 3

Regeneration of the [ImageNet-9 (IN-9)](https://github.com/MadryLab/backgrounds_challenge) training dataset using **SAM 3** (Segment Anything Model 3) instead of GrabCut.

IN-9 is a diagnostic benchmark for measuring how much a model relies on image backgrounds for classification, introduced in:

> Xiao et al., *"Noise or Signal: The Role of Image Backgrounds in Object Recognition"*, ICLR 2021
> ([arXiv:2006.09994](https://arxiv.org/abs/2006.09994))

## Motivation

The original IN-9 training data download links (Dropbox) are **all dead** as of 2024, with no author response ([Issue #9](https://github.com/MadryLab/backgrounds_challenge/issues/9), [#11](https://github.com/MadryLab/backgrounds_challenge/issues/11), [#12](https://github.com/MadryLab/backgrounds_challenge/issues/12)). Only the test set (279MB) remains available via GitHub Releases.

This script regenerates the full IN-9 training set from ImageNet ILSVRC2012 with two key improvements:

| | Original (ICLR 2021) | Ours |
|---|---|---|
| **FG/BG Segmentation** | GrabCut (GMM-based, noisy boundaries) | SAM 3 (deep learning, precise boundaries) |
| **Variants** | 6 implemented in code | All 9 variants |
| **Scope** | Single synset, val only | All 9 superclasses, train/val |

## Background: BG-Gap Metric

```
BG-Gap = Acc(Mixed-Same) - Acc(Mixed-Rand)
```

- **Mixed-Same**: FG composited onto BG from the **same** class → BG is a helpful "hint"
- **Mixed-Rand**: FG composited onto BG from a **random different** class → BG is noise
- **Large BG-Gap** → model relies on backgrounds
- **Small BG-Gap** → model focuses on foreground

## Generated Variants

| Variant | Description | Segmentation |
|---------|-------------|:---:|
| `original` | ImageNet images reorganized into 9 superclasses | - |
| `only_bg_b` | Bounding box blacked out | - |
| `only_bg_t` | Bounding box replaced with tiled background | - |
| `only_fg` | Foreground only (background = black) | SAM 3 |
| `no_fg` | Background only (foreground = black) | SAM 3 |
| `mixed_same` | FG + BG from same superclass | SAM 3 |
| `mixed_rand` | FG + BG from random different superclass | SAM 3 |
| `mixed_next` | FG + BG from next superclass `(i+1) % 9` | SAM 3 |
| `in9l` | Balanced sampling (~20K/class, no segmentation) | - |

## 9 Superclasses

| ID | Superclass | # ImageNet Classes |
|----|------------|:--:|
| 0 | Dog | 116 |
| 1 | Bird | 52 |
| 2 | Vehicle | 42 |
| 3 | Reptile | 36 |
| 4 | Carnivore | 35 |
| 5 | Insect | 27 |
| 6 | Instrument | 26 |
| 7 | Primate | 20 |
| 8 | Fish | 16 |

370 ImageNet classes → 9 superclasses (remaining 630 excluded). Mapping defined in `in_to_in9.json`.

## Requirements

- Python 3.8+
- PyTorch, torchvision
- [SAM 3](https://github.com/facebookresearch/sam3) (or SAM 2 as fallback)
- OpenCV (`opencv-python-headless`)
- ImageNet ILSVRC2012 images + bounding box annotations

```bash
pip install git+https://github.com/facebookresearch/sam3.git opencv-python-headless
```

## Usage

### Basic

```bash
python generate_in9.py \
    --in_dir /path/to/imagenet \
    --out_dir /path/to/output \
    --ann_dir /path/to/annotations \
    --split train
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--in_dir` | (required) | ImageNet root (contains `train/`, `val/`) |
| `--out_dir` | (required) | Output directory |
| `--ann_dir` | (required) | Bounding box annotation directory |
| `--split` | `train` | `train` or `val` |
| `--segmentor` | `sam3` | `sam3` or `grabcut` (for original reproduction) |
| `--sam_checkpoint` | None | SAM model checkpoint path |
| `--sam_model_cfg` | None | SAM model config name |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--superclasses` | all | Comma-separated indices (e.g., `0,1,2`) |
| `--skip_mixed` | False | Skip mixed variant generation |
| `--skip_in9l` | False | Skip IN-9L generation |
| `--seed` | 42 | Random seed |

### Partial Run (for testing)

```bash
# Process only superclass 0 (Dog), skip mixed and IN-9L
python generate_in9.py \
    --in_dir /path/to/imagenet \
    --out_dir /path/to/output \
    --ann_dir /path/to/annotations \
    --superclasses 0 --skip_mixed --skip_in9l
```

### Reproduce Original (GrabCut)

```bash
python generate_in9.py \
    --in_dir /path/to/imagenet \
    --out_dir /path/to/output_grabcut \
    --ann_dir /path/to/annotations \
    --segmentor grabcut
```

## Output Structure

```
output/
├── original/train/0~8/
├── only_bg_b/train/0~8/
├── only_bg_t/train/0~8/
├── only_fg/train/0~8/
├── no_fg/train/0~8/
├── fg_mask/train/0~8/     # Binary masks (.npy)
├── mixed_same/train/0~8/
├── mixed_rand/train/0~8/
├── mixed_next/train/0~8/
└── in9l/train/0~8/
```

## Pipeline Details

### Filtering (identical to the original paper)

All filtering thresholds match the original paper (Appendix A) and [the author's code](https://github.com/rabbit-abacus/roleofimagebackgrounds):

- Single bounding box only
- `FRACTION = 0.9` — skip if bbox covers > 90% of cropped image
- `MIN_MASK_AMOUNT = 0.1` — skip if FG mask < 10% of bbox area
- `FRACTION_2 = 0.5` — skip if bbox in crop < 50% of bbox after resize-only
- Transforms: `Resize(256, NEAREST)` + `CenterCrop(256)` (train) / `CenterCrop(224)` (val)

### Two-Phase Processing

1. **Phase 1**: For each of the 9 superclasses, iterate all synsets → filter images → run SAM 3 segmentation → save `original`, `only_bg_b`, `only_bg_t`, `only_fg`, `no_fg`, `fg_mask.npy`
2. **Phase 2**: After all superclasses complete Phase 1, generate `mixed_same`, `mixed_rand`, `mixed_next` by compositing FG onto BG from filtered images (matching the original code's behavior of selecting BG from already-accepted images)

### Differences from Original Code

| Aspect | Original | This Implementation |
|--------|----------|-------------------|
| Segmentation | GrabCut (5 iterations) | SAM 3 bbox prompt |
| FG mask in Phase 2 | Re-run GrabCut (JPEG artifact avoidance) | Load from `.npy` (lossless) |
| BG selection scope | Same synset (per-synset processing artifact) | Same superclass (matches paper definition) |
| Mixed-Rand/Next | Not implemented | Implemented |
| IN-9L | Not implemented | Implemented |

## Evaluation

Use [MadryLab's evaluation code](https://github.com/MadryLab/backgrounds_challenge):

```bash
python in9_eval.py --arch resnet50 --data-path /path/to/output --eval-dataset mixed_same
python in9_eval.py --arch resnet50 --data-path /path/to/output --eval-dataset mixed_rand
# BG-Gap = mixed_same_acc - mixed_rand_acc
```

## References

- Xiao, K., Engstrom, L., Ilyas, A., & Madry, A. (2021). *Noise or Signal: The Role of Image Backgrounds in Object Recognition*. ICLR 2021. [arXiv:2006.09994](https://arxiv.org/abs/2006.09994)
- Ravi, N. et al. (2026). *SAM 3: Segment Anything with Concepts*. Meta AI. [GitHub](https://github.com/facebookresearch/sam3)
- [MadryLab/backgrounds_challenge](https://github.com/MadryLab/backgrounds_challenge) — Official evaluation code
- [rabbit-abacus/roleofimagebackgrounds](https://github.com/rabbit-abacus/roleofimagebackgrounds) — Original dataset generation code

## License

This code is for research purposes. ImageNet images require a separate license from [image-net.org](https://image-net.org/).
