# Semantic Segmentation Based Optimal Drone Landing Zone

Pixel-level semantic segmentation of aerial and drone imagery, used to find safe areas for a drone to land.
The model labels every pixel of a top-down image (for example water, land, road, building, vegetation), and
the resulting map can be filtered for open, flat, hazard-free regions large enough for the drone's footprint.

Model: **U-Net with an ImageNet-pretrained ResNet50 encoder**, built in TensorFlow / Keras.

---

## How it works

```
Aerial image ──▶ U-Net (ResNet50 encoder + decoder) ──▶ per-pixel class map ──▶ safe-landing-zone selection
```

### Architecture

```
Input (H × W × 3)
 │ ENCODER: ResNet50, ImageNet weights          DECODER
 ├─ conv1_relu        (H/2)  ─────── skip ─────────▶ block 4, 64 filters
 ├─ conv2_block3_out  (H/4)  ─────── skip ──────▶ block 3, 128 filters
 ├─ conv3_block4_out  (H/8)  ─────── skip ───▶ block 2, 256 filters
 ├─ conv4_block6_out  (H/16) ─────── skip ▶ block 1, 512 filters
 └─ conv5_block3_out  (H/32, bottleneck) ──┘
                                     UpSampling ×2 → 1×1 conv + softmax → H × W × n_classes
```

- **Encoder:** ResNet50 pretrained on ImageNet (`include_top=False`). Transfer learning means it already
  recognises edges and textures, so it needs less aerial data to train.
- **Skip connections:** four intermediate ResNet50 feature maps are passed to the decoder so that fine
  spatial detail (region boundaries, small objects) is recovered.
- **Decoder block:** `Conv2DTranspose` (learned ×2 upsampling) → concatenate with the skip features →
  two × (3×3 Conv → BatchNorm → ReLU).
- **Output:** a 1×1 convolution with softmax gives a probability for each class at every pixel; the
  predicted class is the argmax.

### Loss

**Categorical cross-entropy + Dice loss.** Cross-entropy gives stable per-pixel learning; Dice loss
optimises region overlap directly and reduces the effect of class imbalance (small classes such as
water or roads covering few pixels).

### Training setup (`src/semantic_segmentation.py`)

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-4 |
| Batch size | 16 |
| Epochs | up to 100 |
| Learning-rate schedule | ReduceLROnPlateau (×0.5 after 5 epochs without improvement, min 1e-6) |
| Early stopping | patience 10, restores best weights |
| Checkpointing | best model saved to `model.h5` |
| Metrics | pixel accuracy, Dice coefficient, Mean IoU |
| Split | 80% train / 20% test, 10% of train held out for validation (seed 42) |

### Data augmentation (albumentations, training only)

Top-down imagery has no fixed "up", so geometric augmentations are safe and effective:

- Horizontal / vertical flip, random 90° rotation, transpose
- One of: elastic transform, grid distortion, optical distortion
- Random brightness / contrast and random gamma (lighting and time-of-day variation)

Geometric augmentations are applied identically to the image and its mask. Masks are always resized
with **nearest-neighbour** interpolation so that no invalid in-between class labels are created.

---

## Datasets

Training data combines several public aerial and drone segmentation datasets:

- **Semantic Drone Dataset**: [Kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)
- **Semantic Segmentation of Aerial Imagery**: [Kaggle](https://www.kaggle.com/datasets/bulentsiyah/semantic-segmentation-of-aerial-imagery)
- **ISPRS Potsdam and Vaihingen** (urban segmentation): [ISPRS](https://www.isprs.org/)
- **OpenEarthMap** (global land cover): [OpenEarthMap](https://open-earth-map.org)
- **Swiss Drone and Okutama Drone datasets**: [Okutama Segmentation](https://okutama-segmentation.org)

Datasets are not included in this repository. Place them under `data/` using the folder names in
`src/create_metadata.py`.

---

## Two model configurations

The repository contains two configurations. Make sure the script settings match the weights you load.

| Script | Purpose | Classes | Input size |
|---|---|---|---|
| `src/semantic_segmentation.py` | Full training pipeline (augmentation, callbacks, evaluation) | 6 | 256 × 256 |
| `src/train.py` | Loads saved weights from `models/model.h5` and visualises predictions | 12 | 384 × 384 |

**6-class configuration** (colour-coded masks, see `src/preprocess_masks.py`):

| ID | Class | Mask colour (RGB) |
|---|---|---|
| 0 | Water | (80, 227, 194) |
| 1 | Land | (245, 166, 35) |
| 2 | Road | (222, 89, 127) |
| 3 | Building | (208, 2, 27) |
| 4 | Vegetation | (65, 117, 5) |
| 5 | Unlabeled | (155, 155, 155) |

**12-class configuration:** raw mask values found across the combined datasets
(`src/create_mapping.py`) are remapped to class indices 0–11 in `src/train.py`.

---

## Project structure

```
src/
├── create_metadata.py     # builds data/all_metadata.csv (image/mask pairs + train/test split)
├── preprocess_masks.py    # converts colour masks to class-index masks (6-class scheme)
├── create_mapping.py      # lists unique mask values across datasets → class-index mapping
├── inspect_masks.py       # prints unique values in sample masks (sanity check)
├── semantic_segmentation.py  # training pipeline (6 classes, 256×256)
└── train.py               # load trained weights and plot predictions (12 classes, 384×384)
```

`data/`, `models/`, `notebooks/` and `.venv/` are git-ignored.

---

## Setup

```bash
git clone https://github.com/GVishwasReddy/Semantic_Segmentation_Based_Optimal_Drone_Landing_Zone-.git
cd Semantic_Segmentation_Based_Optimal_Drone_Landing_Zone-

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare the data

```bash
python src/create_metadata.py     # creates data/all_metadata.csv
python src/preprocess_masks.py    # colour masks → class-index masks (6-class scheme)
```

### 2. Train

```bash
python src/semantic_segmentation.py
```

Trains the U-Net, saves the best weights to `model.h5`, plots IoU and loss curves, evaluates on the
test split and shows sample predictions (image / ground truth / prediction).

### 3. Visualise predictions with saved weights

Place trained 12-class weights at `models/model.h5`, then run:

```bash
python src/train.py
```

This loads the model and plots predictions on random samples from the test set.

---

## From segmentation to a landing zone

The model outputs a class map; choosing where to land is a post-processing step:

1. Mark safe classes (open land, low vegetation) as 1 and hazards (water, buildings, trees, unlabeled) as 0.
2. Erode the safe mask by a safety margin so the drone keeps clear of hazard edges.
3. Keep connected safe regions larger than the drone's footprint at the current altitude.
4. Use a distance transform to pick the point farthest from any hazard as the landing target.
5. Combine with person detection and re-check during descent before landing.

---

## Known limitations and next steps

- **Unified labels across datasets:** the source datasets use different class definitions and colour
  codes. The training script currently maps any mask value outside the 6 classes to Unlabeled (and
  prints a warning); a per-dataset mapping onto one shared class set would make the combined data consistent.
- **On-board deployment:** export to ONNX and quantize to INT8 for edge accelerators, or swap in a
  lighter encoder such as MobileNet.
- **Terrain slope:** segmentation shows what the ground is, not whether it is flat; adding depth or
  elevation data would let the model reject sloped areas.
