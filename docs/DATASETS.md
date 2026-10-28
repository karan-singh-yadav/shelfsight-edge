# Datasets

## Dataset A: SKU-110K (benchmark)

Source: https://github.com/eg4000/SKU110K_CVPR19 (CVPR 2019, "Precise Detection in Densely Packed
Scenes").

Facts taken from the repository README (checked 2026-10-24):
- The dataset is provided **for exclusive use by the recipient, solely for academic and
  non-commercial purposes.** Do not redistribute it. Read the terms yourself before use, and
  think carefully before publishing anything derived from it (including trained weights) in a
  portfolio repo: this project is not legal advice.
- Annotations are CSV, compatible with the keras-retinanet CSV parser.
- It is a dense-shelf **single-class** detection benchmark: boxes mark product instances, not
  brand/SKU identity. This project must never claim SKU recognition from it.
- Official splits: train / val / test CSVs.

Not stated in the README and therefore **unverified until you download the archive**:
- Exact CSV column layout. This repo's loader assumes the 8 columns
  `image_name,x1,y1,x2,y2,class,image_width,image_height`, headerless or with a header. If the
  real files differ, `load_annotations` fails loudly with the column count rather than guessing.
- Archive size and unpacked disk usage. Check free disk space after getting the file size from
  the download page, before extracting. Do not extract blindly.

### Obtaining it
Use the download links in the SKU-110K README yourself, extract into `data/sku110k/`
(git-ignored), then:

```bash
python scripts/prepare_dataset.py validate --csv data/sku110k/annotations_val.csv --images data/sku110k/images
python scripts/prepare_dataset.py stats    --csv data/sku110k/annotations_val.csv --out reports/sku110k_val_stats.json
python scripts/prepare_dataset.py visualize --csv data/sku110k/annotations_val.csv --images data/sku110k/images --out reports/figures/gt_samples
# small dev subset (50 random images per split) in YOLO layout:
python scripts/prepare_dataset.py convert --images data/sku110k/images --out data/yolo_sku_dev \
    --train-csv data/sku110k/annotations_train.csv --val-csv data/sku110k/annotations_val.csv --limit 50
```
Exact paths depend on how the archive unpacks; adjust after inspecting it.

## Dataset B: custom retail dataset (planned)

1. Collect images you have the right to use; record `image,source,license` in a CSV.
2. Annotate in CVAT, export in YOLO format.
3. `prepare_dataset.py split --images ... --sources sources.csv --out manifest.csv`
   groups exact and near-duplicate images (dHash, Hamming distance <= 4 by default) and assigns
   whole groups to train/val/test.
4. `prepare_dataset.py verify-splits --manifest manifest.csv --images ...` must report no
   leakage before training. `unknown` provenance is flagged as a warning.

## Synthetic data
`make-synthetic` generates rectangles on a gradient, only to test tooling without licensed data.
Never report model metrics measured on it as retail performance.
