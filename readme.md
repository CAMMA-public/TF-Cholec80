<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="visuals/camma_logo_tr.png" width="20%">
</a>
</div>

# TF-Cholec80

Public release of the *Cholec80* dataset for Tensorflow.
-------------------
**EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos (IEEE Transactions on Medical Imaging, 2016)**
_Twinanda, Andru and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas_

*Developer & maintainer: Tong Yu*

[![arXiv](https://img.shields.io/badge/arxiv-1602.03012-red)](https://arxiv.org/abs/1602.03012)


## Description
The Cholec80 dataset contains 80 videos of cholecystectomy surgeries performed by 13 surgeons. Frames extracted from the videos at 1 fps are provided, each frame labeled with the phase and tool presence annotations. The phases have been defined by a senior surgeon in our partner hospital. Since the tools are sometimes hardly visible in the images and therefore difficult to perceive, we define a tool as present in an image if at least half of the tool tip is visible.

Surgical phases:
- Preparation
- CalotTriangleDissection
- ClippingCutting
- GallbladderDissection
- GallbladderRetraction
- CleaningCoagulation
- GallbladderPackaging

Surgical instruments __(instrument presence binary labels follow the order from this list)__:
- Grasper
- Bipolar
- Hook
- Scissors
- Clipper
- Irrigator
- SpecimenBag

## Usage
Download and extract the dataset to your preferred location. You may use the `--verify_checksum` flag to ensure the archive was downloaded without errors. If you wish to keep the archive after extraction please add the `--keep_archive` flag.
 
```bash
python prepare.py --data_rootdir YOUR_LOCATION
```

Create a Tensorflow-compatible `Dataset` using `make_cholec80`:

```python
from tf_dataset import make_cholec80
ds = make_cholec80(12)
```

Only the minibatch size is required.

By default `make_cholec80` will look for config.json in the current workspace. If using a different configuration file, please provide its path as the `config_path` argument.

Videos are indexed from 0 to 79. Limiting the dataset to specific videos can be done by supplying a list of indices as the `video_ids` argument. By default all 80 videos will be included.

Three modes of operation are supported based on the `mode` argument:
- `FRAME` (default): video shuffling, frame interleaving, minibatch shuffling
- `VIDEO`: video shuffling
- `INFER`: no shuffling

## Requirements

- Python 3
- Tensorflow 2
- Matplotlib (needed only for demo notebook)

Developer configuration info:
- Ubuntu 20.04
- CUDA 10.1
- NVIDIA GTX1080Ti GPU

## Citation
```bibtex
@article{article,
author = {Twinanda, Andru and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas},
year = {2016},
month = {02},
title = {EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos},
volume = {36},
journal = {IEEE Transactions on Medical Imaging},
doi = {10.1109/TMI.2016.2593957}
}
```

## License
This code may be used for **non-commercial scientific research purposes** as defined by [Creative Commons 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
