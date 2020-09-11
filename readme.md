<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="visuals/camma_logo_tr.png" width="20%">
</a>
</div>

# TF-Cholec80

Library packaging the *Cholec80* dataset for easy handling with Tensorflow.

-------------------

**EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos (IEEE Transactions on Medical Imaging, 2016)**

_Twinanda, Andru and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas_

[![arXiv](https://img.shields.io/badge/arxiv-1602.03012-red)](https://arxiv.org/abs/1602.03012)

**Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition (IPCAI 2019)**

_Tong Yu, Didier Mutter, Jacques Marescaux, Nicolas Padoy_

[![arXiv](https://img.shields.io/badge/arxiv-1812.00033-red)](https://arxiv.org/abs/1812.00033)

*Developer & maintainer: Tong Yu*

-------------------

## Introduction
Similar to datasets available on `tfds`, *TF-Cholec80* offers a convenient interface for deploying the *Cholec80* dataset within deep learning applications built with Tensorflow. This independent release addresses the shortcomings associated with `tfds` to allow flexible handling of Cholec80 in a variety of scenarios such as single-frame tasks with CNNs, temporal tasks with CNN-RNNs or inference. *TF-Cholec80* is based on the code employed in the experiments performed for this [publication](https://arxiv.org/abs/1812.00033).

## Data overview
The *Cholec80* dataset, first introduced [here](#data), contains 80 videos of cholecystectomy surgeries performed by 13 surgeons. Frames extracted from the videos at 1 fps are provided, each frame labeled with the phase and tool presence annotations. The phases have been defined by a senior surgeon in our partner hospital. Since the tools are sometimes hardly visible in the images and therefore difficult to perceive, we define a tool as present in an image if at least half of the tooltip is visible.

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
** `prepare.py` must be run first before using TF-Cholec80 **. Download and extract the dataset to your preferred location (`--data_rootdir`). You may use the `--verify_checksum` flag to ensure the archive was downloaded without errors. If you wish to keep the archive after extraction please add the `--keep_archive` flag.
 
```bash
python prepare.py --data_rootdir YOUR_LOCATION
```

The path you provided will be written to `tf_cholec80/configs/config.json`.

Create a Tensorflow-compatible `Dataset` using `make_cholec80`:

```python
from tf_cholec80.dataset import make_cholec80
ds = make_cholec80(12)
```

Only the minibatch size is required.

By default `make_cholec80` will look for the `config.json` that was exported during installation (see next section), or in `tf_cholec80/configs` if installation was not performed. If using a different configuration file, please provide its path as the `config_path` argument.

Videos are indexed from 0 to 79. Limiting the dataset to specific videos can be done by supplying a list of indices as the `video_ids` argument. By default all 80 videos will be included.

Three modes of operation are supported based on the `mode` argument:
- `FRAME` (default): video shuffling, frame interleaving, minibatch shuffling
- `VIDEO`: video shuffling
- `INFER`: no shuffling

After running `prepare.py`, you will be able to use `cholec80_tf1_demo.ipynb` and `cholec80_tf2_demo.ipynb` which provide examples of how to use *TF-Cholec80* for both Tensorflow versions.

## Installation
Once the data is downloaded and extracted with `prepare.py`, only `dataset.py` and a `config.json` file are required to use *TF-Cholec80* in your own code. You may also install it afterwards for added convenience:

```bash
pip install .
```
This will export a copy of your current `config.json` that will be used by *TF-Cholec80* in the future.

## Surgical phase recognition
An example of a deep learning task using *TF-Cholec80* can be found [here](). This notebook performs surgical phase recognition with a CNN-biLSTM-CRF with *TF-Cholec80* as the input pipeline.

## Requirements
Storage:
- __Please ensure 166Gb of space are available before downloading__.
- After extraction and archive removal, the dataset will occupy __85.2 Gb__.

Libraries:
- Python 3
- Tensorflow 1.15 or 2.x
- tqdm
- Matplotlib (needed only for demo notebooks)

Developer configuration info:
- Ubuntu 20.04
- CUDA 10.1
- NVIDIA GTX1080Ti GPU

## Citation
### Data
The data itself is associated to this publication; please cite it if using *Cholec80* in any shape or form.
```bibtex
@article{endonet,
author = {Twinanda, Andru and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas},
year = {2016},
month = {02},
title = {EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos},
volume = {36},
journal = {IEEE Transactions on Medical Imaging},
doi = {10.1109/TMI.2016.2593957}
}
```
### Code
If *TF-Cholec80* is useful for your research, please cite the following publication:
```bibtex
@inproceedings{yu2019surgicalphase,
title = {Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition},
author = {Tong Yu, Didier Mutter, Jacques Marescaux, Nicolas Padoy},
booktitle = {International Conference on Information Processing in Computer-Assisted Interventions},
year = {2019}
}
```

## License
This code may be used for **non-commercial scientific research purposes** as defined by [Creative Commons 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
