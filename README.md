# Self-supervised pretraining for cardiovascular magnetic resonance cine segmentation

PyTorch based code used for [Self-supervised pretraining for cardiovascular magnetic resonance cine segmentation](), comparing 4 self-supervised pretraining (SSP) methods and an nnU-Net based baseline for cardiovascular magnetic resonance cine segmentation. The 4 methods are: SimCLR, positional contrastive learning (PCL), DINO, and masked image modeling (MIM), which are all visualized below:
![SSP methods visualizations](docs/ssp-methods-visualization.png?raw=true)

The code uses our own [qcardia-data](https://github.com/q-cardIA/qcardia-data) and [qcardia-models](https://github.com/q-cardIA/qcardia-models) modules.

## Installation
### Environment and PyTorch
It is recommended to make a new environment (tested for Python 3.11.9) and first installing PyTorch and checking GPU availability. Installation instructions for the latest stable PyTorch version can be found in [their "get started" guide](https://pytorch.org/get-started/locally/). Alternatively, they include instructions for [previous stable versions](https://pytorch.org/get-started/previous-versions/). Installing the PyTorch version the package was tested for (PyTorch 2.3.1) might limit warnings or unexpected behaviours.

### Install requirements
The `requirements.txt` file lists the required packages, mostly by installing our own `qcardia-data` and `qcardia-models` modules. It can be installed using pip:
```
pip install -r /path/to/requirements.txt
```

Alternatively, local (editable) copies of [qcardia-data](https://github.com/q-cardIA/qcardia-data) and [qcardia-models](https://github.com/q-cardIA/qcardia-models) can be used. Installation instructions for local (editable) copies can be found on the respective GitHub pages of the qcardia packages.

## Getting started
To rerun our experiments, the [M&Ms](https://www.ub.edu/mnms/) and [M&Ms-2](https://www.ub.edu/mnms-2/) challenge datasets should be downloaded. The datasets should be reformatted before they work with `qcardia-data`, instructions for which can be found in the [qcardia-data demo](https://github.com/q-cardIA/qcardia-data/blob/main/demo/demo.ipynb).

Afterwards, the training scripts should work out of the box. The [train_unet](code/train_unet.py) script trains a U-Net based on the [baseline-config](configs/baseline-config.yaml) configuration file. Similarly, there are training scripts for SimCLR, PCL, DINO, and MIM pretraining, using their respective config files.
