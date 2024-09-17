# Self-supervised pretraining for cardiovascular magnetic resonance cine segmentation

PyTorch based code used for [Self-supervised pretraining for cardiovascular magnetic resonance cine segmentation](), comparing 4 self-supervised pretraining (SSP) methods and an nnU-Net based baseline for 2D cardiovascular magnetic resonance cine segmentation. The 4 methods are: SimCLR, positional contrastive learning (PCL), DINO, and masked image modeling (MIM), which are all visualized below:
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
#### Data setup
Public datasets must first be reformatted so the qcardia-data pipeline can use the data. For the supported public datasets, this can be achieved by:
1. Downloading the public data.
2. Saving the data with the expected folder hierarchy.
3. Updating the configs to point to your local data folder.
4. Running the relevant data setup functions.


The public [M&Ms](https://www.ub.edu/mnms/) and [M&Ms-2](https://www.ub.edu/mnms-2/) challenge datasets can be downloaded from their respective websites. After downloading, our automatic `qcardia-data` reformatting can be used when you've saved the original datasets in their expected folder heirarchy:
```
data
└── original_data
    ├── MnM
    │   ├── dataset
    │   │   ├── A0S9V9
    │   │   │   ├── A0S9V9_sa_gt.nii.gz
    │   │   │   └── A0S9V9_sa.nii.gz
    │   │   └── ...
    │   └── dataset_information.csv
    ├── MnM2
    │   ├── dataset
    │   │   ├── 001
    │   │   │   ├── 001_SA_CINE.nii.gz
    │   │   │   ├── 001_SA_ED_gt.nii.gz
    │   │   │   ├── 001_SA_ED.nii.gz
    │   │   │   ├── 001_SA_ES_gt.nii.gz
    │   │   │   ├── 001_SA_ES.nii.gz
    │   │   │   └── ... (001_LA... -> long axis data unused for now)
    │   │   └── ...
    │   └── dataset_information.csv
    └── ...
```

The `qcardia-data` cine setup function can be then be used, requiring only a path to your local data folder:

```python
from qcardia_data.setup import setup_cine
from pathlib import Path

data_path = Path("path/to/your/data_folder")
setup_cine(data_path)
```
The data setup functions reformat the relevant available original datasets, and can generate default test data splits. A copy of the test split file, as well as a full data split file, are included in this repository. By default, the included configs use the full split file, which should be saved in the `subject_splits` subfolder in your data folder. Also make sure to update your data path in any config files.

More details can be found in the `qcardia-data` [demo](https://github.com/q-cardIA/qcardia-data/blob/main/demo/demo.ipynb) and [README](https://github.com/q-cardIA/qcardia-data).

#### Training
Afterwards, the training scripts should work out of the box. The [train_unet](code/train_unet.py) script trains a U-Net based on the [baseline-config](configs/baseline-config.yaml) configuration file. Similarly, there are training scripts for SimCLR, PCL, DINO, and MIM pretraining, using their respective config files. ***Note that the data path should be updated in each config file.***
