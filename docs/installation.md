# Installation

Clone the repository with Git LFS enabled so the released generation checkpoints are downloaded correctly:

```bash
git lfs install
git clone https://github.com/xiaoshaziydy/mrose.git
cd mrose
git lfs pull
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate mROSE
```

Or install the core pip dependencies:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

Check that the package imports correctly:

```bash
python scripts/quick_import_check.py
```

## Checkpoints

The released generation checkpoints are tracked with Git LFS:

```text
generation/
├── 5utr/Model.pth
├── cds/Model.pth
└── 3utr/Model.pth
```

Verify checkpoint integrity:

```bash
shasum -a 256 -c MODEL_CHECKSUMS.sha256
```

If Git LFS is not installed, these paths may contain small pointer files instead of the real checkpoints.
