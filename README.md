# Domain-Aware Multi-View Contrastive Representation Learning for Protein Subcellular Localization

## Requirements

This project relies on specific Python packages to ensure its proper functioning. The required packages and their versions are listed in the `requirements.txt` file.

## Benchmark Datasets

### Bai’s dataset
download at https://github.com/ghli16/DeepMTC

### Thumuluri’s dataset 
download at https://doi.org/10.6084/m9.figshare.25324903.

## input feature preparation

### interproscan
doamin information can be extracted at https://www.ebi.ac.uk/interpro/about/interproscan/
Proteins without InterPro records were excluded from the analysis.

### esm
Install [ESM-2](Lin et al. 2023)and[ESMFold](https://github.com/facebookresearch/esm)

### dssp
Add permission to execute for DSSP by "./src/mkdssp"

## Training
python train.py

## checkpoint

Thumuluri’s dataset_ckp: "./src/ckp/dataset0_ckp.pt"   
Bai’s dataset_ckp: "./src/ckp/dataset1_ckp.pt"

## demo running

For now, you can use the demo_dataset to test our algorithm. The demo_dataset currently includes 10 proteins, which were randomly selected.

you can run demo.ipynb or demo.py to test our code


