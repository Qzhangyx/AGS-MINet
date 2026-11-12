# AGS-MINet

Adaptive Geometric Sampling Molecular Interaction Network for Protein Functional Site Design.

AGS-MINet jointly designs the amino acid sequences and three-dimensional structures of protein functional sites conditioned on a protein scaffold and a small-molecule ligand. It combines multiple geometric reference frames with adaptive geometric sampling to model intra-protein and protein–ligand interactions.

## Environment

The main dependencies are PyTorch, PyTorch Geometric, torch-scatter, fair-esm, RDKit, Biopython, NumPy, SciPy, LMDB, PyYAML, EasyDict, OmegaConf, tqdm, TensorBoard, WandB, and requests. Relaxation and docking-related utilities additionally require OpenMM, PDBFixer, and the Vina Python interface.

Use compatible versions of PyTorch, CUDA, and the PyTorch Geometric extensions. The ESM installation must provide the components imported by the structural adapter.

## Datasets

We use CrossDocked and Binding MOAD. Please refer to the following repositories for dataset downloads and preprocessing instructions:

- [FAIR](https://github.com/zaixizhang/FAIR#datasets)
- [PocketGen](https://github.com/zaixizhang/PocketGen#benchmark-datasets)

Update `dataset.path`, `dataset.split`, `model.orig_data_path`, and `model.pocket10_path` in the configuration files. Preprocessed data and split indices must match the format and sample ordering expected by this implementation.

## Training

Model and training hyperparameters can be adjusted in `configs/`.

Train on CrossDocked:

```bash
python train_model.py --config ./configs/train_model_cd.yml --device cuda:0 --logdir ./logs_cd
```

Train on Binding MOAD:

```bash
python train_model.py --config ./configs/train_model_bm.yml --device cuda:0 --logdir ./logs_bm
```

Run commands from the project root. Logs and checkpoints are saved under the specified log directory. Checkpoint-loading statements are currently commented out in the training script; setting `model.checkpoint` alone does not resume training.

The current training script uses the `train` subset for optimization and the `test` subset for validation. For independent test evaluation, use a separate validation subset during training.

## Checkpoint and Generation

Place a compatible checkpoint at `checkpoints/checkpoints.pt`, the path currently used by `generate_new.py`. The checkpoint must contain a `model` state dictionary.

```bash
python generate_new.py --config ./configs/train_model_bm.yml --device cuda:0 --target ./examples
```

Example inputs are provided in `examples/8UQF/`. The script currently selects `8UQF` and creates ten input copies. Generated structures are written through the model's structure-export routines; attention outputs are saved as `attention.pkl` in the target sample directory.

## Acknowledgements

We acknowledge [FAIR](https://github.com/zaixizhang/FAIR) and [PocketGen](https://github.com/zaixizhang/PocketGen) for their publicly available code and data resources.
