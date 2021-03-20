# LAMP

This is the official implementation of the paper: ["Layerwise Sparsity for Magnitude-based Pruning"](https://openreview.net/forum?id=H6ATjJ0TKdf), ICLR 2021.

- The key file is the `tools/pruners.py`, where we implement various layerwise sparsity determination methods for the magnitude-based pruning.  
- Run `iterate.py` to run simulations.
- This codebased only contains CIFAR-10 experiments in the paper. To add more experiments, you may want to add models to `tools/models/`, datasets to `tools/datasets/` and `tools/dataloaders.py`, hyperparameter setups to `tools/modelloaders.py`.

Enjoy!  
Best,  
Authors.
