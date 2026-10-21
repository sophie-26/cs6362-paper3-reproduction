# CS 6362 Assignment 2 — Paper 3 Reproduction

PyTorch reproduction of Figure 7A and 7B from *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework* (Higgins et al., ICLR 2017).

## Overview

This project investigates whether increasing the KL divergence penalty in a variational autoencoder encourages disentangled latent representations.

The experiments reproduce:

- **Figure 7A:** Latent responses to known generative factors in the dSprites dataset.
- **Figure 7B:** Decoded latent traversals showing how generated images change when individual latent dimensions are varied.

Each experiment compares a standard VAE (β = 1) with a β-VAE (β = 4) to examine disentanglement and reconstruction quality.

## Implementation

The experiments use PyTorch and the dSprites dataset, with:

- Full dSprites dataset containing 737,280 binary 64 × 64 images
- Fully connected encoder and decoder architectures
- 10-dimensional Gaussian latent representation
- Bernoulli reconstruction loss and β-weighted KL divergence
- β values of 1 and 4
- Adagrad optimizer with learning rate 0.01
- Batch size of 128
- 10 training epochs per model
- One training run per β configuration
- Latent responses evaluated across known generative factors
- Nine traversal values over [-3, 3] for each latent dimension

## How to Reproduce

1. Open the reproduction `.ipynb` notebook in Google Colab.
2. Select a GPU runtime if available.
3. Download the official dSprites dataset from https://github.com/google-deepmind/dsprites.
4. Upload `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` to `/content/`.
5. Run the notebook cells in order.
6. The notebook trains both models, evaluates latent responses, generates decoded traversals, and saves the reproduced figures.

Required Python packages include `torch`, `numpy`, and `matplotlib`.

Training may take substantial time depending on available hardware. Saved checkpoints can be reused when available, but are not required to reproduce the experiments from scratch.

## Limitations

This is a computationally reduced reproduction rather than an exact replication of the original experiments. Both models are trained for 10 epochs, with one training run per β configuration, limiting conclusions about the consistency of disentanglement across random seeds.

Latent responses are estimated using a finite sample of images at each generative factor level. The traversal procedure varies individual latent coordinates while fixing the remaining coordinates to the inferred posterior means of reference images.

The β = 4 model achieved higher reconstruction loss and lower KL divergence than the β = 1 model, consistent with the expected tradeoff from stronger regularization. However, these measurements alone do not establish superior disentanglement. All figures are generated from actual trained models rather than hardcoded results.

## Reference

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017.

https://openreview.net/forum?id=Sy2fzU9gl
