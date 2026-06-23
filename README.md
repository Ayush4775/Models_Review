# Models Review

A comparative study of three normative modelling and generative frameworks used in neuroimaging and statistical neuroscience research.

## Overview

This repository implements and compares three distinct approaches for modelling brain data distributions and detecting deviations from normative trajectories:

| Module | Approach | Use Case |
|--------|----------|----------|
| [`GAMLSS-python/`](GAMLSS-python/) | Generalized Additive Models for Location, Scale and Shape | Flexible statistical normative modelling with z-score outputs |
| [`Diffusion/`](Diffusion/) | Denoising Diffusion Probabilistic Models (DDPM) | Generative modelling of structured distributions |
| [`FAAE/`](FAAE/) | Focal Adversarial Autoencoder | Deep normative modelling with multiple autoencoder variants (AE, VAE, CVAE, AAE, FAAE) |

---

## Modules

### GAMLSS-python
A Python wrapper around R's [`gamlss`](https://www.gamlss.com/) package via `rpy2`. Supports flexible distribution families (SHASH, NO, BCT, etc.) with random effects for site harmonisation. Includes z-score calculation, diagnostic plots, and site transfer utilities.

**Requires:** R with `gamlss` package installed, plus `rpy2`.

### Diffusion
An educational implementation of DDPMs on 2D toy datasets. Covers the full forward/reverse diffusion process and includes interactive tutorials and Jupyter notebooks for understanding the theory.

### FAAE
A normative modelling framework for brain imaging feature vectors (e.g. cortical thickness). Implements six model variants with bootstrap-based training, testing, and group-level deviation analysis. Designed for ADNI-style data.

---

## Setup

Each module has its own dependencies. Install the root requirements first:

```bash
pip install -r requirements.txt
```

Then see the individual `README.md` in each subfolder for module-specific setup.

---

## Utilities

- **`gpr.py`** — Gaussian Process Regression utilities; generates synthetic `(age, score)` datasets under various distributions (normal, beta, GMM, long-tailed) for benchmarking normative models.
- **`pynm_test.py`** — Demo script for GAMLSS normative modelling.

---

## Repository Structure

```
Models_Review/
├── GAMLSS-python/      # Statistical normative modelling (R/Python)
├── Diffusion/          # DDPM generative model
├── FAAE/               # Adversarial autoencoder normative models
├── gpr.py              # GPR & synthetic data utilities
├── pynm_test.py        # GAMLSS demo
└── requirements.txt    # Root-level dependencies
```
