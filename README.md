# Diffusion Policy on Push-T: a from-scratch PyTorch implementation

A learning project that reimplements the **Push-T** experiments of
*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Chi et al.) in plain PyTorch,
built step by step in Jupyter notebooks: data, noise schedule, denoising network, training loop and sampler.
The plan has two phases: the **state-based** variant first (in progress), then the **image-based** variant,
which reuses the same diffusion core and changes only the observation encoder.

> **Status: work in progress.** The model is implemented and training is running.
> Closed-loop evaluation in the Push-T simulator and the ablations are **not done yet**, so this repository
> reports **no success-rate / coverage numbers**.

## What a diffusion policy is (short version)
A policy maps what a robot observes to what it should do. Here the policy is a **conditional denoising
diffusion model over action sequences**: starting from random noise, a network repeatedly removes noise,
guided by the current observation, until a clean sequence of 16 future actions remains. This lets the
policy represent several valid ways of doing the task (multimodality) and commit to one of them.

## Progress

| Part | Status |
|---|---|
| Data inspection, normalization to [-1, 1] (paper App. A.1), training-window construction | done |
| Squared-cosine noise schedule, forward noising, one-step recovery check | done |
| Sinusoidal step embedding, FiLM-conditioned 1D U-Net (Fig. 2b) | done |
| Training loop (Eq. 5 noise-prediction loss), EMA, cosine LR with warmup, resumable checkpoints | done, long run in progress |
| DDPM sampler (100 steps, x0 clipping) | done, runs end to end |
| Receding-horizon rollouts in the Push-T environment (T_a = 8) and target-area-coverage evaluation | to do |
| Ablations (action horizon, EMA) | to do |
| **Phase 2:** image-based Push-T (ResNet-18 encoder, spatial softmax, GroupNorm, random-crop augmentation) | planned |

## Setup used (phase 1: state-based)
- **Data:** the Push-T demonstrations released with the paper: 206 episodes, 25,650 steps.
  Observation = 9 block keypoints (x, y) + circle position = 20 numbers per step; action = 2D target position
  (position control). The dataset is **not** included here; get it from the
  [project page](https://diffusion-policy.cs.columbia.edu/).
- **Windows:** 2 observation steps in, 16 action steps out (T_o = 2, T_p = 16), edges padded by repeating the
  first/last row.
- **Diffusion:** 100 training steps, squared-cosine schedule, noise (epsilon) prediction, MSE loss.
- **Training:** batch 256, AdamW (lr 1e-4, weight decay 1e-6), cosine schedule with 500 warmup steps,
  4500-epoch schedule, EMA of the weights.

### Where this differs from the paper
- The network is a **smaller variant (65.8M parameters, channels 256/512/1024)**; the paper's Table 7 lists
  about 256M for Push-T.
- Kernel size 5, 8 GroupNorm groups and the 256-d step embedding follow my recollection of common practice
  and are not stated in the paper's tables.
- The noise schedule comes from iDDPM, but iDDPM's learned variances are **not** implemented
  (fixed-variance DDPM sampling).
- AdamW and the EMA decay ramp (towards 0.9999) are my choices; the paper does not specify them.

## Results so far
Training loss (noise-prediction MSE, averaged over random noise levels) fell from about 1.0 at the start to
about 0.005 by epoch 800. A low loss alone does not show the policy works; rollout evaluation is the next step.

## Repository layout
```
notebooks/pusht_diffusion_policy.ipynb   # data -> schedule -> network -> training -> sampler
requirements.txt
```
Checkpoints and data are excluded via `.gitignore`.

## Running it
1. Download the Push-T data (`pusht.zip`) from the project page and unzip it so that
   `data/pusht/pusht_cchi_v7_replay.zarr` exists.
2. `pip install -r requirements.txt`
3. Open the notebook and run the cells in order. A GPU is strongly recommended
   (about 0.3 s/step on a Tesla T4).
   The first cell was written for Google Colab; adjust the data and checkpoint paths for your machine.

## Reference
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song.
*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* arXiv:2303.04137 (RSS 2023; IJRR 2024).
https://diffusion-policy.cs.columbia.edu/

<!-- Optional: if you want to state how the project was developed, add a line here, e.g.
"Developed as a self-study project, with AI assistance (Claude) for explanations and code review." -->
