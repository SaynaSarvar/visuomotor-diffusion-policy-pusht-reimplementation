<div align="center">

# 🤖 Diffusion Policy for Push-T task
### A from-scratch implementation: state-based first, then image-based

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2303.04137-b31b1b)](https://arxiv.org/abs/2303.04137)

<img src="assets/method_overview.png" width="900" alt="Training and sampling of a diffusion policy">

</div>

A learning project that reimplements the **Push-T** experiments of
[*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*](https://diffusion-policy.cs.columbia.edu/)
(Chi et al.) in plain PyTorch, built step by step in Jupyter notebooks. The plan has two phases: the
**state-based** variant first (in progress), then the **image-based** variant, which reuses the same diffusion
core and changes only the observation encoder.

> 🚧 **Status: work in progress.** The model is implemented and the long training run is under way.
> Closed-loop evaluation in the Push-T simulator and the ablations are **not done yet**, so this repository
> reports **no success-rate or coverage numbers**.

## 📚 Contents
1. [The task](#-the-task-push-t)
2. [The idea in plain words](#-the-idea-in-plain-words)
3. [Progress](#-progress)
4. [What is implemented](#-what-is-implemented)
5. [Sanity checks](#-sanity-checks-i-ran)
6. [Setup and differences from the paper](#%EF%B8%8F-setup-and-differences-from-the-paper)
7. [Results so far](#-results-so-far)
8. [Roadmap](#%EF%B8%8F-roadmap)
9. [Running it](#%EF%B8%8F-running-it)
10. [Reference](#-reference)

## 🎯 The task: Push-T
A circular pusher (blue) must push a T-shaped block (gray) onto a fixed target T (green/red in the paper's
renderings). Only point contact is available, so the block's motion is under-actuated and the task needs
precise, contact-rich planning. Success is measured by how much of the target area the block covers.

- **Data:** the human demonstrations released with the paper: **206 episodes, 25,650 steps**.
- **State-based observation (20 numbers):** 9 keypoints on the block (x, y each) + the pusher's position.
- **Action (2 numbers):** the pusher's **target position** (position control).

<p align="center">
  <img src="assets/data_keypoints.png" width="560" alt="Rendered frame next to the 9 block keypoints and the pusher position"><br>
  <em>One time step: the rendered image (left) and the state-based observation (right).</em>
</p>
<p align="center">
  <img src="assets/data_episode_paths.png" width="800" alt="Pusher paths in three demonstrations"><br>
  <em>The pusher's path in three human demonstrations (green = start, red = end).</em>
</p>

## 💡 The idea in plain words
A **policy** maps what the robot observes to what it should do. Instead of regressing a single action, a
**diffusion policy** generates a whole **sequence of 16 future actions** by *denoising*:

- **Training:** take a real human action block, add random noise at a random level, and train a network to
  predict the noise that was added (paper Eq. 5). The observation is given to the network as a guide.
- **Sampling:** start from pure noise and apply the network 100 times, each time removing a little noise,
  until a clean action plan remains. Different random starts can end in different valid plans, which is how a
  diffusion policy handles tasks where several strategies work (for example, going around the block on the
  left or on the right).
- **Receding horizon:** execute only the first part of the plan (8 of the 16 actions), then re-plan.

<details>
<summary><b>Why not just regress the action?</b> (click to expand)</summary>

If half of the demonstrations go left and half go right, a regression network trained with MSE learns the
*average* (straight ahead), which is wrong in every demonstration. A generative model represents both options
and commits to one. The paper also argues for stable training: learning the score (the slope of the log
probability) avoids the intractable normalisation constant that makes energy-based policies unstable.
</details>

## ✅ Progress

| Part | Status |
|---|---|
| Data inspection, normalisation to [-1, 1] (paper App. A.1), training windows | ✅ done |
| Squared-cosine noise schedule, forward noising, one-step recovery check | ✅ done |
| Sinusoidal step embedding and FiLM-conditioned 1D U-Net (paper Fig. 2b) | ✅ done |
| Training loop (Eq. 5), EMA, cosine LR with warmup, resumable checkpoints | ✅ done, long run in progress |
| DDPM sampler (100 steps, x0 clipping) | ✅ runs end to end |
| Receding-horizon rollouts in the Push-T environment and target-area-coverage evaluation | ⏳ next |
| Ablations (action horizon, EMA) | ⏳ planned |
| **Phase 2:** image-based Push-T (ResNet-18, spatial softmax, GroupNorm, random-crop augmentation) | 📅 planned |

## 🧩 What is implemented

### 1. Data and training windows
Each training example is a window around a time step: the **last 2 observations** (2 × 20) as the guide and
the **next 16 actions** (16 × 2) as the clean target. Windows near the start or end of an episode are padded by
repeating the first or last row, so the policy also sees the very first and last moments.

The actions are target *positions*, not velocities: the pusher always stays close to the point it is chasing.

<p align="center">
  <img src="assets/action_vs_position.png" width="380" alt="Action path next to the pusher path"><br>
  <em>Position control: the action (dashed) runs just ahead of the pusher's position (solid).</em>
</p>

### 2. Noise schedule and forward noising
The squared-cosine schedule (from iDDPM) over 100 steps: the signal fades slowly at first and fastest in the
middle. `alpha_bar` is the fraction of signal left at each level.

<p align="center">
  <img src="assets/noise_schedule.png" width="800" alt="beta, alpha-bar and signal/noise weights over the 100 diffusion steps">
</p>

Noising a real action block at increasing levels k: a smooth path dissolves into a cloud of random points.

<p align="center">
  <img src="assets/forward_noising.png" width="900" alt="One action block at noise levels k = 0 to 90">
</p>

### 3. The denoising network
A **conditional 1D U-Net** (paper Fig. 2b). The condition (a sinusoidal embedding of the noise level, passed
through a small MLP, joined with the flattened observation) enters every block through **FiLM**, which turns it
into a per-channel scale and shift: `scale * features + shift`.

<p align="center">
  <img src="assets/unet_architecture.png" width="900" alt="Architecture of the conditional 1D U-Net with FiLM conditioning and skip connections">
</p>

The step embedding gives every noise level a distinct fingerprint, with nearby levels similar to each other:

<p align="center">
  <img src="assets/timestep_embedding.png" width="760" alt="Sinusoidal embedding of the diffusion step and its similarity to k = 50">
</p>

### 4. Training
Noise-prediction MSE (Eq. 5), AdamW, cosine learning-rate schedule with 500 warmup steps, an EMA copy of the
weights for evaluation, and checkpoints every 50 epochs that allow resuming after an interruption.

### 5. Sampler check
The 100-step sampler runs end to end. The figure below is **only a pipeline check after 200 training steps**
(very early, before the long run): the 8 samples for one observation are rough, but they already lie in the
region of the human actions (red). It will be replaced by a figure from the trained model.

<p align="center">
  <img src="assets/sampler_check_200steps.png" width="380" alt="Eight sampled action blocks versus the human action block after 200 training steps">
</p>

## 🔍 Sanity checks I ran

| Check | Result |
|---|---|
| Dataset size | 206 episodes, 25,650 steps, mean length 124.5 (min 49, max 246) |
| Actions are positions | median distance between the action and the pusher's position is 19.3 on a 0 to 512 field |
| Normalisation | min = -1, max = +1 exactly; round-trip error at most 3e-5 |
| Training windows | shapes (2, 20) and (16, 2); padding at episode start and end behaves as intended |
| Recovering the clean action from the *true* noise | error around 1e-7 for k = 10, 50, 90 |
| Effect of a slightly wrong noise guess (error scale 0.1) | recovery error grows with k: 0.042 (k = 10), 0.348 (k = 50), 1.41 (k = 90) |
| Network | 65,783,298 parameters; output shape (batch, 16, 2); output depends on both the observation and k |
| Training speed | about 0.3 s/step on a Tesla T4 (about 33 s per epoch) |

## ⚙️ Setup and differences from the paper
- **Windows:** T_o = 2 observation steps in, T_p = 16 action steps out; T_a = 8 will be used when executing plans.
- **Diffusion:** 100 training steps, squared-cosine schedule, noise (epsilon) prediction, MSE loss.
- **Training:** batch 256, AdamW (lr 1e-4, weight decay 1e-6), cosine schedule with 500 warmup steps,
  4500-epoch schedule, EMA of the weights.

Where this differs from the paper:
- The network is a **smaller variant (65.8M parameters, channels 256/512/1024)**; the paper's Table 7 lists
  about 256M for Push-T.
- Kernel size 5, 8 GroupNorm groups and the 256-d step embedding follow my recollection of common practice and
  are not stated in the paper's tables.
- The noise schedule comes from iDDPM, but iDDPM's learned variances are **not** implemented (fixed-variance
  DDPM sampling).
- AdamW and the EMA decay ramp (towards 0.9999) are my choices; the paper does not specify them.

## 📈 Results so far
Training loss (noise-prediction MSE averaged over random noise levels) fell from about 1.0 at the start to
about 0.005 by epoch 800. A low loss alone does not show that the policy works: rollout evaluation in the
simulator is the next step, and the numbers will be added here when they exist.

## 🗺️ Roadmap
- [x] Data, schedule, network, training loop, sampler
- [ ] Finish the 4500-epoch run
- [ ] Evaluate with receding-horizon control in the Push-T environment (target-area coverage; max and
      average of the last 10 checkpoints, as in the paper)
- [ ] Ablations: action horizon T_a, EMA on/off
- [ ] Image-based variant (ResNet-18 encoder) and a state-vs-image comparison

## ▶️ Running it
1. Download the Push-T data (`pusht.zip`) from the [project page](https://diffusion-policy.cs.columbia.edu/)
   and unzip it so that `data/pusht/pusht_cchi_v7_replay.zarr` exists.
2. `pip install -r requirements.txt`
3. Open `notebooks/pusht_diffusion_policy.ipynb` and run the cells in order. A GPU is strongly recommended.
   The first cell was written for Google Colab; adjust the data and checkpoint paths for your machine.

## 📖 Reference
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song.
*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* arXiv:2303.04137 (RSS 2023; IJRR 2024).
https://diffusion-policy.cs.columbia.edu/

<!-- Optional: if you want to state how the project was developed, add a line here, e.g.
"Developed as a self-study project, with AI assistance (Claude) for explanations and code review." -->
