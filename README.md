# DrQ: Data regularized Q with DIAYN Skill Discovery

This is a PyTorch implementation of **DrQ** from

**Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels** by

[Denis Yarats*](https://cs.nyu.edu/~dy1042/), [Ilya Kostrikov*](https://github.com/ikostrikov), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php).

*Equal contribution. Author ordering determined by coin flip.

[[Paper]](https://arxiv.org/abs/2004.13649) [[Webpage]](https://sites.google.com/view/data-regularized-q)

This version has been extended to include **Diversity is All You Need (DIAYN)** for unsupervised skill discovery, as described in:

**Diversity is All You Need: Learning Skills without a Reward Function** by

[Benjamin Eysenbach](https://eysenbach.github.io/), [Abhishek Gupta](https://sites.google.com/view/abhigupta), [Julian Ibarz](https://julianibarz.com/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

[[Paper]](https://arxiv.org/abs/1802.06070)

Additionally, the project's configuration management has been updated from Hydra to `argparse`.

**Update**: The original DrQ authors released a newer version **DrQ-v2**, please check it out [here](https://github.com/facebookresearch/drqv2).

Original implementations in other frameworks: [jax/flax](https://github.com/ikostrikov/jax-rl).

## Features
*   Data-Regularized Q-learning (DrQ) for efficient learning from pixels.
*   Diversity is All You Need (DIAYN) for unsupervised skill learning.
*   Skill-conditioned policies and intrinsic reward mechanisms.
*   Configuration via `argparse` for straightforward command-line execution.
*   Skill-conditioned video logging to TensorBoard with intrinsic reward overlay.
*   Detailed CSV logging of extrinsic rewards, intrinsic rewards, and skills per episode.

## Citation
If you use this repo in your research, please consider citing the original DrQ paper:
```
@inproceedings{yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```
And the DIAYN paper if you use the skill discovery aspects:
```
@inproceedings{eysenbach2019diversity,
  title={Diversity is All You Need: Learning Skills without a Reward Function},
  author={Eysenbach, Benjamin and Gupta, Abhishek and Ibarz, Julian and Levine, Sergey},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=SJx63jRqFm}
}
```

## Requirements
We assume you have access to a GPU that can run CUDA 9.2 or newer.
1.  The simplest way to install all required dependencies is to create an anaconda environment:
    ```bash
    conda env create -f conda_env.yml
    ```
    (Ensure `opencv-python` is listed under `pip` dependencies in `conda_env.yml` for video processing features.)
2.  Activate your environment:
    ```bash
    conda activate drq_diayn
    ```
    (Note: The environment name might be `drq` if `conda_env.yml` was not updated for this project version).

## Running the Code

Train the agent using `python train.py` followed by command-line arguments.

**Key Command-Line Arguments:**

*   `--env`: Environment name (e.g., `cartpole_swingup`, `cheetah_run`).
*   `--seed`: Random seed (default: `1`).
*   `--num_train_steps`: Total number of training environment steps (default: `1000000`).
*   `--num_skills`: Number of skills for DIAYN. Set to `0` to run original DrQ without DIAYN (default: `10`).
*   `--encoder_feature_dim`: Feature dimension for the encoder (default: `50`).
*   `--lr`: Learning rate for actor, critic, and alpha temperature (default: `1e-3`).
*   `--lr_discriminator`: Learning rate for the DIAYN skill discriminator (defaults to `lr` if not specified).
*   `--diayn_intrinsic_reward_coeff`: Coefficient for the DIAYN intrinsic reward (default: `1.0`).
*   `--skill_embedding_dim`: Dimension for the skill embedding vector (default: `10`).
*   `--save_video`: Enable saving of evaluation videos (and logging to TensorBoard if `log_save_tb` is also true). Default: `True`.
*   `--log_save_tb`: Enable TensorBoard logging. Default: `True`.
*   `--device`: Device to use, e.g., `cuda` or `cpu` (default: `cuda`).

Many other arguments are available, mirroring the original `config.yaml` structure. Refer to `train.py` for the full list.

**Example Command (DrQ + DIAYN):**
```bash
CUDA_VISIBLE_DEVICES="0" python train.py --task dmc_walker_run --num_skills 10 --num_train_steps 1000000 --encoder_feature_dim 50 --lr 3e-4 --lr_discriminator 3e-4 --save_video True --log_save_tb True --time_limit 1000 --action_repeat 2 --framestack 3 --num_eval_episodes 1
```

**Example Command (Original DrQ):**
```bash
python train.py --env cartpole_swingup --num_skills 0 --num_train_steps 100000 \
--encoder_feature_dim 50 --lr 1e-3 --save_video --log_save_tb
```

This will produce a `runs` folder (e.g., `./runs/<env_name>/<timestamp>_seed<seed>`), where all outputs are stored, including:
*   TensorBoard logs (`tb` subfolder).
*   CSV logs (`train.csv`, `eval.csv` for aggregated metrics, and `episodes.csv` for detailed per-episode data).
*   Evaluation episode videos (if `--save_video` is enabled, in `video` subfolder).

To launch TensorBoard:
```bash
tensorboard --logdir runs
```

## Logging and Results

*   **TensorBoard:**
    *   Standard training metrics (losses, rewards, entropy, etc.).
    *   Evaluation episode rewards (extrinsic), reported per skill if DIAYN is active.
    *   Sum of intrinsic rewards per episode for each skill (training and evaluation).
    *   Videos of evaluation episodes, one for each skill, with intrinsic reward values overlaid on the frames.
*   **CSV Files:**
    *   `train.csv` / `eval.csv`: Aggregated metrics over logging intervals (similar to original DrQ).
    *   `episodes.csv`: Detailed per-episode logs including:
        *   `type` (train/eval)
        *   `episode` number
        *   `step` number (global)
        *   `duration` of the episode
        *   `episode_extrinsic_reward` (the actual reward from the environment)
        *   `skill_id` (the skill used for the episode, -1 if DIAYN is not active)
        *   `episode_intrinsic_reward_sum` (sum of DIAYN intrinsic rewards for the episode)

The console output format remains similar to the original DrQ, providing quick insights into training progress.

## The PlaNet Benchmark & The Dreamer Benchmark
(These sections from the original README remain relevant for the base DrQ performance but do not reflect DIAYN-specific results.)

**DrQ** demonstrates the state-of-the-art performance on a set of challenging image-based tasks from the DeepMind Control Suite (Tassa et al., 2018).
... (rest of these sections can be kept as is) ...

## Acknowledgements
We used [kornia](https://github.com/kornia/kornia) for data augmentation.
The DIAYN implementation is based on the principles from the original paper.
OpenCV (`opencv-python`) is used for text overlay on videos.
This project builds upon the original DrQ PyTorch implementation.
```
