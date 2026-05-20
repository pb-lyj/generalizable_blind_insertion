Generalizable Vision-Free Robotic Insertion Policy via Physically-Consistent Tactile Representation Learning

This repository provides a two-stage pipeline for vision-free insertion.
First, train a tactile representation using 20x20 force array data under `representation/`.
Then, use the learned representation to train and run diffusion policy (dp) for control in `policy/`.

Quick usage (high level):
1) Train tactile representation: follow scripts in `representation/`.
2) Train and run policy: use `policy/` to train and perform inference with dp.
