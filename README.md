# Shadows Aren’t So Dangerous: A Speedy and Robust Defense Against Shadow-Based Adversarial Attacks

This repository holds the code for our paper [PAPER-NAME] by Andrew Wang, Gopal Nookula, Wyatt Mayor and Ryan Smith.
The defenses are based off the adversarial attacks against road sign recognition models in [THIS PAPER] by Zhong et. al (2022). Our defenses achieve 78% robustness with a simple edge profile channel retrained on the paper's original CNN.
Please see our paper [HERE] for more details.

<!-- Each author's defense is in a folder in the format `{author_name}Net`, and instructions for running each defense can be found in `{author_name}Net/instructions.md`. Each folder also has `requirements.txt` which contain the dependencies needed. -->

> :warning: Note that for AndrewNet, requirements.txt was generated assuming an Anaconda installation. Please read the instructions inside requirements.txt for usage instructions.

Note that a number of files are stored with `git-lfs`, so you should first install `git lfs` and then enter `git lfs install` into
the root directory of this repository.

## Setup

Setup instructions can be found in `./AndrewNet/instructions.md`.

## Directory Overview

The

## Attributions

Please see `./AndrewNet/instructions.md` for attributions to others' work.