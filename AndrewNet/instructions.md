# AndrewNet Instructions

This document provides instructions and documentation regarding training and testing AndrewNet.

The primary entry point is `andrewnet.py`. For the purposes of this document, we assume you are currently in `./AndrewNet/` and all commands are being entered from there.

```
usage: andrewnet.py [-h] [-d DATA_FRACTION] [-train_l TRAIN_DATASET_LOCATION]
                    [-test_l TEST_DATASET_LOCATION] [-m MODEL_TO_TEST]
                    [-p PROPORTION]
                    {TRAIN,TEST_A,TEST_B,TEST_C1,TEST_C2}

Entry point into the ANet Model.

positional arguments:
  {TRAIN,TEST_A,TEST_B,TEST_C1,TEST_C2}
                        Regime to train the model on.

options:
  -h, --help            show this help message and exit
  -d DATA_FRACTION, --data_fraction DATA_FRACTION
                        Fraction of data to use for training.
  -train_l TRAIN_DATASET_LOCATION, --train_dataset_location TRAIN_DATASET_LOCATION
                        Location of the training dataset. Requires: dataset is
                        pickled.
  -test_l TEST_DATASET_LOCATION, --test_dataset_location TEST_DATASET_LOCATION
                        Location of the test dataset. Requires: dataset is
                        pickled.
  -m MODEL_TO_TEST, --model_to_test MODEL_TO_TEST
                        Location of the model to test. Requires: model is
                        .pth. If None, then use the most recent model in
                        ./checkpoints.
  -p PROPORTION, --proportion PROPORTION
                        Fraction of test data to use for testing.
```

## Environment Setup

The code was originally written in an Anaconda environment. The command


```sh
conda create --name my_name --file requirements.txt
```

can be used to setup a Conda virtual environment with the name `my_name` with the required dependencies.

## Checkpoint

We provide a checkpoint model in `./checkpoints/sample_model.pth` that we trained on the GTSRB training set. The training parameters, such as learning rate, can be found in `andrew_net.train_model` and model architecture can be found in `andrewnet.py`. The sample model is the same model found in the paper results. 

## Dataset

Placeholder text.

## Training and Testing

There are four regimes currently available to access.

**Regime TRAIN** trains the model on the provided training dataset.
 > :warning: Requires: the training dataset is saved as a `.pkl` file with an iterable of images accessible by the identifier `images` and an iterable of associated labels accessible by the identifier `labels`.

  The training regime takes these images and uses Canny Edge Detection to generate an edge profile, then appends the edge profile as a 4th channel to the original image.

  Some of the original images have a shadow randomly applied to them, as described in Zhong et al (2022), and some of the combined images are transformed with random shear, rotation, and translation.

Each of the lettered regimes test the model using the provided testing dataset, and collects stastictis on the robustness of the model.
> :warning: Requires: the training dataset is saved as a `.pkl` file with an iterable of images accessible by the identifier `images` and an iterable of associated labels accessible by the identifier `labels`.

**Regime TEST_A** tests the model on adversarially perturbed (shadows added) images from the test set, with associated edge profiles added as a channel.

**Regime TEST_B** tests the model on benign images from the test set, with associated edge profiles added as a channel.

**Regime TEST_C** tests the model on adversarial images from the test set, with added Gaussian noise $\mu = 0, \sigma = \frac{\sigma_{\text{train}}}{2}$, where $\sigma_{\text{train}}$ is the standard deviation of all pixels in all channels in the training set.

### Repeated Testing

An editable `test.sh` file is provided to train and test the same regime multiple times in order to get a sample of possible results. It can be called with `bash test.sh`. Ensure that it has permissions to execute with `chmod u+x test.sh`. 

## Results

If you are using Regime TRAIN, then results are saved into `./checkpoints` with the time that training was finished.

The file `zresults.json` contains a dictionary with the filenames of models as keys and their training accuracies as values. Confidence bars on the accuracy can be calculated from here.

If you are using one of the testing regimes, then a `.json` file containing statistics will be saved into `./testing_results`.

## Saliency

Saliency code is provided for completeness. Please note that this code is "quick and dirty" code intended to make images for the paper. Therefore, the quality of this code is not up to par with the rest of the code.

## Attributions

A number of files and functions in this paper are copies or reproductions of other's work. We give them attribution here.

`shadow_attack.py` and `shadow_utils.py`, as well as some functions in `utils.py` are adapted from Zhong et al. (2022), https://arxiv.org/abs/2203.03818.

`misc_functions.py`, `vanilla_backprop.py`, and portions of the code in `saliency.py` are adapted or reused from https://github.com/utkuozbulak/pytorch-cnn-visualizations. 
