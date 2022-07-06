# Testing Instructions

This folder allows for testing the robustness of preprocessing defenses on the Zhong et al paper.

We use shadow_level = 0.43 (i.e., $k$) to be consistent with the paper.

## Current Testing Regimes

* *Regime 1*: use the pre-trained model with the GTSRB training dataset and test the accuracy of the binary edge profile versions of adversarially perturbed GTSRB test images. Further, collect the confidence information and obtain statistics on the confidence percentages for correctly classified and misclassified examples.

* *Regime 2*: add edge profiles of the GTSRB training dataset as a channel to the existing GTSRB images, and retrain the model with these. Then, 

  * *Regime 2A*: test the model on binary edge profile versions of adversarially perturbed GTSRB test images; that is, the edge profile is added as a channel to the adversarially perturbed images.
  * *Regime 2B*: test the model on solely binary edge profile versions of adversarially perturbed GTSRB test images; that is, not added as a channel.
  * *Regime 2C*: test the model on standard (i.e., no edge profile) adversarially perturbed GTSRB images.

  and collect statistics on accuracy and confidence for both subregimes.


## Steps to test

1. Run `python -m testing.testing [regime-id] [output-file-path]` **from the `shadows_model` folder** to run the testing regime and receive output in `[output-file-path]`. Output will be a `.json` file containing statistics as a Python dictionary.
2. Run `zsh cleanup.sh` or `bash cleanup.sh`, depending on your terminal, to clean up the `data/` folder for future use.