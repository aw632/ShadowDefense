# Testing Instructions

This folder allows for testing the robustness of preprocessing defenses on the Zhong et al paper. 

## Current Testing Regimes

* *Regime 1*: use the pre-trained model with the GTSRB training dataset and test the accuracy of the binary edge profile versions of adversarially perturbed GTSRB test images. Further, collect the confidence information and obtain statistics on the confidence percentages for correctly classified and misclassified examples.

* *Regime 2*: add edge profiles of the GTSRB training dataset as a channel to the existing GTSRB images, and retrain the model with these. Then, 

  * *Regime 2A*: test the model on binary edge profile versions of adversarially perturbed GTSRB images.
  * *Regime 2B*: test the model on standard (i.e., no edge profile) adversarially perturbed GTSRB images.

  and collect statistics on accuracy and confidence for both subregimes.