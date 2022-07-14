import pickle
from os import listdir

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from torchvision import transforms


from cnn_networks import AndrewNetCNN
from dataset import AndrewNetDataset
from shadow_utils import (
    SmoothCrossEntropyLoss,
    brightness,
    judge_mask_type,
    load_mask,
)
from shadow_attack import attack
from utils import predraw_shadows_and_edges, auto_canny

LOSS_FUN = SmoothCrossEntropyLoss(smoothing=0.1)
POSITION_LIST, MASK_LIST = load_mask()


def test_regime_a(testing_dataset, device, filename):
    # load the latest model
    if not filename:
        files = sorted(listdir("./checkpoints/"))
        filename = f"./checkpoints/{files[-2]}"
    # use 3 channels since we are using adversarial images with no edge profile.
    model = AndrewNetCNN(num_channels=4).to(device)
    model = model.float()
    model.load_state_dict(
        torch.load(
            filename,
            map_location=device,
        )
    )
    model.eval()

    with open(testing_dataset, "rb") as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data["data"], test_data["labels"]

    num_successes = 0
    total_num_query = 0
    num_images =int(np.floor(len(images) * 1))
    for index in trange(num_images):
        mask_type = judge_mask_type("GTSRB", labels[index])
        if brightness(images[index], MASK_LIST[mask_type]) >= 120:
            success, num_query = attack(
                images[index], labels[index], POSITION_LIST[mask_type], our_model=model
            )
            num_successes += success
            total_num_query += num_query

    avg_queries = round(float(total_num_query) / num_images, 4)
    robustness = 1 - round(float(num_successes / num_images), 4)
    print(f"Attack robustness: {robustness}")
    print(f"Average queries: {avg_queries}")
    results = {
        "robustness": robustness,
        "avg_queries": avg_queries,
    }
    with open(f"./testing_results/results_{files[-2][6:]}.json", "wb") as f:
        pickle.dump(results, f)


def main():
    print("Hello World!")


if __name__ == "__main__":
    main()
