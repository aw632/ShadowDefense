import pickle
from os import listdir
import json

import torch
from tqdm import trange

from cnn_networks import AndrewNetCNN
from dataset import AndrewNetDataset
from torch.utils.data import ConcatDataset, DataLoader
from shadow_attack import attack
from shadow_utils import SmoothCrossEntropyLoss, brightness, judge_mask_type, load_mask
from utils import auto_canny, predraw_shadows_and_edges
import numpy as np

LOSS_FUN = SmoothCrossEntropyLoss(smoothing=0.1)
POSITION_LIST, MASK_LIST = load_mask()


def test_regime_a(testing_dataset, device, filename=None):
    """Test_regime_a uses the images and labels from testing_dataset and loads
    the model weights from filename. If filename is none, it loads the latest
    model from ./checkpoints. It then sends the model to device.

    Requires: testing_dataset is saved as a .pkl file with an iterable of [images]
    accessible by the identifier images and an iterable of associated labels
    accessible by the identifier [labels].

    Args:
        testing_dataset (str): path to the testing dataset as a .pkl file.
        device (str): torch device, one of "cuda", "mps", "cpu".
        filename (str, optional): Name of the file to load model weights from. Defaults to None.
    """

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
    num_images = int(np.floor(len(images) * 1))
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
    with open(
        f"./testing_results/results_A_{files[-2][6:len(files[-2]) - 4]}.json", "wb"
    ) as f:
        json.dump(results, f)


def test_regime_b(testing_dataset, device, filename=None):
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

    new_labels = torch.LongTensor(labels)
    datasets = []
    print("Generating image datasets...")
    for trans, adv in [(False, False)]:
        new_images = predraw_shadows_and_edges(
            images, new_labels, use_adv=adv, use_transform=trans
        )
        datasets.append(AndrewNetDataset(new_images, new_labels))
    dataset_test = ConcatDataset(datasets)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

    print("******** I'm testing the AndrewNet Model Now! *****")
    with torch.no_grad():
        acc = 0.0

        for data_batch in dataloader_test:
            train_predict = model(data_batch[0].to(device))
            acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()

    print(
        f"Benign Test Acc: {round(float(acc / len(dataset_test)), 4)}",
        end=" ",
    )
    results = {
        "Benign Test Acc": round(float(acc / len(dataset_test)), 4),
    }
    with open(
        f"./testing_results/results_B_{files[-2][6:len(files[-2]) - 4]}.json", "wb"
    ) as f:
        json.dump(results, f)


def main():
    print("Hello World!")


if __name__ == "__main__":
    main()
