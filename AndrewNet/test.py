import pickle
from os import listdir
import json

import torch
import cv2
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


def prep_regimes(testing_dataset, device, filename=None):
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

    Returns:
        tuple: tuple of (filename, images, labels, model)
    """
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

    with open(testing_dataset, "rb") as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data["data"], test_data["labels"]

    return files, filename, images, labels, model


def test_regime_a(testing_dataset, device, proportion, filename=None):
    """See instructions.md for details on Test Regime A.

    Requires: testing_dataset is saved as a .pkl file with an iterable of [images]
    accessible by the identifier images and an iterable of associated labels
    accessible by the identifier [labels].

    Args:
        testing_dataset (str): path to the testing dataset as a .pkl file.
        device (str): torch device, one of "cuda", "mps", "cpu".
        filename (str, optional): Name of the file to load model weights from. Defaults to None.
    """

    # load the latest model
    files, filename, images, labels, model = prep_regimes(
        testing_dataset=testing_dataset, device=device, filename=filename
    )
    model.eval()

    num_successes = 0
    total_num_query = 0
    num_images = int(np.floor(len(images) * proportion))
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
        "Robustness": robustness,
        "Average number of queries": avg_queries,
        "Dataset Size": num_images,
        "Batch Size": 64,
    }
    with open(
        f"./testing_results/results_A_{files[-2][6:len(files[-2]) - 4]}.json", "wb"
    ) as f:
        json.dump(results, f)


def test_regime_b(testing_dataset, device, filename=None):
    # load the latest model
    files, filename, images, labels, model = prep_regimes(
        testing_dataset=testing_dataset, device=device, filename=filename
    )
    model.eval()

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
        "Dataset Size": len(dataset_test),
        "Batch Size": 64,
    }
    with open(
        f"./testing_results/results_B_{files[-2][6:len(files[-2]) - 4]}.json", "w"
    ) as f:
        json.dump(results, f)


def test_regime_c(
    training_dataset, testing_dataset, device, proportion, attack, filename=None
):
    files, filename, test_images, test_labels, model = prep_regimes(
        testing_dataset=testing_dataset, device=device, filename=filename
    )
    model.eval()

    # calculate the stdev of all pixels across all channels in the training set
    with open(training_dataset, "rb") as dataset:
        train_data = pickle.load(dataset)
        train_images, train_labels = train_data["data"], train_data["labels"]

    new_labels = torch.LongTensor(train_labels)
    all_images = predraw_shadows_and_edges(
        train_images, new_labels, use_adv=False, use_transform=False, make_tensor=False
    )
    all_images = np.array(all_images)
    for trans, adv in [(True, True), (True, False), (False, True)]:
        new_images = predraw_shadows_and_edges(
            train_images,
            new_labels,
            use_adv=adv,
            use_transform=trans,
            make_tensor=False,
        )
        new_images = np.array(new_images)
        all_images = np.concatenate((all_images, new_images), axis=0)
    all_images = np.ndarray.flatten(all_images)
    sigmaprime = np.std(all_images)
    sigma = round(float(sigmaprime / 5), 4)
    mean = 0

    # add gaussian noise with mean, sigma to every image in test_images
    test_images = np.array(test_images)
    noise = np.clip(
        np.random.normal(mean, sigma, test_images.shape).astype(np.uint8), 0, 255
    )
    test_images += noise

    if attack:
        num_successes = 0
        total_num_query = 0
        num_images = int(np.floor(len(test_images) * proportion))
        for index in trange(num_images):
            mask_type = judge_mask_type("GTSRB", test_labels[index])
            if brightness(test_images[index], MASK_LIST[mask_type]) >= 120:
                success, num_query = attack(
                    test_images[index],
                    test_labels[index],
                    POSITION_LIST[mask_type],
                    our_model=model,
                )
                num_successes += success
                total_num_query += num_query

        avg_queries = round(float(total_num_query) / num_images, 4)
        robustness = 1 - round(float(num_successes / num_images), 4)
        print(f"Attack robustness: {robustness}")
        print(f"Average queries: {avg_queries}")
        results = {
            "Robustness": robustness,
            "Average number of queries": avg_queries,
            "Dataset Size": num_images,
            "Batch Size": 64,
        }
        with open(
            f"./testing_results/results_A_{files[-2][6:len(files[-2]) - 4]}.json", "w"
        ) as f:
            json.dump(results, f)
    else:
        img = test_images[0]
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        edge_profile = auto_canny(blur.copy().astype(np.uint8))
        cv2.imshow("edge profile", edge_profile)
        cv2.waitKey(0)

        # # DEBUGGING
        # # cv2.imwrite(f"./testing/test_data/output/{idx}_edge.png", edge_profile)
        edge_profile = edge_profile[..., np.newaxis]
        img = np.concatenate((img, edge_profile), axis=2)

        cv2.imshow("image", img)
        cv2.waitKey(0)
        datasets = []
        print("Generating image datasets...")
        for trans, adv in [(False, False)]:
            new_images = predraw_shadows_and_edges(
                test_images, new_labels, use_adv=adv, use_transform=trans
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
            f"Noisy Test Acc: {round(float(acc / len(dataset_test)), 4)}",
            end=" ",
        )
        results = {
            "Noisy Test Acc": round(float(acc / len(dataset_test)), 4),
            "Dataset Size": len(dataset_test),
            "Batch Size": 64,
        }
        with open(
            f"./testing_results/results_B_{files[-2][6:len(files[-2]) - 4]}.json", "w"
        ) as f:
            json.dump(results, f)


def main():
    print("Hello World!")


if __name__ == "__main__":
    main()
