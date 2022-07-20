import argparse
import pickle
import test
from datetime import datetime
import json

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

from cnn_networks import AndrewNetCNN
from dataset import AndrewNetDataset
from shadow_utils import SmoothCrossEntropyLoss
from utils import predraw_shadows_and_edges, weights_init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANET_MODEL = "./checkpoints/"
LOSS_FUN = SmoothCrossEntropyLoss(smoothing=0.1)


def train_model(args):
    """Trains the model. See instructions.md for more information.
    This function should not be called by the user directly.
    """
    with open(args.train_dataset_location, "rb") as dataset:
        train_data = pickle.load(dataset)
        images, labels = train_data["data"], train_data["labels"]

    new_labels = torch.LongTensor(labels)
    datasets = []
    print("Generating image datasets...")
    for trans, adv in [(False, False), (True, True), (False, True), (True, False)]:
        new_images = predraw_shadows_and_edges(
            images, new_labels, use_adv=adv, use_transform=trans
        )
        datasets.append(AndrewNetDataset(new_images, new_labels))
    dataset_train = ConcatDataset(datasets)

    num_train = len(dataset_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(args.data_fraction * num_train))
    train_idx = indices[:split]
    real_num = len(train_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    print(
        f"There are {num_train} examples in the dataset, and I am using {real_num} of them!"
    )

    dataloader_train = DataLoader(dataset_train, batch_size=64, sampler=train_sampler)

    print("******** I'm training the AndrewNet Model Now! *****")
    num_epoch = 25
    training_model = AndrewNetCNN().to(DEVICE).apply(weights_init)

    overall_acc = 0
    # use momentum optimiezer
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5
    )
    for epoch in range(num_epoch):
        print("NOW AT: Epoch {}".format(epoch))
        training_model.train()
        loss = acc = 0.0

        num_sample = 0
        for data_batch in tqdm(dataloader_train):
            train_predict = training_model(data_batch[0].to(DEVICE))
            batch_loss = LOSS_FUN(train_predict, data_batch[1].to(DEVICE))
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
            loss += batch_loss.item() * len(data_batch[1])
            num_sample += len(data_batch[0])

        epoch_acc = round(float(acc / real_num), 4)
        print(
            f"Train Acc: {epoch_acc}",
            end=" ",
        )
        print(f"Loss: {round(float(loss / real_num), 4)}", end="\n")
        overall_acc += epoch_acc

    remainder_of_file_name = (
        f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{num_epoch}.pth"
    )
    filename = f"{ANET_MODEL}{remainder_of_file_name}"
    torch.save(
        training_model.state_dict(),
        filename,
    )
    print(f"Model saved to {filename}!")
    avg_acc = overall_acc / num_epoch
    with open("./checkpoints/zresults.json", "r") as f:
        results = json.load(f)

    results[remainder_of_file_name] = str(avg_acc)
    with open("./checkpoints/zresults.json", "w") as f:
        json.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description="Entry point into the ANet Model.")
    parser.add_argument(
        "regime",
        type=str,
        choices=["TRAIN", "TEST_A", "TEST_B", "TEST_C1", "TEST_C2"],
        help="Regime to train the model on.",
    )
    parser.add_argument(
        "-d",
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use for training.",
    )
    parser.add_argument(
        "-train_l",
        "--train_dataset_location",
        type=str,
        default="./dataset/GTSRB/train.pkl",
        help="Location of the training dataset. Requires: dataset is pickled.",
    )
    parser.add_argument(
        "-test_l",
        "--test_dataset_location",
        type=str,
        default="./dataset/GTSRB/test.pkl",
        help="Location of the test dataset. Requires: dataset is pickled.",
    )
    parser.add_argument(
        "-m",
        "--model_to_test",
        type=str,
        default=None,
        help="Location of the model to test. Requires: model is .pth. If None, then \
            use the most recent model in ./checkpoints.",
    )
    parser.add_argument(
        "-p",
        "--proportion",
        type=float,
        default=1.0,
        help="Fraction of test data to use for testing.",
    )
    args = parser.parse_args()
    match args.regime:
        case "TRAIN":
            train_model(args)
        case "TEST_A":
            test.test_regime_a(
                args.test_dataset_location,
                DEVICE,
                args.proportion,
                args.model_to_test,
            )
        case "TEST_B":
            test.test_regime_b(args.test_dataset_location, DEVICE, args.model_to_test)
        case "TEST_C1":
            test.test_regime_c(
                args.train_dataset_location,
                args.test_dataset_location,
                DEVICE,
                args.proportion,
                True,
                args.model_to_test,
            )
        case "TEST_C2":
            test.test_regime_c(
                args.train_dataset_location,
                args.test_dataset_location,
                DEVICE,
                args.proportion,
                False,
                args.model_to_test,
            )


if __name__ == "__main__":
    main()
