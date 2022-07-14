import pickle
import torch
from utils import predraw_shadows_and_edges, weights_init
from dataset import AndrewNetDataset
from cnn_networks import AndrewNetCNN
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
import numpy as np
from datetime import datetime
import tqdm as tqdm
from shadow_utils import SmoothCrossEntropyLoss
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANET_MODEL = "checkpoints/model"
LOSS_FUN = SmoothCrossEntropyLoss(smoothing=0.1)


def train_model(args):
    with open(args.train_dataset_location, "rb") as dataset:
        train_data = pickle.load(dataset)
        images, labels = train_data["data"], train_data["labels"]

    new_labels = torch.LongTensor(labels)
    datasets = []
    for trans, adv in [(False, False), (True, True), (False, True), (True, False)]:
        new_images = predraw_shadows_and_edges(
            images, new_labels, use_adv=adv, use_transform=trans
        )
        datasets.append(AndrewNetDataset(new_images, new_labels))
    dataset_train = ConcatDataset(datasets)
    # dataset_train = gtsrb.TrafficSignDataset(x=images, y=labels)

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

    # use momentum optimiezer
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5
    )
    for epoch in tqdm(range(num_epoch)):
        print("NOW AT: Epoch {}".format(epoch))
        training_model.train()
        loss = acc = 0.0

        num_sample = 0
        for data_batch in tqdm(dataloader_train, leave=True):
            train_predict = training_model(data_batch[0].to(DEVICE))
            batch_loss = LOSS_FUN(train_predict, data_batch[1].to(DEVICE))
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
            loss += batch_loss.item() * len(data_batch[1])
            num_sample += len(data_batch[0])
        print(
            f"Train Acc: {round(float(acc / real_num), 4)}",
            end=" ",
        )
        print(f"Loss: {round(float(loss / real_num), 4)}", end="\n")

    filename = f"{ANET_MODEL}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(
        training_model.state_dict(),
        filename,
    )
    print(f"Model saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Entry point into the ANet Model.")
    parser.add_argument(
        "regime",
        type=str,
        choices=["TRAIN", "TEST_A", "TEST_B"],
        help="Regime to train the model on. Must be one of TRAIN, TEST_A, TEST_B, TEST_C.",
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
    args = parser.parse_args()
    match args.regime:
        case "TRAIN":
            train_model(args)
        case "TEST_A":
            raise NotImplementedError
        case "TEST_B":
            raise NotImplementedError


if __name__ == "__main__":
    main()
