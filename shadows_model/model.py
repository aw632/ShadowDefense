import gc

# import pickle
import time

# import json
import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import SmoothCrossEntropyLoss

from utils import draw_shadow
from utils import shadow_edge_blur
from utils import judge_mask_type
from utils import load_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fun = SmoothCrossEntropyLoss(smoothing=0.1)

# BEGIN NEURAL NETWORK
class GtsrbCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.color_map = nn.Conv2d(3, 3, (1, 1), stride=(1, 1), padding=0)
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14336, 1024, bias=True), nn.ReLU(), nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Linear(1024, 43, bias=True)

    def forward(self, x):
        x = self.color_map(x)
        branch1 = self.module1(x)
        branch2 = self.module2(branch1)
        branch3 = self.module3(branch2)

        branch1 = branch1.reshape(branch1.shape[0], -1)
        branch2 = branch2.reshape(branch2.shape[0], -1)
        branch3 = branch3.reshape(branch3.shape[0], -1)
        concat = torch.cat([branch1, branch2, branch3], 1)

        out = self.fc1(concat)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# BEGIN HELPER FUNCTIONS
def pre_process_image(image):
    """Normalizes [image] to 0-1 range with standard Gaussian distribution.

    Args:
        image (C x H x W tensor): a 3D tensor representing image.

    Returns:
        C x H x W tensor: the normalized image
    """
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = image / 255.0 - 0.5
    return image


def transform_image(image, ang_range, shear_range, trans_range, preprocess):
    """Applies set of transformations to [image].

    Returns:
        C x H x W tensor: the transformed image
    """
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = image.shape
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_m = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, rot_m, (cols, rows))
    image = cv2.warpAffine(image, trans_m, (cols, rows))
    image = cv2.warpAffine(image, shear_m, (cols, rows))

    image = pre_process_image(image) if preprocess else image

    return image


def gen_extra_data(
    x_train,
    y_train,
    n_each,
    ang_range,
    shear_range,
    trans_range,
    randomize_var,
    preprocess=True,
):
    """Transforms each of the [x_train] training images [n_each] times, and outputs
    the transformed images with labels.
    """
    x_arr, y_arr = [], []
    n_train = len(x_train)
    for i in range(n_train):
        for i_n in range(n_each):
            img_trf = transform_image(
                x_train[i], ang_range, shear_range, trans_range, preprocess
            )
            x_arr.append(img_trf)
            y_arr.append(y_train[i])

    x_arr = np.array(x_arr, dtype=np.float32())
    y_arr = np.array(y_arr, dtype=np.float32())

    if randomize_var == 1:
        len_arr = np.arange(len(y_arr))
        np.random.shuffle(len_arr)
        x_arr[len_arr] = x_arr
        y_arr[len_arr] = y_arr

    return x_arr, y_arr


def weights_init(m):
    """Initializes the weights of [m] with a normal distribution with mean 0 and std=0.05"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.05)
        nn.init.constant_(m.bias, 0.05)


def model_epoch(training_model, data_loader, train=False, optimizer=None):
    """Trains or evaluates the [training_model] on the data in [data_loader].
    This accounts for one epoch of the model."""
    loss = acc = 0.0

    for data_batch in data_loader:
        train_predict = training_model(data_batch[0].to(device))
        batch_loss = loss_fun(train_predict, data_batch[1].to(device))
        if train:  # if we want to train, then calculate the gradients
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # check the accuracy of the model on this example. sums up a tensor which contains
        # at most one "1" and rest are zeroes.
        acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
        loss += batch_loss.item() * len(data_batch[1])

    return acc, loss


def training(
    training_model, train_loader, train_set, test_loader, test_set, adv_train=False
):

    num_epoch, batch_size = 25, 64
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5
    )

    for epoch in range(num_epoch):

        extra_train, extra_labels = (
            # TODO: define adaug
            adversarial_augmentation(train_loader)
            if adv_train
            else non_adversarial(train_loader)
        )

        # # TODO: build the train and test sets
        # train_set = TrafficSignDataset(extra_train, extra_labels)
        # test_set = TrafficSignDataset(test_data, test_labels)
        # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        epoch_start_time = time.time()

        training_model.train()
        train_acc, train_loss = model_epoch(
            training_model, train_loader, train=True, optimizer=optimizer
        )

        training_model.eval()
        with torch.no_grad():
            test_acc, test_loss = model_epoch(training_model, test_loader)

        print(
            f"[{epoch+1}/{num_epoch}] {round(time.time() - epoch_start_time, 2)}",
            end=" ",
        )
        print(f"Train Acc: {round(float(train_acc / len(train_set)), 4)}", end=" ")
        print(f"Loss: {round(float(train_loss / len(train_set)), 4)}", end=" | ")
        print(f"Test Acc: {round(float(test_acc / len(test_set)), 4)}", end=" ")
        print(f"Loss: {round(float(test_loss / len(test_set)), 4)}")

        del extra_train, extra_labels, train_set, train_loader
        gc.collect()

    torch.save(
        training_model.state_dict(),
        f'./model/{"adv_" if adv_train else ""}model_gtsrb.pth',
    )
