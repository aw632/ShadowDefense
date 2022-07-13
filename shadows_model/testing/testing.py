import pickle
import gtsrb
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import torch.multiprocessing as mp
import torch.nn as nn
from utils import judge_mask_type
from utils import brightness
from utils import load_mask, draw_shadow, shadow_edge_blur
from utils import SmoothCrossEntropyLoss
from shadow_attack import attack
import DexiModel.main as dnm
from DexiModel.model import DexiNed
from DexiModel.datasets import TestDataset
import json
from tqdm import tqdm
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import subprocess
from joblib import Parallel, delayed
import torchvision.models


# BEGIN GLOBALS
REGIME_ONE_MODEL = "model/model_gtsrb.pth"
# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
# )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSITION_LIST, MASK_LIST = load_mask()
# INPUT_DIR = "testing/test_data/input"
# OUTPUT_DIR = "testing/test_data/output"
N_CLASS = 43  # 43 classes in GTSRB
REGIME_TWO_MODEL = "./testing/regime_two_model.pth"
DEXINED_MODEL = "./DexiModel/10_model.pth"
LOSS_FUN = SmoothCrossEntropyLoss(smoothing=0.1)


def regime_one(out_file):
    # load file from REGIME_ONE_MODEL and store in variable "model"
    # model = gtsrb.GtsrbCNN(N_CLASS).to(DEVICE)
    # model.load_state_dict(
    #     torch.load(
    #         REGIME_ONE_MODEL,
    #         map_location=DEVICE,
    #     )
    # )
    # # pre_process = transforms.Compose([pre_process_image, transforms.ToTensor()])
    # model.eval()  # set the model in evaluation mode

    # # generate the adversarial images by calling shadow_attack
    # # note: the images are saved irregardless of the success of the attack
    # with open("./dataset/GTSRB/test.pkl", "rb") as dataset:
    #     test_data = pickle.load(dataset)
    #     images, labels = test_data["data"], test_data["labels"]

    # results = {}
    # success_no_edges, total_num_images = generate_adv_images(images, labels)
    # # push the images through the edge profiler
    # generate_edge_profiles(512, 512)
    # # test it on the model using "test_single_image"
    # success_with_edges, confidence_with_edges, _ = evaluate_edge_profiles()

    # # robustness = 1 - success of attacks
    # results["robustness_no_edges"] = 1 - (success_no_edges / total_num_images)
    # results["robustness_with_edges"] = success_with_edges / total_num_images
    # results["confidence_with_edges"] = confidence_with_edges / total_num_images
    # # save the results to out_file
    # with open(out_file, "w") as f:
    #     json.dump(results, f)
    # raise deprecated error
    raise DeprecationWarning("This regime is deprecated. Use another regime instead.")


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def preprocess_image_nchan(image):
    """Preprocess the image. same as the paper author's but accounts for the
    4th channel.
    """
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    # image[:, :, 3] = cv2.equalizeHist(image[:, :, 3])
    image = image / 255.0 - 0.5
    return image


class RegimeTwoDataset(Dataset):
    """RegimeTwoDataset is a dataset class that takes in adversarial images from input and
    edge profiles from output and returns a dataset of images and labels, where
    images are adversarial images with edge profiles added as a channel.

    Args:
        Dataset (PyTorch Dataset): implements this superclass.
    """

    def __init__(self, images, labels):
        """Initializes the class."""
        super().__init__()
        # input_files = [join(input, file) for file in listdir(input)]
        # output_files = [join(output, file) for file in listdir(output)]
        # assert len(input_files) == len(
        #     output_files
        # ), "Must have the same number of input and output files"
        # self.files = list(zip(input_files, output_files))
        self.images = images
        self.labels = labels
        # self.transform = transform
        # self.use_adv = use_adv

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        # transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406, 0.48], std=[0.229, 0.224, 0.225, 0.225]
        #         ),
        #     ]
        # )
        # transform = transforms.Compose([transforms.ToTensor()])
        # img = transform(img)
        img = img.float()
        return img, label


class RegimeTwoCNN(nn.Module):
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
        self.fc3 = nn.Linear(1024, N_CLASS, bias=True)

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


def transform_img(image, ang_range, shear_range, trans_range, preprocess):
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

    image = preprocess_image_nchan(image) if preprocess else image

    return image


def predraw_shadows_and_edges(images, labels, use_adv, use_transform):
    new_images = []
    for idx in tqdm(range(len(images))):
        img, label = images[idx], labels[idx]
        if use_adv:  # if use_adv is True, then make adversarial images
            mask_type = judge_mask_type("GTSRB", label)
            # if brightness(img, MASK_LIST[mask_type]) >= 120:
            pos_list = POSITION_LIST[mask_type]
            shadow_image, shadow_area = draw_shadow(
                np.random.uniform(-16, 48, 6),
                img,
                pos_list,
                np.random.uniform(0.2, 0.7),
            )
            img = shadow_edge_blur(shadow_image, shadow_area, 3)
        # always add edge profile
        # FOR DEBUGGING
        # cv2.imwrite(f"./testing/test_data/input/{idx}_original.png", img)
        # img = preprocess_image_nchan(
        #     img.astype(np.uint8), use4chan=False
        # )  # improve contrast to help edge detection
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        edge_profile = auto_canny(blur.copy().astype(np.uint8))
        # # DEBUGGING
        # # cv2.imwrite(f"./testing/test_data/output/{idx}_edge.png", edge_profile)
        edge_profile = edge_profile[..., np.newaxis]
        img = np.concatenate((img, edge_profile), axis=2)
        if use_transform:
            # for _ in range(10):
            img = transform_img(
                image=img,
                ang_range=30,
                shear_range=5,
                trans_range=5,
                preprocess=not use_adv,
            )
        else:
            img = preprocess_image_nchan(img.astype(np.uint8))
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        new_images.append(img)
    return new_images


def train_model():
    with open("./dataset/GTSRB/train.pkl", "rb") as dataset:
        train_data = pickle.load(dataset)
        images, labels = train_data["data"], train_data["labels"]

    new_labels = torch.LongTensor(labels)
    datasets = []
    for trans, adv in [(False, False), (True, True), (False, True), (True, False)]:
        directory = "./testing/test_data/"
        filename = f"adv_{adv}_trans_{trans}_predrawn.pkl"
        full_path = f"{directory}{filename}"
        new_images = predraw_shadows_and_edges(
            images, new_labels, use_adv=adv, use_transform=trans
        )
        with open(full_path, "wb") as f:
            print("Saving new_images")
            pickle.dump(new_images, f)
        datasets.append(RegimeTwoDataset(new_images, new_labels))
    dataset_train = ConcatDataset(datasets)
    # dataset_train = gtsrb.TrafficSignDataset(x=images, y=labels)

    num_train = len(dataset_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.4 * num_train))
    train_idx = indices[:split]
    real_num = len(train_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    print(
        f"There are {num_train} examples in the dataset, and I am using {real_num} of them!"
    )

    dataloader_train = DataLoader(dataset_train, batch_size=64, sampler=train_sampler)

    print("******** I'm training the Regime Two Model Now! *****")
    num_epoch = 25
    training_model = RegimeTwoCNN().to(DEVICE).apply(gtsrb.weights_init)
    # training_model = torchvision.models.resnet50(
    #     weights=torchvision.models.ResNet50_Weights.DEFAULT
    # ).to(DEVICE)

    # # let ResNet accept 4 channels
    # first_layer_weights = training_model.conv1.weight.data.clone()
    # training_model.conv1 = nn.Conv2d(
    #     4, 64, kernel_size=7, stride=2, padding=3, bias=False
    # )
    # with torch.no_grad():
    #     training_model.conv1.weight.data[:, :3] = first_layer_weights
    #     training_model.conv1.weight.data[:, 3] = first_layer_weights[:, 0]  # ?
    # training_model.conv1 = training_model.conv1.to(DEVICE)

    # # let ResNet output 43 neurons for GTSRB
    # # block.expansion for ResNet 18 is 4, so 512 * block.expansion = 2048.
    # training_model.fc = nn.Linear(512 * 4, N_CLASS)
    # training_model.fc = training_model.fc.to(DEVICE)
    # training_model = training_model.to(torch.float)

    # use momentum optimiezer
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5
    )
    for epoch in range(num_epoch):
        # epoch_start = time.time()
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
        # epoch_end = time.time()
        print(
            f"Train Acc: {round(float(acc / real_num), 4)}",
            end=" ",
        )
        print(f"Loss: {round(float(loss / real_num), 4)}", end="\n")

    torch.save(
        training_model.state_dict(),
        REGIME_TWO_MODEL,
    )


def regime_two_a(out_file):
    """See instructions.md
    Args:
        out_file: file to write the results to
        fresh_start: if true, will train a fresh model, otherwise load the model.
    """
    # if fresh_start:
    #     train_model()
    #     subprocess.call(["sh", "./cleanup.sh"])
    model = RegimeTwoCNN().to(DEVICE)
    model = model.double()
    model.load_state_dict(
        torch.load(
            REGIME_TWO_MODEL,
            map_location=DEVICE,
        )
    )
    model.eval()

    with open("./dataset/GTSRB/test.pkl", "rb") as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data["data"], test_data["labels"]

    dataset_test = RegimeTwoDataset(
        images=images, labels=labels, transform=True, use_adv=True
    )
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

    print("******** I'm testing the Regime Two Model Now! *****")
    with torch.no_grad():
        loss = acc = 0.0

        for data_batch in dataloader_test:
            train_predict = model(data_batch[0].to(DEVICE))
            batch_loss = LOSS_FUN(train_predict, data_batch[1].to(DEVICE))
            acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
            loss += batch_loss.item() * len(data_batch[1])

    total_num_images = dataloader_test.__len__()
    results = {}
    print(f"Test Acc: {round(float(acc / total_num_images), 4)}")
    results["robustness_with_edges"] = round(float(acc / total_num_images), 4)
    # results["confidence_with_edges"] = confidence_with_edges / total_num_images
    # save the results to out_file
    with open(out_file, "w") as f:
        json.dump(results, f)


def regime_two_b(out_file, fresh_start=False):
    """See instructions.md
    Args:
        out_file: file to write the results to
        fresh_start: if true, will train a fresh model, otherwise load the model.
    """
    if fresh_start:
        train_model()
        subprocess.call(["sh", "./cleanup.sh"])
    model = RegimeTwoCNN().to(DEVICE)
    model.load_state_dict(
        torch.load(
            REGIME_TWO_MODEL,
            map_location=DEVICE,
        )
    )
    model.eval()

    raise NotImplementedError("Regime Two B is not implemented yet")


def test(regime, out_file):
    """Control flow for the desired testing regime.

    Args:
        regime (string): One of "TWO_A, TWO_B, TWO_C".
        out_file (string): name of the output file. Will be saved in
        /shadows_mode/testing.
    """
    match regime:
        # case "ONE":
        #     regime_one(out_file)
        case "TRAIN":
            train_model()
        case "TWO_A":
            regime_two_a(out_file)
        case "TWO_B":
            regime_two_b(out_file)
        case "TWO_C":
            raise ValueError("Regime 2C is not implemented")


def main():
    # open config.json
    with open("testing/config.json", "r") as f:
        config = json.load(f)
        #     output = "{}_{}.json".format(config["output"], config["regime"])
        regime = config["regime"]
        print("******** Now testing Regime {}! *****".format(regime))
        test(regime, "{}_{}.json".format(config["output"], regime))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
    # with open("./dataset/GTSRB/train.pkl", "rb") as dataset:
    #     train_data = pickle.load(dataset)
    #     images, labels = train_data["data"], train_data["labels"]

    # new_images = predraw_shadows_and_edges(
    #     images, torch.LongTensor(labels), False, False
    # )
