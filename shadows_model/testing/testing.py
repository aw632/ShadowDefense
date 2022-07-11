import pickle
import gtsrb
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
from utils import judge_mask_type
from utils import brightness
from utils import load_mask
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

# BEGIN GLOBALS
REGIME_ONE_MODEL = "model/model_gtsrb.pth"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSITION_LIST, MASK_LIST = load_mask()
INPUT_DIR = "testing/test_data/input"
OUTPUT_DIR = "testing/test_data/output"
N_CLASS = 43  # 43 classes in GTSRB
REGIME_TWO_MODEL = "./testing/regime_two_model.pth"


def generate_adv_images(images, labels):
    """Generates adversarial images by applying an artificial geometric shadow.
    See "Shadows Can Be Dangerous" paper by Zhong et al, 2022 for more details.

    Args:
        images (array of tensors): array of 3 channel RGB tensors representing images.
        labels : class labels for each image in images.

    Returns:
        success_no_edges (int): number of images that were successfully attacked.
        total_num_images (int): total number of images in images.
    """
    success_no_edges = 0
    total_num_images = 0
    for index in tqdm(range(len(images))):
        mask_type = judge_mask_type("GTSRB", labels[index])
        if brightness(images[index], MASK_LIST[mask_type]) >= 120:
            adv_img, success, _ = attack(
                images[index], labels[index], POSITION_LIST[mask_type], testing=True
            )
            success_no_edges += success
            total_num_images += 1
            cv2.imwrite(
                f"{INPUT_DIR}/{index}_{labels[index]}_{success}.png",
                adv_img,
            )
    print("******** Finished Generating Adversarial Images. *****")
    return success_no_edges, total_num_images


def generate_edge_profiles(width, height):
    """Generates edge profiles using DexiNed model. Outputs the edge profiles into
    OUPUT_DIR.

    Args:
        width (int): width of the edge profile in pixels
        height (int): height of the edge profile in pixels
    """
    dataset_val = TestDataset(
        INPUT_DIR,
        test_data="CLASSIC",
        img_width=width,
        img_height=height,
        mean_bgr=[103.939, 116.779, 123.68, 137.86][0:3],
        test_list=None,
        arg=None,
        # arg not needed since test_data = CLASSIC
    )
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=6)

    deximodel = DexiNed().to(DEVICE)
    print("******** Starting Testing. *****")
    dnm.test(
        "10_model.pth",
        dataloader_val,
        deximodel,
        DEVICE,
        OUTPUT_DIR,
    )


def evaluate_edge_profiles():
    success_with_edges = 0
    confidence_with_edges = 0
    total_num_images = 0
    for file in tqdm(listdir(OUTPUT_DIR)):
        path = join(OUTPUT_DIR, file)
        if isfile(path):
            # extract the string between the first and second underscore in file
            # and convert it to an integer
            label = int(file.split("_")[1])
            _, success, confidence = gtsrb.test_single_image(path, label)
            success_with_edges += (
                success  # note, success here refers to a successful classification
            )
            confidence_with_edges += confidence
            total_num_images += 1

    return success_with_edges, confidence_with_edges, total_num_images


def regime_one(out_file):
    # load file from REGIME_ONE_MODEL and store in variable "model"
    model = gtsrb.GtsrbCNN(N_CLASS).to(DEVICE)
    model.load_state_dict(
        torch.load(
            REGIME_ONE_MODEL,
            map_location=DEVICE,
        )
    )
    # pre_process = transforms.Compose([pre_process_image, transforms.ToTensor()])
    model.eval()  # set the model in evaluation mode

    # generate the adversarial images by calling shadow_attack
    # note: the images are saved irregardless of the success of the attack
    with open("./dataset/GTSRB/test.pkl", "rb") as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data["data"], test_data["labels"]

    results = {}
    success_no_edges, total_num_images = generate_adv_images(images, labels)
    # push the images through the edge profiler
    generate_edge_profiles(512, 512)
    # test it on the model using "test_single_image"
    success_with_edges, confidence_with_edges, _ = evaluate_edge_profiles()

    # robustness = 1 - success of attacks
    results["robustness_no_edges"] = 1 - (success_no_edges / total_num_images)
    results["robustness_with_edges"] = success_with_edges / total_num_images
    results["confidence_with_edges"] = confidence_with_edges / total_num_images
    # save the results to out_file
    with open(out_file, "w") as f:
        json.dump(results, f)


class RegimeTwoDataset(Dataset):
    """RegimeTwoDataset is a dataset class that takes in adversarial images from input and
    edge profiles from output and returns a dataset of images and labels, where
    images are adversarial images with edge profiles added as a channel.

    Args:
        Dataset (PyTorch Dataset): implements this superclass.
    """

    def __init__(self, input, output, transform):
        """Initializes the class.

        Args:
            input (str): Directory from which to load adversarial images.
            output (str): Directory from which to load edge profiles.
            transform (bool): Whether to apply transforms to the images. See
            transform_image function in the original paper's code.
        """
        super().__init__()
        input_files = [join(input, file) for file in listdir(input)]
        output_files = [join(output, file) for file in listdir(output)]
        assert len(input_files) == len(
            output_files
        ), "Must have the same number of input and output files"
        self.files = list(zip(input_files, output_files))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def transform(self, image):
        """Applies several randomized transformations to image, including
        shear, translation, and angle of rotation to improve robustness.
        """
        # magic numbers from the paper
        ang_range, shear_range, trans_range = 30, 5, 5
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

        return image

    def preprocess_image(self, image):
        """Preprocess the image. same as the paper author's but accounts for the
        4th channel.
        """
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
        image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        image[:, :, 3] = cv2.equalizeHist(image[:, :, 3])
        image = image / 255.0 - 0.5
        return image

    def __getitem__(self, idx):
        adv_image_path, edge_profile_path = self.files[idx]
        adv_image = cv2.imread(adv_image_path, cv2.IMREAD_COLOR)
        edge_profile = cv2.imread(edge_profile_path, cv2.IMREAD_GRAYSCALE)

        assert (
            adv_image.shape[0] == edge_profile.shape[0]
            and adv_image.shape[1] == edge_profile.shape[1]
        ), "Adv image and edge profile must be the same size"

        transform = transforms.Compose([transforms.ToTensor()])
        adv_image = transform(adv_image)
        edge_profile = transform(edge_profile)
        # dim 0 is the channels.
        img = torch.cat((adv_image, edge_profile), dim=0)
        img = self.preprocess_image(img)
        if self.transform:
            img = self.transform(img)

        # image path looks like testing/test_data/output/<id>_<label>_<success>.png
        label = int(adv_image_path.split("_")[2])
        return img, label


class RegimeTwoCNN(nn.Module):
    def __init__(self):

        super().__init__()
        self.color_map = nn.Conv2d(4, 4, (1, 1), stride=(1, 1), padding=0)
        self.module1 = nn.Sequential(
            nn.Conv2d(4, 32, (5, 5), stride=(1, 1), padding=2),
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


def train_model():
    """Train model trains the GtsrbCNN on the Regime 2 dataset."""
    # generate adversarial images
    with open("./dataset/GTSRB/train.pkl", "rb") as dataset:
        train_data = pickle.load(dataset)
        images, labels = train_data["data"], train_data["labels"]

    _, _ = generate_adv_images(images, labels)
    # push the images through the edge profiler
    generate_edge_profiles(32, 32)
    # we want 32 x 32 edge profiles, greyscale, so we can add as a channel
    dataset_train_without_augmentations = RegimeTwoDataset(
        input=INPUT_DIR, output=OUTPUT_DIR, transform=False
    )
    dataset_train_with_augmentations = RegimeTwoDataset(
        input=INPUT_DIR, output=OUTPUT_DIR, transform=True
    )
    dataset_train = ConcatDataset(
        [dataset_train_without_augmentations, dataset_train_with_augmentations]
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=64, shuffle=False, num_workers=6
    )
    loss_fun = SmoothCrossEntropyLoss(smoothing=0.1)

    num_epoch = 15
    optimizer = torch.optim.Adam(RegimeTwoCNN.parameters(), lr=0.001, weight_decay=1e-5)
    training_model = RegimeTwoCNN().to(DEVICE).apply(gtsrb.weights_init)
    for _ in range(num_epoch):
        # epoch_start = time.time()
        training_model.train()
        loss = acc = 0.0

        for data_batch in dataloader_train:
            train_predict = training_model(data_batch[0].to(DEVICE))
            batch_loss = loss_fun(train_predict, data_batch[1].to(DEVICE))
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
            loss += batch_loss.item() * len(data_batch[1])
        # epoch_end = time.time()
        print(
            f"Train Acc: {round(float(acc / dataset_train.__len__()), 4)}",
            end=" ",
        )
        print(f"Loss: {round(float(loss / dataset_train.__len__()), 4)}", end=" | ")

    torch.save(
        training_model.state_dict(),
        REGIME_TWO_MODEL,
    )


def regime_two_a(out_file, fresh_start=False):
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

    with open("./dataset/GTSRB/test.pkl", "rb") as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data["data"], test_data["labels"]

    generate_adv_images(images, labels)
    generate_edge_profiles(32, 32)
    (
        success_with_edges,
        confidence_with_edges,
        total_num_images,
    ) = evaluate_edge_profiles()
    results = {}
    results["robustness_with_edges"] = success_with_edges / total_num_images
    results["confidence_with_edges"] = confidence_with_edges / total_num_images
    # save the results to out_file
    with open(out_file, "w") as f:
        json.dump(results, f)


def test(regime, out_file):
    """Control flow for the desired testing regime.

    Args:
        regime (string): One of "ONE, TWO_A, TWO_B, TWO_C".
        out_file (string): name of the output file. Will be saved in
        /shadows_mode/testing.
    """
    match regime:
        case "ONE":
            regime_one(out_file)
        case "TWO_A":
            regime_two_a(out_file, fresh_start=True)
        case "TWO_B":
            raise ValueError("Regime 2B is not implemented")
        case "TWO_C":
            raise ValueError("Regime 2C is not implemented")


def main():
    # open config.json
    with open("testing/config.json", "r") as f:
        config = json.load(f)
        output = "{}_{}.json".format(config["output"], config["regime"])
        test(config["regime"], output)


if __name__ == "__main__":
    main()
