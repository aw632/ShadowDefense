import pickle
import gtsrb
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import torch.multiprocessing as mp
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
# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
# )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSITION_LIST, MASK_LIST = load_mask()
INPUT_DIR = "testing/test_data/input"
OUTPUT_DIR = "testing/test_data/output"
N_CLASS = 43  # 43 classes in GTSRB
REGIME_TWO_MODEL = "./testing/regime_two_model.pth"
DEXINED_MODEL = "./DexiModel/10_model.pth"
LOSS_FUN = SmoothCrossEntropyLoss(smoothing=0.1)


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
    # for index in tqdm(range(len(images))):
    for index in range(1):
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
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

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


# class RegimeTwoDataset(Dataset):
#     """RegimeTwoDataset is a dataset class that takes in adversarial images from input and
#     edge profiles from output and returns a dataset of images and labels, where
#     images are adversarial images with edge profiles added as a channel.

#     Args:
#         Dataset (PyTorch Dataset): implements this superclass.
#     """

#     def __init__(self, input, output, transform):
#         """Initializes the class.

#         Args:
#             input (str): Directory from which to load adversarial images.
#             output (str): Directory from which to load edge profiles.
#             transform (bool): Whether to apply transforms to the images. See
#             transform_image function in the original paper's code.
#         """
#         super().__init__()
#         input_files = [join(input, file) for file in listdir(input)]
#         output_files = [join(output, file) for file in listdir(output)]
#         assert len(input_files) == len(
#             output_files
#         ), "Must have the same number of input and output files"
#         self.files = list(zip(input_files, output_files))
#         self.transform = transform

#     def __len__(self):
#         return len(self.files)

#     def transform_img(self, image):
#         """Applies several randomized transformations to image, including
#         shear, translation, and angle of rotation to improve robustness.
#         """
#         # magic numbers from the paper
#         ang_range, shear_range, trans_range = 30, 5, 5
#         ang_rot = np.random.uniform(ang_range) - ang_range / 2
#         rows, cols, ch = image.shape
#         rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

#         # Translation
#         tr_x = trans_range * np.random.uniform() - trans_range / 2
#         tr_y = trans_range * np.random.uniform() - trans_range / 2
#         trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

#         # Shear
#         pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

#         pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
#         pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

#         pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

#         shear_m = cv2.getAffineTransform(pts1, pts2)

#         image = cv2.warpAffine(image, rot_m, (cols, rows))
#         image = cv2.warpAffine(image, trans_m, (cols, rows))
#         image = cv2.warpAffine(image, shear_m, (cols, rows))

#         return image

#     def preprocess_image(self, image):
#         """Preprocess the image. same as the paper author's but accounts for the
#         4th channel.
#         """
#         image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
#         image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
#         image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
#         image[:, :, 3] = cv2.equalizeHist(image[:, :, 3])
#         image = image / 255.0 - 0.5
#         return image

#     def __getitem__(self, idx):
#         adv_image_path, edge_profile_path = self.files[idx]
#         adv_image = cv2.imread(adv_image_path, cv2.IMREAD_COLOR)
#         edge_profile = cv2.imread(edge_profile_path, cv2.IMREAD_GRAYSCALE)

#         assert (
#             adv_image.shape[0] == edge_profile.shape[0]
#             and adv_image.shape[1] == edge_profile.shape[1]
#         ), "Adv image and edge profile must be the same size"

#         transform = transforms.Compose([transforms.ToTensor()])
#         adv_image = transform(adv_image)
#         edge_profile = transform(edge_profile)
#         # dim 0 is the channels.
#         img = torch.cat((adv_image, edge_profile), dim=0)
#         img = img.numpy()
#         img = self.preprocess_image(img.astype(np.uint8))
#         if self.transform:
#             img = self.transform_img(img)

#         # image path looks like testing/test_data/output/<id>_<label>_<success>.png
#         label = int(adv_image_path.split("_")[2])
#         img = torch.from_numpy(img)
#         return img, label


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / (
        (np.max(img) - np.min(img)) + epsilon
    ) + img_min
    return img


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


class RegimeTwoDataset(Dataset):
    """RegimeTwoDataset is a dataset class that takes in adversarial images from input and
    edge profiles from output and returns a dataset of images and labels, where
    images are adversarial images with edge profiles added as a channel.

    Args:
        Dataset (PyTorch Dataset): implements this superclass.
    """

    def __init__(self, images, labels, transform, use_adv):
        """Initializes the class."""
        super().__init__()
        # input_files = [join(input, file) for file in listdir(input)]
        # output_files = [join(output, file) for file in listdir(output)]
        # assert len(input_files) == len(
        #     output_files
        # ), "Must have the same number of input and output files"
        # self.files = list(zip(input_files, output_files))
        self.images = images
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.use_adv = use_adv

    def __len__(self):
        return len(self.images)

    def transform_img(self, image):
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
        img, label = self.images[idx], self.labels[idx]
        # resize image to 224 by 224
        if self.use_adv:  # make edge profiles of adversarial images
            mask_type = judge_mask_type("GTSRB", label)
            if brightness(img, MASK_LIST[mask_type]) >= 120:
                img, _, _ = attack(img, label, POSITION_LIST[mask_type], testing=True)
        cv2.imwrite("./testing/test_data/input/{}_{}_adv.png".format(idx, label), img)
        #  #FOR DEBUGGING ONLY
        # add the edge profile
        transform = transforms.Compose([transforms.ToTensor()])
        edge_profile = auto_canny(img)
        cv2.imwrite(
            "./testing/test_data/output/{}_{}edge.png".format(idx, label), edge_profile
        )  # FOR DEBUGGING ONLY
        edge_profile = transform(edge_profile)
        img = transform(img)
        img = torch.cat((img, edge_profile), dim=0)
        img = img.numpy()
        if self.transform:
            img = self.transform_img(img)
        img = self.preprocess_image(img.astype(np.uint8))
        img = torch.from_numpy(img)
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
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.4),
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.4),
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.4),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14336, 1024, bias=True), nn.ReLU(), nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
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
    with open("./dataset/GTSRB/train.pkl", "rb") as dataset:
        train_data = pickle.load(dataset)
        images, labels = train_data["data"], train_data["labels"]

    datasets = []
    for trans, adv in tqdm(
        [(False, False), (True, True), (True, False), (False, True)]
    ):
        datasets.append(RegimeTwoDataset(images, labels, transform=trans, use_adv=adv))
    # datasets.append(
    #     RegimeTwoDataset(images=images, labels=labels, transform=False, use_adv=True)
    # )
    dataset_train = ConcatDataset(datasets)

    num_train = len(dataset_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.0001 * num_train))
    train_idx = indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    dataloader_train = DataLoader(
        dataset_train, batch_size=64, sampler=train_sampler, num_workers=6
    )

    print("******** I'm training the Regime Two Model Now! *****")
    num_epoch = 10
    training_model = RegimeTwoCNN().to(DEVICE).apply(gtsrb.weights_init)
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5
    )
    training_model = training_model.double()
    for epoch in range(num_epoch):
        # epoch_start = time.time()
        print("NOW AT: Epoch {}".format(epoch))
        training_model.train()
        loss = acc = 0.0

        counter = 0
        for data_batch in tqdm(dataloader_train):
            train_predict = training_model(data_batch[0].to(DEVICE))
            batch_loss = LOSS_FUN(train_predict, data_batch[1].to(DEVICE))
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
            loss += batch_loss.item() * len(data_batch[1])
            counter += 1
        # epoch_end = time.time()
        print(
            f"Train Acc: {round(float(acc / counter), 4)}",
            end=" ",
        )
        print(f"Loss: {round(float(loss / counter), 4)}", end="\n")

        torch.save(
            training_model.state_dict(),
            "./testing/regime_two_model_early{}".format(epoch),
        )

    torch.save(
        training_model.state_dict(),
        REGIME_TWO_MODEL,
    )


# def train_model():
#     """Train model trains the GtsrbCNN on the Regime 2 dataset."""
#     # generate adversarial images
#     with open("./dataset/GTSRB/train.pkl", "rb") as dataset:
#         train_data = pickle.load(dataset)
#         images, labels = train_data["data"], train_data["labels"]

#     _, _ = generate_adv_images(images, labels)
#     # push the images through the edge profiler
#     generate_edge_profiles(32, 32)
#     # we want 32 x 32 edge profiles, greyscale, so we can add as a channel
#     dataset_train_without_augmentations = RegimeTwoDataset(
#         input=INPUT_DIR, output=OUTPUT_DIR, transform=False
#     )
#     dataset_train_with_augmentations = RegimeTwoDataset(
#         input=INPUT_DIR, output=OUTPUT_DIR, transform=True
#     )
#     dataset_train = ConcatDataset(
#         [dataset_train_without_augmentations, dataset_train_with_augmentations]
#     )
#     dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=False)

#     print("******** I'm training the Regime Two Model Now! *****")
#     num_epoch = 15
#     training_model = RegimeTwoCNN().to(DEVICE).apply(gtsrb.weights_init)
#     optimizer = torch.optim.Adam(
#         training_model.parameters(), lr=0.001, weight_decay=1e-5
#     )
#     training_model = training_model.double()
#     for _ in range(num_epoch):
#         # epoch_start = time.time()
#         training_model.train()
#         loss = acc = 0.0

#         for data_batch in dataloader_train:
#             train_predict = training_model(data_batch[0].to(DEVICE))
#             batch_loss = LOSS_FUN(train_predict, data_batch[1].to(DEVICE))
#             batch_loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
#             loss += batch_loss.item() * len(data_batch[1])
#         # epoch_end = time.time()
#         print(
#             f"Train Acc: {round(float(acc / dataset_train.__len__()), 4)}",
#             end=" ",
#         )
#         print(f"Loss: {round(float(loss / dataset_train.__len__()), 4)}", end="\n")

#     torch.save(
#         training_model.state_dict(),
#         REGIME_TWO_MODEL,
#     )


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
    print(DEVICE)
    main()
