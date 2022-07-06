import pickle
from gtsrb import GtsrbCNN
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import judge_mask_type
from utils import pre_process_image
from utils import brightness
from utils import load_mask
from shadow_attack import attack
import DexiNed.main
from DexiNed.model import DexiNed
from DexiNed.datasets import TestDataset
import json
import cv2

# BEGIN GLOBALS
MODEL_PATH = "model/model_gtsrb.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHADOW_LEVEL = 0.43
POSITION_LIST, MASK_LIST = load_mask()
INPUT_DIR = "testing/data/input"
OUTPUT_DIR = "testing/data/input"
N_CLASS = 43  # 43 classes in GTSRB


def regime_one(out_file):
    # load file from MODEL_PATH and store in variable "model"
    model = GtsrbCNN(N_CLASS).to(DEVICE)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device(DEVICE),
        )
    )
    pre_process = transforms.Compose([pre_process_image, transforms.ToTensor()])
    model.eval()  # set the model in evaluation mode

    # generate the adversarial images by calling shadow_attack
    # note: the images are saved irregardless of the success of the attack
    with open("./dataset/GTSRB/test.pkl", "rb") as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data["data"], test_data["labels"]

    results = {}
    success_no_edges = 0
    for index in range(len(images)):
        mask_type = judge_mask_type("GTSRB", labels[index])
        if brightness(images[index], MASK_LIST[mask_type]) >= 120:
            adv_img, success, _ = attack(
                images[index], labels[index], POSITION_LIST[mask_type], testing=True
            )
            success_no_edges += success
            cv2.imwrite(
                f"{INPUT_DIR}/{index}_{labels[index]}_{success}.png",
                adv_img,
            )
    print("******** Finished Generating Adversarial Images. *****")
    # push the images through the edge profiler
    dataset_val = TestDataset(
        INPUT_DIR,
        test_data="CLASSIC",
        img_width=512,
        img_height=512,
        mean_bgr=[103.939, 116.779, 123.68, 137.86][0:3],
        test_list=None,
        arg=None,
        # arg not needed since test_data = CLASSIC
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=16
    )

    print("******** Starting Testing. *****")
    DexiNed.main.test(
        "10_model.pth",
        dataloader_val,
        DexiNed,
        DEVICE,
        OUTPUT_DIR,
    )
    # test it on the model using "test_single_image"
    print("To be implemented later")
    # robustness = 1 - success of attacks
    results["success_no_edges"] = (1 - success_no_edges) / len(images)
    # save the results to out_file
    with open(out_file, "w") as f:
        json.dump(results, f)


def test(regime, out_file):
    match regime:
        case "ONE":
            regime_one(out_file)
        case "TWO_A":
            pass
        case "TWO_B":
            pass
        case "TWO_C":
            pass


def main():
    # open config.json
    with open("testing/config.json", "r") as f:
        config = json.load(f)
        test(config["regime"], config["output"])


if __name__ == "__main__":
    main()
