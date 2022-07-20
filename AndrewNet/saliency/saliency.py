import torch
from cnn_networks import AndrewNetCNN
from utils import preprocess_image_nchan
from vanilla_backprop import VanillaBackprop
from torchvision import transforms
import cv2
import numpy as np


def grad_times_saliency():
    from PIL import Image
    from misc_functions import convert_to_grayscale, save_gradient_images
    from torch.autograd import Variable

    orig_model = AndrewNetCNN(num_channels=3)
    orig_model = orig_model.double()
    orig_model.load_state_dict(
        torch.load("./checkpoints/model_gtsrb.pth", map_location=torch.device("cpu"))
    )
    # img = Image.open("input_image.png").convert("RGB")
    img = cv2.imread("adv_input_image.png", cv2.IMREAD_COLOR)
    img = preprocess_image_nchan(img)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = img.unsqueeze_(0)
    img = Variable(img, requires_grad=True)
    VBP = VanillaBackprop(orig_model)
    vanilla_grads = VBP.generate_gradients(img, 0)
    grad_times_image = vanilla_grads * img.detach().numpy()[0]
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    save_gradient_images(grayscale_vanilla_grads, "adv_vanilla_grads.png")


if __name__ == "__main__":
    # with open("dataset/GTSRB/train.pkl", "rb") as f:
    #     train_data = pickle.load(f)
    #     train_images, train_labels = train_data["data"], train_data["labels"]
    # rand_int = np.random.randint(0, len(train_images))
    # img, label = train_images[rand_int], train_labels[rand_int]
    # mask_type = judge_mask_type("GTSRB", label)
    # # if brightness(img, MASK_LIST[mask_type]) >= 120:
    # pos_list = POSITION_LIST[mask_type]
    # shadow_image, shadow_area = draw_shadow(
    #     np.random.uniform(-16, 48, 6),
    #     img,
    #     pos_list,
    #     np.random.uniform(0.2, 0.7),
    # )
    # img = shadow_edge_blur(shadow_image, shadow_area, 3)
    # cv2.imwrite("adv_input_image.png", img)

    grad_times_saliency()
