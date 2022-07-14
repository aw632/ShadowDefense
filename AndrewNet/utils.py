import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from shadow_utils import draw_shadow, judge_mask_type, load_mask, shadow_edge_blur

POSITION_LIST, MASK_LIST = load_mask()


def auto_canny(image, sigma=0.33):
    """Finds good parameters for Canny Edge Detection with the median of the image.

    Args:
        image (ndarray): image as an ndarray.
        sigma (float, optional): sigma parameter in Canny Edge Detection. Defaults to 0.33.

    Returns:
        _type_: _description_
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def preprocess_image_nchan(image):
    """Preprocess the image. same as the paper author's.
    Note I commented out the equalization for the last channel because it's a black
    and white image - nothing will change.
    """
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    # image[:, :, 3] = cv2.equalizeHist(image[:, :, 3])
    image = image / 255.0 - 0.5
    return image


def transform_img(image, ang_range, shear_range, trans_range, preprocess):
    """Randomly transform the image with rotation, shear and translation. Afterwards,
    preprocess the image.

    Args:
        image (ndarry): image as an ndarray
        ang_range (int): range of angle rotation
        shear_range (int): range of shear
        trans_range (int): range of translation
        preprocess (bool): whether to preprocess the transformed image or not.

    Returns:
        ndarray: a transformed image.
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

    image = preprocess_image_nchan(image) if preprocess else image

    return image


def predraw_shadows_and_edges(images, labels, use_adv, use_transform):
    """Draw shadows and generate edge profiles for every image in images, then
    stack the edge profile as a new channel for images.

    Args:
        images (iterable): Iterable of images stored as ndarray with dtype=uint8.
        labels (iterable): Iterable of labels stored as integers.
        use_adv (bool): if True, then draw shadows.
        use_transform (bool): if True, then apply random transformations to the images.

    Returns:
        list: a list of ndarray with dtype=float64 representing images.
    """
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


def weights_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.05)
        nn.init.constant_(m.bias, 0.05)
