from torch.utils.data import Dataset


class AndrewNetDataset(Dataset):
    """RegimeTwoDataset is a dataset class that takes in adversarial images from input and
    edge profiles from output and returns a dataset of images and labels, where
    images are adversarial images with edge profiles added as a channel.

    Args:
        Dataset (PyTorch Dataset): implements this superclass.
    """

    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = img.float()
        return img, label
