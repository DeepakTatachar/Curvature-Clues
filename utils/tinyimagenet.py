import torch
from PIL import Image
import os
import os.path
import sys

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def valid(value):
    return True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=valid, val=False):
    images = []
    dir = os.path.expanduser(dir)

    if val:
        d = os.path.join(dir, "val", "images")
        file = open(os.path.join(dir, "val", "val_annotations.txt"), "r+")
        lines = file.readlines()
        targets = {i.split("\t")[0]: class_to_idx[i.split("\t")[1]] for i in lines}
        file.close()
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, targets[fname])
                    images.append(item)

        return images

    target = "train"
    d = os.path.join(dir, target)
    for root, _, fnames in sorted(os.walk(d, followlinks=True)):
        for fname in sorted(fnames):
            file_is_image = any(file_extension in fname.lower() for file_extension in IMG_EXTENSIONS)

            if not file_is_image:
                continue

            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = (path, class_to_idx[fname.split("_")[0]])
                images.append(item)

    return images


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.loader = default_loader
        self.transform = transform
        self.target_transform = None

        classes, class_to_idx = self._find_classes(os.path.join(root, "train"))
        samples = make_dataset(self.root, class_to_idx, val=(not train))
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
