import random
from pathlib import Path

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def get_default_transform(resolution):
    photo_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    sketch_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )
    # prompt_transform = lambda classname: f"a photo of {classname}"
    prompt_transform = lambda classname: "a high-quality, detailed, and professional image"
    return photo_transform, sketch_transform, prompt_transform


def get_train_dataset(data_root, resolution):
    return get_dataset(data_root, data_split="train", resolution=resolution)


def get_val_dataset(data_root, resolution):
    return get_dataset(data_root, data_split="val", resolution=resolution)


def get_dataset(data_root, data_split, resolution):
    photo_transform, sketch_transform, prompt_transform = get_default_transform(resolution)
    dataset = SketchyDataset(
        data_root,
        split=data_split,
        photo_transform=photo_transform,
        sketch_transform=sketch_transform,
        prompt_transform=prompt_transform,
    )
    return dataset


class SketchyDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        photo_transform=transforms.ToTensor(),
        sketch_transform=transforms.ToTensor(),
        prompt_transform=lambda x: x,
    ):
        super(SketchyDataset, self).__init__()

        photo_data_root = Path(data_root, split, "photo")
        sketch_data_root = Path(data_root, split, "sketch")

        photo_paths, sketch_paths, photo_infos = {}, {}, []
        classes = sorted([it.name for it in photo_data_root.iterdir()])
        for classname in classes:
            photo_paths[classname] = list(photo_data_root.joinpath(classname).glob("*.jpg"))
            sketch_paths[classname] = list(sketch_data_root.joinpath(classname).glob("*.jpg"))

        for classname, fpaths in photo_paths.items():
            photo_infos.extend([{"class": classname, "fpath": it} for it in fpaths])

        self.classes = classes
        self.sketch_paths = sketch_paths
        self.photo_infos = photo_infos

        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform
        self.prompt_transform = prompt_transform

    def __getitem__(self, index):
        info = self.photo_infos[index]
        classname, photo_path = info["class"], info["fpath"]
        sketch_path = random.choice(self.sketch_paths[classname])

        photo = self.photo_transform(Image.open(photo_path))
        sketch = self.sketch_transform(Image.open(sketch_path))
        prompt = self.prompt_transform(classname)

        return {"photos": photo, "sketches": sketch, "prompts": prompt}

    def __len__(self):
        return len(self.photo_infos)
