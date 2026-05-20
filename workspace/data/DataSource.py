import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np

class SubtractMeanImage(object):
    def __init__(self, mean_image):
        self.mean_image = Image.fromarray(mean_image.astype('uint8'))
        if isinstance(self.mean_image, Image.Image):
            self.mean_image = np.array(self.mean_image)

    def __call__(self, img):
        img_array = np.array(img)
        mean_image_resized = np.array(Image.fromarray(self.mean_image).resize(img_array.shape[1::-1], Image.BILINEAR))
        result = img_array - mean_image_resized
        return Image.fromarray(np.clip(result, 0, 255).astype('uint8'))

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing

        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
        else:
            self.mean_image = self.generate_mean_image()

        if self.train:
            self.transform = T.Compose([
                T.Resize(self.resize),
                SubtractMeanImage(self.mean_image),
                T.RandomCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Resize(self.resize),
                SubtractMeanImage(self.mean_image),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image

        # Initialize mean_image
        mean_image = None

        # Iterate over all training images
        # Resize, Compute mean, etc...
        for img_path in self.images_path:
            img = Image.open(img_path).convert('RGB')
            img = T.Resize(self.resize)(img)
            if mean_image is None:
                mean_image = np.zeros_like(np.array(img), dtype=np.float64)
            mean_image += np.array(img, dtype=np.float64)

        mean_image /= len(self.images_path)

        # Store mean image
        np.save(self.mean_image_path, mean_image)

        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        data = self.transform(data)

        return data, img_pose

    def __len__(self):
        return len(self.images_path)