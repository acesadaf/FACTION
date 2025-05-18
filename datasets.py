import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate



import matplotlib.pyplot as plt
from torchvision.utils import save_image
import json
import pandas as pd
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "Debug28",
    "Debug224",
    "ColoredMNIST",
    "RotatedMNIST",
    "RotatedColoredMNIST",
    "FairFace",
    "CelebA",
    "FFHQ",
    "NYSF"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 8001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams=None):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        original_dataset_tr = MNIST(root, train=True, download=True, transform=transform)
        original_dataset_te = MNIST(root, train=False, download=True, transform=transform)

        data = ConcatDataset([original_dataset_tr, original_dataset_te])
        original_images = torch.cat([img for img, _ in data])
        original_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in data])

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes



class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45']

    def __init__(self, root, test_envs, hparams=None):
        super(ColoredMNIST, self).__init__(root, [0, 15, 30, 45],
                                           self.rotate_dataset, (2, 32, 32,), 2)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 32, 32)

        environement_color = -1
        if angle == 0:
            environement_color = 0.9
        elif angle == 15:
            environement_color = 0.8
        elif angle == 30:
            environement_color = 0.7
        elif angle == 45:
            environement_color = 0.6


        for i in range(len(images)):
            x[i] = rotation(images[i])

        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        # # Flip label with probability 0.25
        # labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environement_color, len(labels)))

        x=torch.squeeze(x,dim=1)
        x = torch.stack([x, x], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        x[torch.tensor(range(len(x))), (1 - colors).long(), :, :] *= 0

        x =  x.float()  # .div_(255.0)
        y = labels.view(-1).long()



        return TensorDataset(x, y, colors)


    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()



class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45']

    def __init__(self, root, test_envs, hparams=None):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class RotatedColoredMNIST(MultipleEnvironmentMNIST):

    def __init__(self, root, test_envs, hparams):
        self.sample_pos = True
        self.ENVIRONMENTS = [0, 15, 30, 45]
        self.test_env = test_envs

        super(RotatedColoredMNIST, self).__init__(root, self.ENVIRONMENTS, self.rotate_dataset, (1, 28, 28,), 2)

    def rotate_dataset(self, images, labels, angle):
        angles = [0, 15, 30, 45]

        images = images.unsqueeze(1)

        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,))),
            transforms.ToTensor()
        ])
        x = torch.zeros(len(images), 1, 32, 32)

        # Set environment-specific color-flipping probability
        environement_color = -1
        if angle == 0:
            environement_color = 0.9
        elif angle == 15:
            environement_color = 0.8
        elif angle == 30:
            environement_color = 0.7
        elif angle == 45:
            environement_color = 0.6

        for i in range(len(images)):
            x[i] = rotation(images[i])

        labels = (labels < 5).float()
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environement_color, len(labels)))

        x = torch.squeeze(x, dim=1)
        x = torch.stack([x, x, x], dim=1)
        x[torch.arange(len(x)), (1 - colors).long(), :, :] *= 0
        x = x.float()

        y = labels.view(-1).long()
        return TensorDataset(x, y, colors)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()



class SensitiveImageFolder(ImageFolder):
    def __init__(self,
        root,
        transform = None,
        ):
        super().__init__(root,
                         transform,
                         )
        path_list = root.split('/')
        path_list.pop()
        dict_path = "/".join(path_list)
        with open(dict_path + '/data.json') as f:
            self.dict = json.load(f)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]

        z = self.dict[file_name][2]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, z

    def __len__(self):
        return  len(self.samples)
    
class TypeTwoSensitiveImageFolder(ImageFolder):
    def __init__(self,
        root,
        transform = None,
        ):
        super().__init__(root,
                         transform,
                         )
        # path_list = root.split('/')
        # path_list.pop()
        # dict_path = "/".join(path_list)
        # with open(dict_path + '/data.json') as f:
        #     self.dict = json.load(f)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]
        age, gender, race, date_time = file_name.split('_')
        race = int(race)
        if race == 0:
            z = 1 
        else:
            z = 0 
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, z

    def __len__(self):
        return  len(self.samples)


class TypeThreeSensitiveImageFolder(ImageFolder):
    def __init__(self,
        root,
        transform = None,
        ):
        super().__init__(root,
                         transform,
                         )
        path_list = root.split('/')

        path_list.pop() #get out of domain folder
        path_list.pop() #get to CelebA root folder (structure is 4 domains and one data.json, different from FairFace)
        dict_path = "/".join(path_list)
        with open(dict_path + '/CelebAdata.json') as f:
            self.dict = json.load(f)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]
        attributes = self.dict[file_name]
        gender = attributes["Male"]

        if gender == 1:
            z = 1 
        else:
            z = 0
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, z

    def __len__(self):
        return  len(self.samples)




class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams=None):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        augment_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = SensitiveImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)
            #print(self.datasets)
            #import pdb; pdb.set_trace()

        self.input_shape = (3, 224, 224,) 
        self.num_classes = len(self.datasets[-1].classes)


class TypeTwoMultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams=None):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        augment_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = TypeTwoSensitiveImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class TypeThreeMultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams=None):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        augment_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = TypeThreeSensitiveImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class FFHQSensitiveImageFolder(ImageFolder):
    def __init__(self,
        root,
        transform = None,
        ):
        super().__init__(root,
                         transform,
                         )
        # path_list = root.split('/')
        # path_list.pop()
        # dict_path = "/".join(path_list)
        # with open(dict_path + '/data.json') as f:
        #     self.dict = json.load(f)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]
        id, gender_and_ext = file_name.split('_')
        gender = gender_and_ext[0]
        #import pdb; pdb.set_trace()
        gender = int(gender)
        z = gender

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, z

    def __len__(self):
        return  len(self.samples)



class FFHQMultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        augment_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = FFHQSensitiveImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class FairFace(MultipleEnvironmentImageFolder):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['0', '1', '2', '3', '4', '5', '6']
    def __init__(self, root=None, test_envs=None,hparams=None):
        self.dir = os.path.join("Path to dataset")
        super().__init__(self.dir, test_envs, False, hparams)
        #data_aug = False

class NYSF(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get one row of data
        row = self.dataframe.iloc[idx]

        label = torch.tensor(1 if row['FRISKED_FLAG'] == "Y" else 0, dtype = torch.float32)

        row = row.drop('FRISKED_FLAG')
        
        z = torch.tensor(1 if row['SUSPECT_RACE_DESCRIPTION'] == "BLACK" else 0, dtype = torch.float32)
        row = row.drop('SUSPECT_RACE_DESCRIPTION')


        row = row.values

        object_array = row
        bool_indices = np.where(object_array == True)[0]  # Find indices of True values
        int_indices = np.where(np.logical_not(np.isin(object_array, [True, False])))[0]  # Find indices of non-boolean values

        object_array[bool_indices] = 1  # Set True to 1
        object_array[np.where(object_array == False)[0]] = 0 
        float_array = object_array.astype(np.float32)

        features = torch.tensor(float_array, dtype=torch.float32)

        label = label.type(torch.LongTensor)
        z = z.type(torch.LongTensor)
        return features, label, z


class CelebA(TypeThreeMultipleEnvironmentImageFolder):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['0', '1', '2', '3',] 
    def __init__(self, root=None, test_envs=None,hparams=None):
        self.dir = os.path.join("Path to Dataset")
        super().__init__(self.dir, test_envs, False, hparams)
        #data_aug = False



class FFHQ(FFHQMultipleEnvironmentImageFolder):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['0', '1', '2', '3',]
    def __init__(self, root=None, test_envs=None, hparams=None):
        self.dir = os.path.join("path to dataset")
        hparams = hparams or {'data_augmentation': False}
        super().__init__(self.dir, test_envs, hparams.get('data_augmentation', False), hparams)