import os
import PIL
import torch
import urllib
import tarfile
import imageio
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Facades(datasets.vision.VisionDataset):
    def __init__(self, root, folder, download=True, transform=None):
        super(Facades, self).__init__(root=root, transform=transform)

        if download:
            if not os.path.isdir(root):
                os.mkdir(root)
            path = os.path.join(root, "facades.tar.gz")
            if not os.path.isfile(path):
                urllib.request.urlretrieve("http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz", path)
            with tarfile.open(path) as tar:
                tar.extractall(root)

#        self.files = {"train":[], "test":[], "val":[]}
#        for key in ["train", "test", "val"]:

        #if folder not in ["train", "test", "val"]:
        #    raise NotFoundError

        self.files = []
        for path, dirs, files in os.walk(os.path.join(self.root, "facades", folder)):
            self.files += map(lambda x: os.path.join(path, x), files)

    def __getitem__(self, index):
        sample = PIL.Image.open(self.files[index])
        # split label and image before applying transform
        img = sample.crop((0, 0, 256, 256))
        mask = sample.crop((256, 0, 512, 256))
        if self.transform:
            return self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.files)

# for testing
if __name__ == "__main__":
    example_path = "datasets\\facades\\test\\1.jpg"
    example = PIL.Image.open(example_path)

    t = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    facades = Facades(root="datasets", folder="test", transform=t)
    loader = torch.utils.data.DataLoader(facades, batch_size=16, shuffle=True, num_workers=0,
            drop_last=True)

    data = iter(loader)

    img, mask = data.next()
