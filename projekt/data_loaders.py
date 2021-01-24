import os
import PIL
import urllib
import tarfile
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DownloadableABVisionDataset(datasets.vision.VisionDataset):
    def __init__(self, root, folder, archive_name, archive_ext, archive_url, sizeA, sizeB,
            download=True, transform=None):
        super(DownloadableABVisionDataset, self).__init__(root=root, transform=transform)
        self.sizeA = sizeA
        self.sizeB = sizeB

        if download:
            if not os.path.isdir(root):
                os.mkdir(root)
            path = os.path.join(root, archive_name + "." + archive_ext)
            if not os.path.isfile(path):
                urllib.request.urlretrieve(archive_url, path)
            with tarfile.open(path) as tar:
                tar.extractall(root)

        self.files = []
        for path, dirs, files in os.walk(os.path.join(self.root, archive_name, folder)):
            self.files += map(lambda x: os.path.join(path, x), files)

    def __getitem__(self, index):
        sample = PIL.Image.open(self.files[index])
        wA, hA = self.sizeA
        wB, hB = self.sizeB
        imgA = sample.crop((0, 0, wA, hA))
        imgB = sample.crop((wA, 0, wA + wB, hB))
        if self.transform:
            return self.transform(imgA, imgB)
        return imgA, imgB

    def __len__(self):
        return len(self.files)

class Facades(DownloadableABVisionDataset):
    def __init__(self, root, folder, download=True, transform=None):
        super(Facades, self).__init__(root=root, transform=transform, download=download, folder=folder,
                sizeA=(256,256), sizeB=(256,256), archive_name="facades", archive_ext="tar.gz",
                archive_url="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz")

class Maps(DownloadableABVisionDataset):
    def __init__(self, root, folder, download=True, transform=None):
        super(Maps, self).__init__(root=root, transform=transform, download=download, folder=folder,
                sizeA=(600,600), sizeB=(600,600), archive_name="maps", archive_ext="tar.gz",
                archive_url="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz")

class Cityscapes(DownloadableABVisionDataset):
    def __init__(self, root, folder, download=True, transform=None):
        super(Cityscapes, self).__init__(root=root, transform=transform, download=download, folder=folder,
                sizeA=(256,256), sizeB=(256,256), archive_name="cityscapes", archive_ext="tar.gz",
                archive_url="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz")
