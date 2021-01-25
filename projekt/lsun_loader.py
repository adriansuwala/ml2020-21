import os
import PIL
import torch
import imageio
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Phase 0: just make it work --> root is a path to unzipped data0

class HashedLSUN(datasets.vision.VisionDataset):
    def __init__(self, root, transform=None):
        super(HashedLSUN, self).__init__(root=root, transform=transform)

        self.root = os.path.join(self.root, "lsun", "bedroom")

        # just load all file paths into memory (~303k strings)
        self.files = []
        for path, dirs, files in os.walk(self.root):
            self.files += map(lambda x: os.path.join(path, x), files)

    def __getitem__(self, index):
        sample = PIL.Image.open(self.files[index])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.files)

# for testing
if __name__ == "__main__":
    example_path = "datasets/data0\\lsun\\bedroom\\0\\0\\0\\000038527b455eaccd15e623f2e229ecdbceba2b.jpg"
    example = PIL.Image.open(example_path)

    t = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    lsun = HashedLSUN(root="datasets/data0", transform=t)
    loader = torch.utils.data.DataLoader(lsun, batch_size=64, shuffle=True, num_workers=0,
            drop_last=True)

    data = iter(loader)

    img = data.next()
