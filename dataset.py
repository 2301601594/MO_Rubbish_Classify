from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import glob

index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}


# Init Dataset
def init_dataset(data_path):
    paths = os.listdir(data_path)
    datas = []
    for path in paths:
        image_paths = glob.glob(os.path.join(data_path, path, "*.jpg"))
        for image_path in image_paths:
            datas.append((image_path, path))

    return datas


# Create Dataset
class RubbishDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(RubbishDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform if transform else transforms.ToTensor()
        self.datas = init_dataset(data_path)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        image_path, label = self.datas[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, index[label]

if __name__ == '__main__':
    val_path = './garbage_26x100/val'
    val_set = RubbishDataset(val_path, transform=transforms.ToTensor())
    print(len(val_set))
