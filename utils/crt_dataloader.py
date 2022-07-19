import torch
from torch.utils.data import Dataset, Subset, DataLoader

class ComprehensiveRobustnessDataset(Dataset):
    
    def __init__(self, imagenet_data, edsr_data, cae_data):
        # 45 epochs -> effectively 45 * 2 = 90 epochs
        # In each epoch, 2 * len(imagenet_data) images are trained once.
        self.imagenet_data = torch.utils.data.ConcatDataset([imagenet_data])
        self.deepaug_data = torch.utils.data.ConcatDataset([edsr_data, cae_data])

    def __getitem__(self, index):
        return self.imagenet_data[index], self.deepaug_data[index]

    def __len__(self):
        return min(len(self.imagenet_data), len(self.deepaug_data))

class TriComprehensiveRobustnessDataset(Dataset):
    
    def __init__(self, imagenet_data, edsr_data, cae_data, third_domain_data):
        # 30 epochs -> effectively 30 * 3 = 90 epochs
        # In each epoch, 3 * len(imagenet_data) images are trained once.
        self.imagenet_data = torch.utils.data.ConcatDataset([imagenet_data])
        self.deepaug_data = torch.utils.data.ConcatDataset([edsr_data, cae_data])
        self.third_domain_data = torch.utils.data.ConcatDataset([third_domain_data])
        assert len(self.imagenet_data) == len(self.third_domain_data)

    def __getitem__(self, index):
        return self.imagenet_data[index], self.deepaug_data[index], self.third_domain_data[index]

    def __len__(self):
        return min(len(self.imagenet_data), len(self.deepaug_data))

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from dataloaders.imagenet import imagenet_datasets, imagenet_deepaug_dataset, imagenet_texture_debias_dataset

    train_data, val_data = imagenet_datasets(data_dir='/ssd1/haotao/datasets/imagenet')
    edsr_data = imagenet_deepaug_dataset(data_dir='/ssd1/haotao/datasets/imagenet-DeepAug-EDSR')
    cae_data = imagenet_deepaug_dataset(data_dir='/ssd1/haotao/datasets/imagenet-DeepAug-CAE')
    texture_debias_data = imagenet_texture_debias_dataset(data_dir='/ssd1/haotao/datasets/imagenet')
    # combine datasets:
    # train_data = ComprehensiveRobustnessDataset(imagenet_data=train_data, edsr_data=edsr_data, cae_data=cae_data)
    train_data = TriComprehensiveRobustnessDataset(imagenet_data=train_data, edsr_data=edsr_data, cae_data=cae_data, third_domain_data=texture_debias_data)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=12)

    # for i, ((images, labels), (images_deepaug, labels_deepaug)) in enumerate(train_loader):
    #     print(i) 
    for i, ((images, labels), (images_deepaug, labels_deepaug), (images_texture_debias, labels_texture_debias)) in enumerate(train_loader):
        print(labels[0:5], labels_deepaug[0:5], labels_texture_debias[0:5]) 
        break
