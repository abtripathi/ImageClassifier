import torch
from torchvision import datasets,transforms
import os.path

# loader function

def get_loaders(data_dir:str='flowers'):
    
   
        data_dir = data_dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),normalize])

        test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
        validation_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),normalize])

        #  Load the datasets with ImageFolder
        train_image_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
        test_image_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)
        validation_image_datasets = datasets.ImageFolder(valid_dir,transform=validation_transforms)

        #  Using the image datasets and the trainforms, define the dataloaders
        traindataloaders = torch.utils.data.DataLoader(train_image_datasets,batch_size=32,shuffle=True)
        testdataloaders = torch.utils.data.DataLoader(test_image_datasets,batch_size=32,shuffle=True)
        validationdataloaders = torch.utils.data.DataLoader(validation_image_datasets,batch_size=32,shuffle=True)
           
        return traindataloaders,validationdataloaders,test_image_datasets.class_to_idx

  
  
   
    