import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from visualization import show_img

def load_data(path, batchSize, train_ratio=0.8, test_ratio=0.2):
    
    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]
  #  dataset_for_visualization = datasets.ImageFolder(path, transform=[transforms.ToTensor()])
  #  loader_for_visualization = torch.utils.data.DataLoader(dataset_for_visualization,
   #                                          batch_size=batchSize,
   #                                          pin_memory=True)
    
    #for i,(item, _) in enumerate(loader_for_visualization):
     #   if i == 0:
      #      show_img(item[0:9])
      #  else:
      #      break
    
    dataset = datasets.ImageFolder(path, transform=transforms.Compose(tra))
    
    n_train = int(train_ratio * len(dataset))
    n_test = int(test_ratio * len(dataset))
    
    data_train, data_test = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    trainloader = torch.utils.data.DataLoader(data_train,
                                             batch_size=batchSize,
                                             pin_memory=True)
    
    testloader = torch.utils.data.DataLoader(data_test,
                                             batch_size=batchSize,
                                             pin_memory=True)
    return trainloader, data_train, testloader, data_test

    