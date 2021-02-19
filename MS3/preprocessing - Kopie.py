import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from utils import ReassignedDataset
import torchvision.transforms as transforms
import faiss

def preprocessing(model, features, n_components = 256):
    #pca, whitening
    _, ndim = features.shape
    npdata =  features.astype('float32')
    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, n_components, eigen_power=-0.5)
    #l2 normalization
    mat.train(npdata)
    assert mat.is_trained
    pca_reduced = mat.apply_py(npdata)

    #print(pca_reduced)
    row_sums = np.linalg.norm(pca_reduced, axis=1)
    f_normalized = features / row_sums[:, np.newaxis]
    return f_normalized

def clustering(pre_data, k = 2):
    #random_cen = random(pre_data)
    kmeans = KMeans(n_clusters=k).fit(pre_data)
    clustered_ind = kmeans.fit_predict(pre_data)
    images_lists = [[] for i in range(k)]
    for i in range(pre_data.shape[0]):
        images_lists[clustered_ind[i]].append(i)

    return clustered_ind, images_lists

def compute_features(dataloader, model, N, batch, labels):
    
   
    # discard the label information in the dataloader
    for i, (input_tensor, label) in enumerate(dataloader):
        """
        if label not in labels and labels != []:
            continue
        else:
            print('found label')
        """
        input_var = torch.autograd.Variable(input_tensor.cuda())
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch: (i + 1) * batch] = aux
        else:
            # special treatment for final batch
            features[i * batch:] = aux
        
      
        
        if ((i % 50) == 0) and (i>0):
            print(f'{i} features computed')

    return features

def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)