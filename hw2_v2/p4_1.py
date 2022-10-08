import nbimporter
import numpy as np
import numpy.matlib as matlib
import scipy.ndimage
from skimage import io
import skimage.transform
import os,time
import util
import multiprocess
import threading
import queue
import torch
import torchvision

def preprocess_image(image, size = (224, 224, 3), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H,W,3)

    [output]
    * image_processed: torch.array of shape (3,H,W)
    '''

    # ----- TODO -----
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]
    '''
    HINTS:
    1.> Resize the image (look into skimage.transform.resize)
    2.> normalize the image
    3.> convert the image from numpy to torch
    '''
    resized_img = skimage.transform.resize(image, size)
    resized_img = resized_img.transpose((2, 0, 1))

    # Create mean 
    mean_1 = matlib.repmat(mean[0], resized_img.shape[1], resized_img.shape[2])
    mean_2 = matlib.repmat(mean[1], resized_img.shape[1], resized_img.shape[2])
    mean_3 = matlib.repmat(mean[2], resized_img.shape[1], resized_img.shape[2])
    mean   = np.stack([mean_1, mean_2, mean_3])
    # print(mean.shape)

    # Create STD
    std_1 = matlib.repmat(std[0], resized_img.shape[1], resized_img.shape[2])
    std_2 = matlib.repmat(std[1], resized_img.shape[1], resized_img.shape[2])
    std_3 = matlib.repmat(std[2], resized_img.shape[1], resized_img.shape[2])
    std   = np.stack([std_1, std_2, std_3])

    # Normalize and return the torch image
    norm_img = (resized_img - mean) / std
    return torch.from_numpy(norm_img)

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''
    i, image_path, vgg16 = args
    image = io.imread(image_path) / 255
    img_torch = preprocess_image(image)
    
    '''
    HINTS:
    1.> Think along the lines of evaluate_deep_extractor
    '''
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    return [i, vgg_feat_feat]

def build_recognition_system(vgg16, num_workers=4):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("./data/train_data.npz", allow_pickle=True)
    '''
    HINTS:
    1.> Similar approach as Q1.2.2 and Q3.1.1 (create an argument list and use multiprocessing)
    2.> Keep track of the order in which input is given to multiprocessing
    '''
    args = []

    # Parse out the training files
    train_files = train_data.get("files")
    if train_files is None:
        raise ValueError("No valid training files available :(")
    train_files = ["./data/" + str(file) for file in train_files]

    out  = np.zeros((len(train_files), 4096))

    # Gather arguments for multiprocessing
    for idx, train_sample in enumerate(train_files):
        args.append((idx, train_sample, vgg16))

    # Do the processing
    with multiprocess.Pool(num_workers) as p:
        # Run the function and save the result
        result = p.map(get_image_feature, args)

        # Save the results in the dictionary
        for r in result:
            out[r[0]] = r[1]
        
    # Use the dictionary to get the ordered features
    # ordered_features = np.array(list(out.values()))
    '''
    HINTS:
    1.> reorder the features to their correct place as input
    '''
    print("done", out.shape)
    labels = train_data.get("labels")
    np.savez('trained_system_deep.npz', features=out, labels=labels)

def helper_func(args):
    # Parse the args
    i, test_sample, vgg16, trained_features, labels = args

    # Calculate the test features
    test_features = get_image_feature((i, test_sample, vgg16))[1]

    # Calculate the distance between the two features
    dist = np.sum((np.array(test_features) - trained_features) ** 2, axis = 1)

    # The smallest distance is the most similar
    pred_label = labels[np.argmin(dist)]

    return [i, pred_label]


def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    '''
    HINTS:
    (1) Students can write helper functions (in this cell) to use multi-processing
    '''
    test_data = np.load("./data/test_data.npz", allow_pickle=True)

    trained_system = np.load("trained_system_deep.npz", allow_pickle=True)
    image_names = test_data['files']
    image_names = ["./data/" + str(file) for file in image_names]
    test_labels = test_data['labels']

    trained_features = trained_system['features']
    train_labels = trained_system['labels']

    print("Trained features shape: ", trained_features.shape)
    
    '''
    HINTS:
    1.> [Important] Can write a helper function in this cell of jupyter notebook for multiprocessing
    
    2.> Helper function will compute the vgg features for test image (get_image_feature) and find closest
        matching feature from trained_features.
    
    3.> Since trained feature is of shape (N,K) -> smartly repeat the test image feature N times (bring it to
        same shape as (N,K)). Then we can simply compute distance in a vectorized way.
    
    4.> Distance here can be sum over (a-b)**2
    
    5.> np.argmin over distance can give the closest point
    '''
    # Gather arguments for multiprocessing
    args = []
    pred_labels_dict = {}
    for idx, test_sample in enumerate(image_names):
        args.append((idx, test_sample, vgg16, trained_features, train_labels))

    # Do the processing
    with multiprocess.Pool(num_workers) as p:
        # Run the function and save the result
        res = p.map(helper_func, args)
        for r in res:
            pred_labels_dict[r[0]] = r[1]

    pred_labels = np.array(list(pred_labels_dict.values()))

    print("Predicted labels shape: ", pred_labels.shape)
    
    '''
    HINTS:
    1.> Compute the confusion matrix (8x8)
    '''
    num_classes = len(np.unique(test_labels))
    conf_mtx = np.zeros((num_classes, num_classes))
    for i in range(len(pred_labels)):
        conf_mtx[test_labels[i], pred_labels[i]] += 1

    accuracy = np.trace(conf_mtx) / np.sum(conf_mtx)
    
    np.save("./trained_conf_matrix.npy",conf_mtx)
    return conf_mtx, accuracy