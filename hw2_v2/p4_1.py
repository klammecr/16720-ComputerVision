import nbimporter
import numpy as np
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
import torchvision.transforms


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
    
    '''
    HINTS:
    1.> Think along the lines of evaluate_deep_extractor
    '''
    img_torch = torch.torch(image)
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    return [i, vgg_feat_feat]

def build_recognition_system(vgg16, num_workers=2):
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
    out  = {}

    # Parse out the training files
    train_files = train_data.get("files")
    if train_files is None:
        raise ValueError("No valid training files available :(")
    train_files = ["./data/" + str(file) for file in train_files]

    # Gather arguments for multiprocessing
    for idx, train_sample in enumerate(train_files):
        args.append((idx, train_sample, vgg16))

    # Do the processing
    with multiprocess.Pool(num_workers) as p:
        # Run the function and save the result
        r = p.map(get_image_feature, args)
        out[r[0]] = r[1]
        
    # Use the dictionary to get the ordered features
    ordered_features = np.array(list(out.values()))
    '''
    HINTS:
    1.> reorder the features to their correct place as input
    '''
    print("done", ordered_features.shape)
    labels = train_data.get("labels")
    np.savez('trained_system_deep.npz', features=ordered_features, labels=labels)

def helper_func(args):
    # Parse the args
    i, test_sample, vgg16, trained_features = args

    # Calculate the test features
    test_features = get_image_feature((i, test_sample, vgg16))

    # Calculate the distance between the two features
    dist = np.sum((test_features - trained_features) ** 2, axis = 1)

    # The smallest distance is the most similar
    pred_label = np.argmin(dist)

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
        args.append((idx, test_sample, vgg16, trained_features))

    # Do the processing
    with multiprocess.Pool(num_workers) as p:
        # Run the function and save the result
        r = p.map(get_image_feature, args)
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