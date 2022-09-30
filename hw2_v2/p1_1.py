from binascii import b2a_base64
import cv2
from skimage.color import rgb2gray
from scipy import ndimage
import numpy as np

# Do Not Modify
import nbimporter
from util import display_filter_responses
import numpy as np
import multiprocess
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import random
import cv2
import math

from skimage import io
#-------------------------------------------------------------------------


def plot_harris_points(image,points):
    fig = plt.figure(1)
    for x,y in zip(points[0],points[1]):
        plt.plot(y,x,marker='v')
    plt.imshow(image)
    plt.show()

def get_harris_corners(image, alpha, k = 0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (2, alpha) that contains interest points
    '''
    
    '''
    HINTS:
    (1) Visualize and Compare results with cv2.cornerHarris() for debug (DO NOT SUBMIT cv2's implementation)
    '''
    # ----- TODO -----
    
    ######### Actual Harris #########
    bw_img = rgb2gray(image)
    '''
    HINTS:
    1.> For derivative images we can use cv2.Sobel filter of 3x3 kernel size
    2.> Multiply the derivatives to get Ix * Ix, Ix * Iy, etc.
    '''
    '''
    HINTS:
    1.> Think of R = det - trace * k
    2.> We can use ndimage.convolve
    3.> sort (argsort) the values and pick the alpha larges ones
    3.> points_of_interest should have this structure [[x1,x2,x3...],[y1,y2,y3...]] (2,alpha)
        where x_i is across H and y_i is across W
    '''

    # Compute the image derivatives
    kernel_size = 3
    Ix = cv2.Sobel(bw_img, cv2.CV_64F, 1, 0, ksize = kernel_size)
    Iy = cv2.Sobel(bw_img, cv2.CV_64F, 0, 1, ksize = kernel_size)

    # Create an average filter to efficiently calculate the the statistics for each patch
    avg_kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    # Compute the covariance matrix for the second moment at each patch
    IxIx = ndimage.convolve(Ix ** 2, avg_kernel)
    IyIy = ndimage.convolve(Iy ** 2, avg_kernel)
    IxIy = ndimage.convolve(Ix * Iy, avg_kernel)

    # Compute the determinant and the trace
    H_det = IxIx * IyIy - IxIy ** 2
    H_tr  = IxIx + IyIy

    # Compute R which will look at the eigenvalues,
    # If the eigen values are closer to 0, that means that the area of the ellipse has not changed much (close to identity) and is relatively flat
    # If one eigenvalue is bigger, we likely have have an edge, stretching the area in one direction
    # If both eigenvalues are big, we have a dot/corner which will stretch the ellipse among the major and minor axes
    # Changes the surface to threshold to create a curve R > T : R = lambda1 * lambda2 - k * (lambda1 + lambda2) **2
    R = H_det - k * (H_tr ** 2)

    # Sort R and get the "alpha best" corner points. 
    best_corner_points_mine = np.array(np.unravel_index(np.argsort(R, None)[-alpha:], R.shape))
    # We will have y, x after argsort


    # Make sure no x or y values go out of bounds
    assert best_corner_points_mine[0].max() < image.shape[0]
    assert best_corner_points_mine[1].max() < image.shape[1]

    # Best corner points for openCV algo
    #R_true = cv2.cornerHarris(bw_img.astype(dtype="float32"), 2, 3, k)
    #best_corner_points_true = np.array(np.unravel_index(np.argsort(R_true, None)[-alpha:], R.shape, order="C"))
    
    ######### Actual Harris #########
    return best_corner_points_mine

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''
    # Normalize the image [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    assert image.max() <= 1.0 and image.min() >= 0.0

    # Grayscale image
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    # Grayscale image with 3 channels
    if(len(image.shape) == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]

    image = skimage.color.rgb2lab(image)

    scales = [1, 2, 4, 8, 8 * (2**.5)]
    filter_responses = []
    '''
    HINTS: 
    1.> Iterate through the scales (5) which can be 1, 2, 4, 8, 8$\sqrt{2}$
    2.> use scipy.ndimage.gaussian_* to create filters
    3.> Iterate over each of the three channels independently
    4.> stack the filters together to (H, W,3F) dim
    '''
    # ----- TODO -----
    
    # Lits of filters as lambda functions
    filters = []
    # Gaussian filter
    filters.append(lambda img, scale: scipy.ndimage.gaussian_filter(img, scale, mode = "reflect"))
    # Laplacian of gaussian filter
    filters.append(lambda img, scale: scipy.ndimage.gaussian_laplace(img, scale, mode = "reflect"))
    # DoG filter in the x direction
    filters.append(lambda img, scale: scipy.ndimage.gaussian_filter1d(img, scale, axis = 1, order = 1, mode = "reflect"))
    # DoG filter in the y direction
    filters.append(lambda img, scale: scipy.ndimage.gaussian_filter1d(img, scale, axis = 0, order = 1, mode = "reflect"))

    # YOUR CODE HERE
    for scale in scales:
        for filter in filters:
            for channel_idx in range(image.shape[2]):
                # Extract the channel and store the filter response for the filter at a given scale
                img_channel = image[:, :, channel_idx]
                filter_responses.append(filter(img_channel, scale))

    return np.stack(filter_responses[:], axis = -1)

def compute_dictionary_one_image(args):
    '''
    Extracts samples of the dictionary entries from an image. Use the the 
    harris corner detector implmented from previous question to extract 
    the point of interests. This should be a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''
    i, alpha, image_path = args
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    f_name = 'tmp/%05d.npy' % i
    
    # Read the data and get the corner points
    img = cv2.imread(image_path)
    corner_points = get_harris_corners(img, alpha) # list of tuples of (x, y)
    
    # Get the response at the corner points
    img_filter_responses    = extract_filter_responses(img)
    corner_filter_responses = np.array([img_filter_responses[point[0], point[1]] for point in corner_points.T])

    # Write the corner points to the file
    np.save(f_name, corner_filter_responses)