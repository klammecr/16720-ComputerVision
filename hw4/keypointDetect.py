import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):    
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255

    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    for i in range(len(levels) - 1):
        DoG_pyramid.append(gaussian_pyramid[:, :, i+1] - gaussian_pyramid[:, :, i])

    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)           

    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''

    principal_curvature = None
    gxx = []
    gyy = []
    gxy = []
    gyx = []   
             
    for l in range(DoG_pyramid.shape[2]):
        # Computing 1st order derivatives
        gx = cv2.Sobel(DoG_pyramid[:,:,l],cv2.CV_64F,1,0,ksize=3,borderType=cv2.BORDER_CONSTANT)
        gy = cv2.Sobel(DoG_pyramid[:,:,l],cv2.CV_64F,0,1,ksize=3,borderType=cv2.BORDER_CONSTANT)
  
        # Computing 2nd order derivatives
        gxx.append(cv2.Sobel(gx,cv2.CV_64F,1,0,ksize=3,borderType=cv2.BORDER_CONSTANT))
        gxy.append(cv2.Sobel(gx,cv2.CV_64F,0,1,ksize=3,borderType=cv2.BORDER_CONSTANT))
        gyx.append(cv2.Sobel(gy,cv2.CV_64F,1,0,ksize=3,borderType=cv2.BORDER_CONSTANT))
        gyy.append(cv2.Sobel(gy,cv2.CV_64F,0,1,ksize=3,borderType=cv2.BORDER_CONSTANT))

    gxx = np.stack(gxx, axis=-1)
    gxy = np.stack(gxy, axis=-1)     
    gyx = np.stack(gyx, axis=-1)     
    gyy = np.stack(gyy, axis=-1) 
  
    principal_curvature = np.divide(np.square(np.add(gxx, gyy)), (np.multiply(gxx, gyy) - np.multiply(gxy, gyx)))



    ##################
    # TO DO ...
    # Compute principal curvature here

    return principal_curvature

def isInbound(y, x, l, p):
    return (0 <= p[0] < y) & (0 <= p[1] < x) & ((0 <= p[2] < l))


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    h, w, l = principal_curvature.shape
    principal_curvature_clamped = (DoG_pyramid > th_contrast) & (principal_curvature < th_r)

    locs = np.where(principal_curvature_clamped == True)
    locsDoG_tmp = np.zeros([len(locs[0]), 3])

    locsDoG_tmp[:,0] = locs[0]
    locsDoG_tmp[:,1] = locs[1]
    locsDoG_tmp[:,2] = locs[2]
    locsDoG_tmp = locsDoG_tmp.astype(int)
    direction = [-1, 0, 1]
    for i in range(locsDoG_tmp.shape[0]):
        p = locsDoG_tmp[i, :]
        neighbors = []
        for dx in range(3):
            for dy in range(3):
                p_tmp = [p[0] + direction[dy], p[1] + direction[dx], p[2]]
                if isInbound(h, w, l, p_tmp):
                    neighbors.append(DoG_pyramid[p_tmp[0], p_tmp[1], p_tmp[2]])

        p_tmp = [p[0], p[1], p[2] - 1]
        if isInbound(h, w, l, p_tmp):
            neighbors.append(DoG_pyramid[p_tmp[0], p_tmp[1], p_tmp[2]])
        p_tmp = [p[0], p[1], p[2] + 1]
        if isInbound(h, w, l, p_tmp):
            neighbors.append(DoG_pyramid[p_tmp[0], p_tmp[1], p_tmp[2]])

        if (DoG_pyramid[p[0], p[1], p[2]] == min(neighbors)) | (DoG_pyramid[p[0], p[1], p[2]] == max(neighbors)):
            if locsDoG is None:
                locsDoG = p.reshape((1,3))
            else:
                locsDoG = np.concatenate((locsDoG,p.reshape((1,3))), axis=0)

    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    levels = [-1,0,1,2,3,4]
    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)  
    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')






    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # # test DoG pyramid

    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    for i in range(locsDoG.shape[0]):
        cv2.circle(im,(locsDoG[i, 1],locsDoG[i, 0]), 1, (0,255,0), -1)
    cv2.imshow('Pyramid of image', im)

    


    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()