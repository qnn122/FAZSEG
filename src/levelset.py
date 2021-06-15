import numpy as np
import scipy.ndimage.filters as filters
import cv2
import matplotlib.pyplot as plt
import torch
from skimage.color import rgb2gray

# Prepare hyper-parameter for levelset
timestep = 1        # time step
mu = 0.2/timestep   # coefficient of the distance regularization term R(phi)
iter_inner = 2
iter_outer = 5
lmda = 5            # coefficient of the weighted length term L(phi)
alfa = -3           # coefficient of the weighted area term A(phi)
epsilon = 1.5       # parameter that specifies the width of the DiracDel
sigma = 0.8         # scale parameter in Gaussian kernel
potentialFunction = 'double-well'


def levelset(image, mask, iter_outer, isplot=False):
    '''Apply levelset on a given image
    Parameters:
    -----------
    image : can be tensor with shape=(c, H, W) or a numpy array
        original image
    mask : numpy.ndarray with shape=(H,W)
        2D array of the ROIs to be processed
    iter_outer : int
        number of iteration of applying levelset
    isplot: bool
        whether to plot the iterations
    
    Returns:
    --------
    phi : numpy array with shape=[H,W]
        a gray scale image of ROIs

    See more: segmensingle.py
    ---------
    '''
    tol = 0; #tolerance
    T = 0.1; # threshold to stop the iteration, TODO: more examination
    
    # Prepare input image
    if type(image) == torch.Tensor:
        image_temp = image.permute(1,2,0).numpy()
    else:
        image_temp = image.copy()
        
    #image_temp =  cv2.cvtColor(image_temp, cv2.COLOR_RGB2GRAY)
    levelset_img = (rgb2gray(image_temp)*255).astype(np.float32)
    #levelset_img = levelset_img.astype(np.uint8)
    #levelset_img = levelset_img.astype(np.float32)
    
    # Smoothing image and Calculate edge indicator function
    img_smooth = filters.gaussian_filter(levelset_img, sigma)  # smooth image by Gaussian convolution, input image pixal in range [0, 255]
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1+f)    # edge indicator function.

    # Start level set evolution
    c0 = 2  # TODO: more examination
    phi = mask.copy()       # range [0,1], float
    phi = phi*c0*2 - c0     # phi before levelset must fall from [-2, 2.x] for `double-well` method
    #phi = 1 - phi
    
    if isplot:
        plt.figure(figsize=(30,15))
    
    delta = np.zeros(iter_outer)

    for n in range(iter_outer):
        # Upate phi
        new_phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
        
        # Measure the difference between current phi and the last phi
        delta_temp = new_phi - phi
        delta[n] = delta_temp.sum()
        
        if n > 3:
            tol = ((delta[n] - delta[n-1])/(256**2))*100 # unit: %
        
        # update delta
        phi = new_phi
        
        # Show progress
        if isplot:
            plt.subplot(1, iter_outer, n+1)
            plt.imshow(phi)
        
        if n > 3 and np.abs(tol) < T:
            break

    return phi, delta
    
def del2(M):
    dx = 1
    dy = 1
    rows, cols = M.shape
    dx = dx * np.ones((1, cols - 1))
    dy = dy * np.ones((rows - 1, 1))

    mr, mc = M.shape
    D = np.zeros((mr, mc))

    if (mr >= 3):
        ## x direction
        ## left and right boundary
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:, 0] * dx[:, 1])
        D[:, mc - 1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc - 1]) \
                       / (dx[:, mc - 3] * dx[:, mc - 2])

        ## interior points
        tmp1 = D[:, 1:mc - 1]
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = np.kron(dx[:, 0:mc - 2] * dx[:, 1:mc - 1], np.ones((mr, 1)))
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3

    if (mr >= 3):
        ## y direction
        ## top and bottom boundary
        D[0, :] = D[0, :] + \
                  (M[0, :] - 2 * M[1, :] + M[2, :]) / (dy[0, :] * dy[1, :])

        D[mr - 1, :] = D[mr - 1, :] \
                       + (M[mr - 3, :] - 2 * M[mr - 2, :] + M[mr - 1, :]) \
                         / (dy[mr - 3, :] * dx[:, mr - 2])

        ## interior points
        tmp1 = D[1:mr - 1, :]
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr - 2, :])
        tmp3 = np.kron(dy[0:mr - 2, :] * dy[1:mr - 1, :], np.ones((1, mc)))
        D[1:mr - 1, :] = tmp1 + tmp2 / tmp3

    return D / 4


def drlse_edge(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters, potentialFunction):  # Updated Level Set Function
    """Main function of LevelSet
    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param timestep: time step
    :param iters: number of iterations
    :param potentialFunction: choice of potential function in distance regularization term.
%              As mentioned in the above paper, two choices are provided: potentialFunction='single-well' or
%              potentialFunction='double-well', which correspond to the potential functions p1 (single-well)
%              and p2 (double-well), respectively.
    """
    phi = phi_0.copy()
    [vy, vx] = np.gradient(g)
    for k in range(iters):
        phi = NeumannBoundCond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        smallNumber = 1e-10
        Nx = phi_x / (s + smallNumber)  # add a small positive number to avoid division by zero
        Ny = phi_y / (s + smallNumber)
        curvature = div(Nx, Ny)
        if potentialFunction == 'single-well':
            distRegTerm = filters.laplace(phi, mode='wrap') - curvature  # compute distance regularization term in equation (13) with the single-well potential p1.
        elif potentialFunction == 'double-well':
            distRegTerm = distReg_p2(phi)  # compute the distance regularization term in eqaution (13) with the double-well potential p2.
        else:
            print('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
        diracPhi = Dirac(phi, epsilon)
        areaTerm = diracPhi * g  # balloon/pressure force
        edgeTerm = diracPhi * (vx * Nx + vy * Ny) + diracPhi * g * curvature
        phi = phi + timestep * (mu * distRegTerm + lmda * edgeTerm + alfa * areaTerm)
    return phi


def distReg_p2(phi):
    """
        compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)  # compute first order derivative of the double-well potential p2 in equation (16)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))  # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + filters.laplace(phi, mode='wrap')


def div(nx, ny):
    [junk, nxx] = np.gradient(nx)
    [nyy, junk] = np.gradient(ny)
    return nxx + nyy


def Dirac(x, sigma):
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def NeumannBoundCond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    [ny, nx] = f.shape
    g = f.copy()

    g[0, 0] = g[2, 2]
    g[0, nx-1] = g[2, nx-3]
    g[ny-1, 0] = g[ny-3, 2]
    g[ny-1, nx-1] = g[ny-3, nx-3]

    g[0, 1:-1] = g[2, 1:-1]
    g[ny-1, 1:-1] = g[ny-3, 1:-1]

    g[1:-1, 0] = g[1:-1, 2]
    g[1:-1, nx-1] = g[1:-1, nx-3]

    return g