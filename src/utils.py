import numpy as np

T = 0.6  # threshold to take binary, consider to be changed

def binarize_phi(phi):
    '''
    Convert messy mask after levelset to a binary mask
    :param phi: <numpy array [H, W]> masked after processed by levelset
    '''
    binary_phi = phi.copy()
    binary_phi = binary_phi - np.min(binary_phi)
    binary_phi = binary_phi / np.max(binary_phi)
    binary_phi = 1 - binary_phi
    binary_phi = (binary_phi > T) * 1.0
    return binary_phi