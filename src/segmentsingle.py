from torchvision import transforms
from PIL import Image
import numpy as np
from src.levelset import levelset
from src.dataset import FAZ_Preprocess


def segmentsingle(im, model, SCALE, isSaveoutput):
    '''Segment single image using UNet and Levelset
    Parameters:
    ----------
    im : numpy.ndarray. shape=(H,W)
            A gray-scae OCTA image, value ranging fron 0-255
    model: torch.model
            a pytorch preloaded model
    SCALE: int
            Size of resized image, depending on model
    isSaveoutput: bool
            Whether to save output to file 
    
    Returns:
    --------
    enhanced: ndarrray
            Enhanced OCT image
    mask: ndarray
            Mask after being processed with UNet
    phi: ndarray
            Mask afer being processed with Unet and Levelset
    delta: ndarray
            List of differences between 2 consecutive phi
    
    Examples:
    --------
    >>> import cv2
    >>> model = get_torchvision_model("Se_resnext50", True, 1, "focal")
    >>> state_dict = torch.load("models/Se_resnext50-920eef84.pth",  map_location=torch.device('cpu'))
    >>> state_dict = state_dict["state"]
    >>> model.load_state_dict(state_dict)
    >>> im = cv2.imread('./images/raw/1.png', cv2.IMREAD_GRAYSCALE) # ensure input image is 2D array and pixels range in [0, 255]
    >>> enhanced, mask, phi = segmentsingle(im, model, 256, False)
    '''
    # Transformation:
    tensor_transform = transforms.Compose([
        transforms.Resize((SCALE, SCALE)),
        transforms.ToTensor()
    ])

    # Filtered by Hessian-based filter
    enhanced = FAZ_Preprocess(im,[0.5,1, 1.5, 2, 2.5],1, 2).vesselness2d()
    enhanced_ts = tensor_transform(Image.fromarray(enhanced.astype(np.float32)*255).convert("RGB"))
    
    # Predict using Unet:
    model.eval()
    mask = enhanced_ts.unsqueeze(0)
    mask = model(mask)
    mask = (mask.to("cpu").detach().numpy() > 0.6)*1
    mask = mask.reshape((SCALE, SCALE))
    
    # Levelset
    phi, delta = levelset(enhanced_ts, mask, 20, isplot=False)

    return enhanced, mask, phi, delta