import numpy as np
from sklearn.metrics import f1_score
    
def scoring(gt, pred):
    #gt = gt.to("cpu").numpy()
    '''
    Calculate False Positive, False Negative, IoU between 
    ground truth mast and predicted mask
    :param gt: <numpy array> ground truth mask
    :param pred: <numbpy array> predicted mask
    :return: FP, FN, IoU
    '''
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    # Calculate false positive
    false_pos = np.sum(pred)- intersection
    FP = false_pos / np.sum(pred)
    FP = round(FP, 2)
    
    # Calculate false negative
    false_neg = np.sum(gt) - intersection
    FN = false_neg / np.sum(gt)
    FN = round(FN, 2)

    # Calculate IoU
    if union != 0:
        iou = intersection / union
        iou = round(iou, 2)
    return iou, FP, FN

def multi_label_f1(y_gt, y_pred):
    """ Calculate F1 for each class

    Parameters
    ----------
    y_gt: torch.Tensor
        groundtruth
    y_pred: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    f1_out = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = (y_pred.to("cpu").numpy() > 0.5) * 1.0
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        try:
            f1_out.append(f1_score(gt_np[:, i], pred_np[:, i]))
        except:
            f1_out.append(0)
    return f1_out