import numpy as np
from collections import OrderedDict

'''
SMOOTH = 1e-6

def mIoU(outputs, labels):

    N = max(labels.max(), outputs.max()) + 1
    N = int(N.item())
    pred_cnt = torch.bincount(outputs.flatten().long(), minlength=N)
    act_cnt = torch.bincount(labels.flatten().long(), minlength=N)

    cm = confusion_matrix(outputs, labels)

    Nr = torch.diag(cm)
    Dr = pred_cnt + act_cnt - Nr # A ⋃ B

    iou = Nr/Dr
    miou = torch.mean(iou)    
    return iou, miou

def pixelAcc(outputs, labels):

    cm = confusion_matrix(outputs, labels)
    Nr = torch.diag(cm).sum()

    acc = Nr/cm.sum()
    return acc

def confusion_matrix(outputs, labels):
    N = max(labels.max(), outputs.max()) + 1
    N = int(N.item())

    y_true = labels.flatten().long()
    y_pred = outputs.flatten().long()
    y = N * y_true + y_pred
    y = torch.bincount(y, minlength=N*N)
    if len(y) < N * N:
        y = torch.cat(y, torch.zeros(N * N - len(y), dtype=torch.long))
    y = y.reshape(N, N)
    return y

def get_all_eval_mats(outputs, labels):
    #outputs [H, W]
    #labels [H, W]

    N = max(labels.max(), outputs.max()) + 1
    N = int(N.item())

    pred_cnt = torch.bincount(outputs.flatten().long(), minlength=N)
    act_cnt = torch.bincount(labels.flatten().long(), minlength=N)

    y_true = labels.flatten().long()
    y_pred = outputs.flatten().long()
    y = N * y_true + y_pred
    y = torch.bincount(y, minlength=N*N)
    if len(y) < N * N:
        y = torch.cat(y, torch.zeros(N * N - len(y), dtype=torch.long))
    
    # Confusion Matrix
    cm = y.reshape(N, N)

    Nr = torch.diag(cm)
    Dr = pred_cnt + act_cnt - Nr # A ⋃ B

    # IOU and mIoU
    iou = Nr/Dr
    miou = torch.mean(iou)  

    # pixel Acc
    PA = Nr.sum()/cm.sum()

    # Precesion 
    precesion = cm[0][1]/(cm[0][1] + cm[0][0])

    # Recall
    recall = cm[0][1]/(cm[0][1] + cm[1][1])

    mats = {
        'miou': miou,
        'pa': PA,
        'precesion': precesion,
        'recall': recall,
    }
    return mats
'''


class BCEvaluator:
    """
    Evaluate semantic segmentation
    """
    def __init__(self, num_classes=2):

        self._N = num_classes
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def update(self, pred, gt):
        pred = pred.astype(np.int)
        gt = gt.astype(np.int)
        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
        ).reshape(self._N, self._N)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """

        acc = np.zeros(self._N, dtype=np.float)
        iou = np.zeros(self._N, dtype=np.float)
        tp = self._conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        results = OrderedDict({"bin_seg": res})

        return results