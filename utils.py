import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter1d
import pdb

def read_mapping_dict(file_path):
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def encode_content(content, nrows, ncols, actions_dict):
    '''encode a sequence of actions into a matrix form for the cnn model'''
    encoded_content = np.zeros([nrows, ncols])

    start = 0
    s = 0
    e = 0

    for i in range(len(content)):
        if content[i] != content[start]:
            frame_label = np.zeros((ncols))
            frame_label[actions_dict[content[start]]] = 1
            s = int(nrows*(1.0*start/len(content)))
            e = int(nrows*(1.0*i/len(content)))
            encoded_content[s:e]=frame_label
            start = i
    frame_label = np.zeros((ncols))
    frame_label[actions_dict[content[start]]] = 1
    encoded_content[e:]=frame_label

    return encoded_content

def get_label_length_seq(content):
    '''get the sequence of labels and length for a given frame-wise action labels'''
    label_seq = []
    length_seq = []
    start = 0
    for i in range(len(content)):
        if content[i] != content[start] :
            label_seq.append(content[start])
            length_seq.append(i-start)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content)-start)

    return label_seq, length_seq
'''
'Write the prediction output to a file
'''
def write_predictions(path, f_name, recognition):
#    recognition = recognition.detach().cpu().numpy()
    if not os.path.exists(path):
        os.makedirs(path)
    f_ptr = open(path+"/"+f_name+".recog","w")

    f_ptr.write("### Frame level recognition: ###\n")
    f_ptr.write(' '.join(recognition))

    f_ptr.close()

def label_to_action(content):
    tmp = []
    for i in range(len(content)):
        tmp.append()


def action_embedding_label(label, n_class):
    '''get labels, returns one-hot embedding'''
    seq_length = len(label)
    encoded_content = np.zeros([seq_length, n_class])

    start=1
    s=0
    for i in range(len(label)):
        if i == 0:
            #sos embedding
            frame_label = np.zeros((n_class))
            frame_label[int(label[0])] = 1
            encoded_content[i] = frame_label
            continue
        if label[i] != label[start]:
            frame_label = np.zeros((n_class))
            frame_label[int(label[start])] = 1
            encoded_content[start:i]=frame_label
            start = i
    frame_label = np.zeros((n_class))
    frame_label[int(label[-1])] = 1
    encoded_content[-1] = frame_label

    return encoded_content

def cal_top5(pred, gold, trg_pad_idx, smoothing=False):
    pred = pred.topk(5, dim=1)[1]   #[B, 5]
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = 0
    for i in range(5):
        n_correct += pred[:, i].eq(gold).masked_select(non_pad_mask).sum().item()

    return n_correct

def cal_seg_performance(pred, gold, trg_pad_idx):
    loss = cal_loss(pred, gold.long(), -100, smoothing=False)       #-100 is ignore_index
    pred = pred.max(1)[1]
    n_correct = pred.eq(gold).sum()
    n_word = len(gold) - gold.eq(-100).sum()

    return loss, n_correct, n_word

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    '''Apply label smoothing if needed'''
    loss = cal_loss(pred, gold.long(), trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
#    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    '''Calculate cross entropy loss, apply label smoothing if needed'''

#    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class -1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx)
    return loss

def len_activation(lengths, mask):
    return F.normalize(torch.exp(lengths)*mask, p=1, dim=-1)

def _post_process(result, sigma):
    new_res = gaussian_filter1d(result.cpu().detach().numpy(), sigma=sigma, axis=0)
    return new_res

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def eval_file(gt_content, recog_content, obs_percentage, classes):
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        if ground_truth[i] == recognized[i]:
            n_T[classes[ground_truth[i]]] += 1
        else:
            n_F[classes[ground_truth[i]]] += 1

    return n_T, n_F

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels.detach().cpu())
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0

    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls / len(classes)

def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        scores: numpy nd array, shape = (instance_count, label_count)
        labels: numpy nd array, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
#    rankings = scores.argsort()[:, ::-1].copy()
    rankings = torch.flip(scores.argsort(), [0,1])
    maxk = np.max(ks)  # trim to max k to avoid extra computation

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]


if __name__ == '__main__':
    file_path = './cvpr18_data/bf_splits/groundTruth/P03_cam01_P03_cereals.txt'
    f = open(file_path, 'r')
    content=[]
    for l in f :
        content.append(l.split('\n')[0])

    label, length = get_label_length_seq(content)



