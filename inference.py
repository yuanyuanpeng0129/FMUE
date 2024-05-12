# coding:gbk
import os
import argparse
import torch
import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import models_vit
from util.datasets import build_dataset
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score,average_precision_score,precision_score,f1_score,recall_score
import matplotlib.pyplot as plt
from scipy import interp
import csv
from sklearn import metrics
from itertools import cycle

from sklearn.metrics import confusion_matrix

from util.pos_embed import interpolate_pos_embed
import util.misc as misc
import seaborn as sns
import sys
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='../OCT_data/', type=str,
                        help='dataset path')####���ǵ�OCT����·��
    parser.add_argument('--nb_classes', default=11, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--save_path', default='./trained/FMUE.pth',
                        help='path where to save model path')###����ģ�͵�·��
    parser.add_argument('--method', default='FMUE',
                        help='path where to save the test result')###����ģ�͵�·��

    return parser

def sen(Y_test, Y_pred, n):  # nΪ������
    sen = ['Sen']
    con_mat = confusion_matrix(Y_test, Y_pred)
    print(con_mat)
    sns.heatmap(con_mat,annot = True,fmt = 'd',cmap="OrRd",annot_kws={'size':11, 'color':'black'},cbar=False)
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.draw()
    plt.savefig('./results/{}.tif'.format(args.method))
    plt.pause(1)
    plt.close()

    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen

def F1_score(Y_test, Y_pred, n):  # nΪ������
    F1_score = ['F1 Score']
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        sen1 = tp / (tp + fn)
        pre1 = tp / (tp + fp)

        F1 = 2*(pre1*sen1)/(pre1+sen1)
        F1_score.append(F1)

    return F1_score

def pre(Y_test, Y_pred, n):
    pre = ['Pre']
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:, i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)

    return pre

def spe(Y_test, Y_pred, n):
    spe = ['Spe']
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe

def val(val_dataloader, model, args, mode, device):

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()
    labels = []
    outputs = []

    predictions = []
    gts = []

    prediction_list = []
    predict_return = []
    label_list_return = []
    preds_confu = []
    label_confu = []
    label_list = []

    correct = 0.0
    num_total = 0
    tbar = tqdm.tqdm(val_dataloader, desc='\r')  # ����һ��������ʾ��Ϣ��ֻ��Ҫ��װ����ĵ����� tqdm(iterator)��desc������ǰ׺

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            OCT_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            
            output = model(OCT_img)
            evidences = [F.softplus(output)]
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            pred = E / (S.expand(E.shape))
            
            #label_pred = np.argmax(b.cpu().data.numpy())

            data_bach = pred.size(0)
            num_total += data_bach
            one_hot = torch.zeros(data_bach, args.nb_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pred_decision = pred.argmax(dim=-1)
            pred_con = pred.argmax(dim=-1)
            for idx in range(data_bach):
                outputs.append(pred.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                gts.append(cls_label.cpu().detach().float().numpy()[idx])

                predict_return.append(torch.unsqueeze(pred.cpu()[idx], dim=0))
                label_list_return.append(torch.unsqueeze(one_hot.cpu()[idx], dim=0))
                preds_confu.append(pred_con.cpu().numpy()[idx])
                label_confu.append(cls_label.cpu().numpy()[idx])

                prediction_list.append(pred_con.cpu().detach().float().numpy()[idx])
                label_list.append(cls_label.cpu().detach().float().numpy()[idx])

    epoch_auc = metrics.roc_auc_score(labels, outputs)
    Acc = metrics.accuracy_score(gts, predictions)
    F1Score = f1_score(gts, predictions,average='macro')

    print("{} == Acc: {}, AUC: {}, F1Score: {}\n".format(mode,round(Acc,6),round(epoch_auc,6),round(F1Score,6)
        ))
    torch.cuda.empty_cache()
    # return epoch_auc,Acc
    return label_list_return, predict_return, preds_confu, label_confu

def main(args=None):
    # bulid model
    device = torch.device(args.device)
    args.device = device

    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    model.eval()

    checkpoint = torch.load(args.save_path, map_location='cpu')

    print("Load trained checkpoint from: %s" % args.save_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load trained model
    model.load_state_dict(checkpoint_model, strict=False)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('&&&&&&&&&&&&&&&')
    print(msg)

    print('Done!')
    model.to(device)

    dataset_test = build_dataset(is_train='test', args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
            
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    test_loader = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    label_list_return, predict_return, preds_confu, label_confu = val(
        test_loader, model, args, mode="test",device=device)  # ��֤�����
    label_list_return = torch.cat(label_list_return, dim=0).numpy()
    predict_return = torch.cat(predict_return, dim=0).numpy()
    # ����ÿһ���ROC
    fpr_Pri = dict()
    tpr_Pri = dict()
    roc_auc_Pri = dict()
    for i in range(args.nb_classes):
        fpr_Pri[i], tpr_Pri[i], _ = roc_curve(label_list_return[:, i], predict_return[:, i])
        roc_auc_Pri[i] = auc(fpr_Pri[i], tpr_Pri[i])

    # Compute micro-average ROC curve and ROC area����������
    fpr_Pri["micro"], tpr_Pri["micro"], _ = roc_curve(label_list_return.ravel(), predict_return.ravel())
    roc_auc_Pri["micro"] = auc(fpr_Pri["micro"], tpr_Pri["micro"])

    # Compute macro-average ROC curve and ROC area������һ��
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_Pri[i] for i in range(args.nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.nb_classes):
        mean_tpr += interp(all_fpr, fpr_Pri[i], tpr_Pri[i])

    # Finally average it and compute AUC
    mean_tpr /= args.nb_classes
    fpr_Pri["macro"] = all_fpr
    tpr_Pri["macro"] = mean_tpr
    roc_auc_Pri["macro"] = auc(fpr_Pri["macro"], tpr_Pri["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr_Pri["macro"], tpr_Pri["macro"],
             label='Average (AUC = {0:0.4f})'
                   ''.format(roc_auc_Pri["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['r', 'aqua', 'darkorange', 'cornflowerblue', 'yellow', 'peru', 'black', 'violet', 'crimson','deeppink','brown'])
    Clases_name = ['normal','acute VKH','acute CSC', 'acute RAO', 'acute RVO',  'AMD','DR',
                   'macular-off RRD', 'mCNV', 'MTM','RP']

    for i, color in zip(range(args.nb_classes), colors):
        plt.plot(fpr_Pri[i], tpr_Pri[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.4f})'.format(Clases_name[i], roc_auc_Pri[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("./results/{}.png".format(args.method))
    plt.close()

    Sens = sen(label_confu, preds_confu, n=args.nb_classes)
    Precisions = pre(label_confu, preds_confu, n=args.nb_classes)
    F1_scores = F1_score(label_confu, preds_confu, n=args.nb_classes)
    Spes = spe(label_confu, preds_confu, n=args.nb_classes)


    all_metrics = [Sens,Precisions,F1_scores,Spes]
    header = ['Metrics','Normal','acute VKH','acute CSC', 'acute RAO', 'acute RVO',  'AMD','DR',
                   'Macular-off RRD', 'mCNV', 'MTM','RP']
    with open("./results/{}.csv".format(args.method), 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(
            all_metrics
        )


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()  # ��������

    main(args=args)