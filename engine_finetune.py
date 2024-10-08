# Partly revised by YP @Anhui Medical University
# --------------------------------------------------------

import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from pycm import *
import numpy as np
from sklearn import metrics
from util.loss import un_loss

def train(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            evidences = [F.softplus(outputs)]
            loss_un = 0
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))
                
            loss_CE = criterion(b, targets)

            loss_un += un_loss(targets, alpha[0], args.nb_classes, epoch, args.epochs, device)
            loss_ACE = torch.mean(loss_un)
            loss = loss_CE+loss_ACE

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args,data_loader, model, device, task, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    labels = []
    outputs = []

    predictions = []
    gts = []
    num_total = 0
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            
            evidences = [F.softplus(output)]
            loss_un = 0
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            loss_CE = criterion(b, target)


            loss_un += un_loss(target, alpha[0], args.nb_classes, epoch, args.epochs, device)
            loss_ACE = torch.mean(loss_un)
            loss = loss_CE+loss_ACE

            pred = torch.softmax(b,dim=1)

            data_bach = pred.size(0)
            num_total += data_bach
            one_hot = torch.zeros(data_bach, args.nb_classes).to(device).scatter_(1, target.unsqueeze(1), 1)
            pred_decision = pred.argmax(dim=-1)
            for idx in range(data_bach):
                outputs.append(pred.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                gts.append(target.cpu().detach().float().numpy()[idx])
            
        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            
    metric_logger.synchronize_between_processes()

    epoch_auc = metrics.roc_auc_score(labels, outputs)
    Acc = metrics.accuracy_score(gts, predictions)
    if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
        os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))

    with open(os.path.join(args.save_model_path,"{}/{}_Metric.txt".format(args.net_work,args.net_work)),'a+') as Txt:
        Txt.write("Epoch {}: {} == Acc: {}, AUC: {}\n".format(
            epoch,mode, round(Acc,6),round(epoch_auc,6)
        ))
    print("Epoch {}: {} == Acc: {}, AUC: {}\n".format(
            epoch,mode,round(Acc,6),round(epoch_auc,6)
        ))
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, epoch_auc,Acc

