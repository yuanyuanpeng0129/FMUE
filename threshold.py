# coding:gbk
import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from torch.nn import functional as F
from sklearn import metrics

from util.pos_embed import interpolate_pos_embed
import util.misc as misc
import vit_model
from util.datasets import build_dataset

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
    parser.add_argument('--data_path', default='./OCT_data/', type=str,
                        help='dataset path')####我们的OCT数据路径
    parser.add_argument('--nb_classes', default=16, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)

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
                        help='path where to save model path')###保存模型的路径
    parser.add_argument('--method', default='AdaptFormer_CE_seed1',
                        help='path where to save the test result')###保存模型的路径

    return parser

def val(val_dataloader, model, args, mode, device):

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()


    u_list = []
    u_label_list = []

    tbar = tqdm.tqdm(val_dataloader, desc='\r') 

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            OCT_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model.forward(OCT_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.nb_classes / S

            un_gt = 1 - torch.eq(b.argmax(dim=-1), cls_label).float()

            data_bach = pred.size(0)
            for idx in range(data_bach):
                u_list.append(u.cpu()[idx].numpy())
                u_label_list.append(un_gt.cpu()[idx].numpy())

    return u_list, u_label_list


def main(args=None):
    # bulid model
    device = torch.device(args.device)
    args.device = device

    model = vit_model.__dict__[args.model](
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


    dataset_test = build_dataset(is_train='val', args=args)

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

    u_list, u_label_list = val(test_loader, model, args, mode="Validation", device=device) 
    
    precision, recall, thresh = metrics.precision_recall_curve(u_label_list, u_list)
    
    max_j = max(zip(precision, recall), key=lambda x:  2*(x[1]*x[0])/(x[1] + x[0]))
    pred_thresh = thresh[list(zip(precision, recall)).index(max_j)]
    print("opt_pred ===== {}".format(pred_thresh))




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()  # 配置设置

    main(args=args)