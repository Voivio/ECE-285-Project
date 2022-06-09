import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import datetime
import models
from datasets import *
from datasets.validation_folders import ValidationSet

import numpy as np
from utils import tensor2array, save_checkpoint, save_optimizer_lr_scheduler
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors, compute_loss_with_gt
from torch.utils.tensorboard import SummaryWriter

import argparse

parser = argparse.ArgumentParser(description='Deep VO with Sequential Learning Optimization')

parser.add_argument('--dataroot', type=str, default='E:\sfm_kitti\small_256', help='path to dataset')
parser.add_argument('--sequence-length', type=int, help='sequence length for training', default=3)
parser.add_argument('--epochId', type=int, default=200, help='The number of epochs being trained')
parser.add_argument('--batch_size', type=int, default=2, help='The size of a train batch')
parser.add_argument('--valbatch_size', type=int, default=1, help='The size of a val batch')
parser.add_argument('--initLR', type=float, default=1e-4, help='The initial learning rate')
parser.add_argument('--multi_step_LR', action='store_true', default=False, help='The epoch to decrease learning rate')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--experiment', default='train', help='The path to store sampled images and models')
parser.add_argument('--workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--reco_frq', type=int, default=100, help='Number of iterations to record loss')
parser.add_argument('--dataset', type=str, choices=['kitti'], default='kitti', help='the dataset to train')
parser.add_argument('--with_gt', action='store_true', default=True, help='use ground truth for validation')
parser.add_argument('--with_pretrain', type=bool, default=True, help='use ground truth for validation')
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W',
                    default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss',
                    metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1,
                    help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int, default=0, help='with the the mask for stationary points')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--resume_checkpoint', type=bool, default=False, help='resume previous training or not')

iteration = 0
best_error = -1


def main():
    args = parser.parse_args()

    # Initialize tensorboard
    timestamp = datetime.datetime.now().strftime("%m_%d_%H%M")
    save_path = 'run/' + args.experiment + timestamp
    writer = SummaryWriter(save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Prepare dataset
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    train_set = ValidationSet(
        args.dataroot,
        transform=train_transform,
        dataset=args.dataset
    )
    val_set = ValidationSet(
        args.dataroot,
        transform=valid_transform,
        dataset=args.dataset
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.valbatch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    # Move network and containers to gpu
    dep_net = models.DepthNet(args.with_pretrain).to(device)
    dep_net = torch.nn.DataParallel(dep_net)

    optim_params = [
        {'params': dep_net.parameters(), 'lr': args.initLR},
    ]

    optimizer = optim.Adam(optim_params,
                           weight_decay=args.weight_decay)
    if args.multi_step_LR:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [0.6 * args.epochId, 0.8 * args.epochId])

    global best_error

    if args.checkpoint_path:
        print("resuming existing checkpoint...")
        dep_net.load_state_dict(save_path + 'disp_net_checkpoint.pth.tar')
        optimizer.load_state_dict(save_path + 'optimizer.pth.tar')
        lr_scheduler.load_state_dict(save_path + 'lr_scheduler.pth.tar')
        best_error = np.load(save_path + "best_error.npy")

    for epoch in range(args.epochId):
        train(args, device, train_loader, dep_net, optimizer, epoch, writer)
        errors, error_names = validate_with_gt(args, device, val_loader, dep_net, epoch, writer)
        if args.multi_step_LR:
            lr_scheduler.step()
        print("One val finished")
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            save_path, {
                'epoch': epoch + 1,
                'state_dict': dep_net.module.state_dict()
            }, is_best, filename="depth_only_cp.pth.tar")
        save_optimizer_lr_scheduler(
            save_path, {
                'epoch': epoch + 1,
                'state_dict': optimizer.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': lr_scheduler.state_dict()
            },
            is_best)

        if is_best:
            np.save(save_path+"best_error.npy", best_error)


def train(args, device, train_loader, disp_net, optimizer, epoch, writer):
    global iteration
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()

    total_loss = 0

    for i, (tgt_img, depth) in enumerate(train_loader):
        # tgt_img : (B, C, H, W)
        # tgt_img : (B, 1, H, W)
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # compute output
        tgt_depth = 1 / disp_net(tgt_img)
        loss = compute_loss_with_gt(depth, tgt_depth, args.dataset)

        total_loss += loss

        # record loss
        if (iteration + 1) % args.reco_frq == 0:
            total_loss /= args.reco_frq

            print(f"training loss: {total_loss:>7f}  Epoch:{epoch:>d}  Curr Iter: [{iteration + 1:>5d}]")
            # ...log the running loss
            writer.add_scalar('training total loss',
                              total_loss,
                              iteration + 1)

            total_loss = 0

        # back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
    print('One  epoch training finished')


@torch.no_grad()
def validate_with_gt(args, device, val_loader, disp_net, epoch, writer):
    disp_net.eval()
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    total_err = np.array([0.0] * 6)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1 / output_disp[:, 0]

        if i == 0:
            # only write the result once for each validation
            if epoch == 0:
                writer.add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                writer.add_image('val target Depth',
                                 tensor2array(depth_to_show, max_value=10),
                                 epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1 / depth_to_show).clamp(0, 10)
                writer.add_image('val target Disparity Normalized',
                                 tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                 epoch)

            writer.add_image('val Dispnet Output Normalized',
                             tensor2array(output_disp[0], max_value=None, colormap='magma'),
                             epoch)
            writer.add_image('val Depth Output',
                             tensor2array(output_depth[0], max_value=10),
                             epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)
        errors = compute_errors(depth, output_depth, args.dataset)
        total_err += np.array(errors)

    total_err /= i + 1
    for error, name in zip(list(total_err), error_names):
        print("validation " + f"{name}: ", f'{error}')
        writer.add_scalar(name, error, epoch)

    return total_err, error_names


if __name__ == '__main__':
    main()
