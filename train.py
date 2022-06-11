import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import datetime
import models
from datasets import *
import numpy as np
from utils import tensor2array, save_checkpoint, save_single_network_optimizer_lr_scheduler
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
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
parser.add_argument('--depth_net_checkpoint', type=str, default="", help='path to depth net checkpoint')
parser.add_argument('--pose_net_checkpoint', type=str, default="", help='path to pose net checkpoint')

parser.add_argument('--resume_optimizer', type=bool, default=False, help='resume the state of optimizer and error training or not')

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

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.Resize((256, 256)),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    train_set = SequenceFolder(
        args.dataroot,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        dataset=args.dataset
    )

    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.dataroot,
            transform=valid_transform,
            dataset=args.dataset
        )
    else:
        val_set = SequenceFolder(
            args.dataroot,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
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
    pose_net = models.PoseNet(args.with_pretrain).to(device)

    dep_net = torch.nn.DataParallel(dep_net)
    pose_net = torch.nn.DataParallel(pose_net)

    optim_params = [
        {'params': dep_net.parameters(), 'lr': args.initLR},
        {'params': pose_net.parameters(), 'lr': args.initLR}
    ]

    optimizer = optim.Adam(optim_params,
                           weight_decay=args.weight_decay)
    if args.multi_step_LR:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [0.6 * args.epochId, 0.8 * args.epochId])

    global best_error

    # resume model with existing weights
    if args.depth_net_checkpoint:
        print("resuming depth net checkpoint...")
        weights = torch.load(args.depth_net_checkpoint)
        dep_net.module.load_state_dict(weights['state_dict'])

    if args.pose_net_checkpoint:
        print("resuming pose net checkpoint...")
        weights = torch.load(args.pose_net_checkpoint)
        pose_net.module.load_state_dict(weights['state_dict'])

    if args.resume_optimizer:
        optimizer.load_state_dict(save_path + 'optimizer.pth.tar')
        best_error = np.load(save_path + "best_error.npy")
        if args.multi_step_LR:
            lr_scheduler.load_state_dict(save_path + 'lr_scheduler.pth.tar')

    for epoch in range(args.epochId):
        train_loss = train(args, device, train_loader, dep_net, pose_net, optimizer, epoch, writer)
        if args.with_gt:
            errors, error_names = validate_with_gt(args, device, val_loader, dep_net, epoch, writer)
        else:
            errors, error_names = validate_without_gt(args, device, val_loader, dep_net, pose_net, epoch, writer)
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
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)
        save_single_network_optimizer_lr_scheduler(
            save_path, None, "",
            {
                'epoch': epoch + 1,
                'state_dict': optimizer.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': lr_scheduler.state_dict()
            },
            is_best)

        if is_best:
            np.save(save_path+"best_error.npy", best_error)


def train(args, device, train_loader, disp_net, pose_net, optimizer, epoch, writer):
    global iteration
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()
    pose_net.train()

    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        # tgt_img : (B, C, H, W)
        # ref_images : (B, C, H, W)[], default length 2. one before tgt and one after tgt
        # intrinsics: (B, 3, 3)
        # intrinsics_inv: (B, 3, 3)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        # tgt_depth: (B, 1, H, W); ref_depths: (B, 1, H, W)[], default length 2

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        # poses: (B, 6)[], default length 2; poses_inv: (B, 6)[], default length 2

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        total_loss += loss
        total_loss1 += loss_1
        total_loss2 += loss_2
        total_loss3 += loss_3
        # record loss
        if (iteration + 1) % args.reco_frq == 0:
            total_loss /= args.reco_frq
            total_loss1 /= args.reco_frq
            total_loss2 /= args.reco_frq
            total_loss3 /= args.reco_frq

            print(f"training loss: {total_loss:>7f}  Epoch:{epoch:>d}  Curr Iter: [{iteration + 1:>5d}]")
            # ...log the running loss
            writer.add_scalar('training mixed loss',
                              total_loss,
                              iteration + 1)

            writer.add_scalar('training photometric loss',
                              total_loss1,
                              iteration + 1)

            writer.add_scalar('training disparity smoothness',
                              total_loss2,
                              iteration + 1)

            writer.add_scalar('training depth consistency loss',
                              total_loss3,
                              iteration + 1)

            total_loss = 0
            total_loss1 = 0
            total_loss2 = 0
            total_loss3 = 0

        # back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
    print('One  epoch training finished')


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1 / disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1 / disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


@torch.no_grad()
def validate_without_gt(args, device, val_loader, disp_net, pose_net, epoch, writer):
    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        total_loss += loss
        total_loss1 += loss_1
        total_loss2 += loss_2
        total_loss3 += loss_3

    total_loss /= i + 1
    total_loss1 /= i + 1
    total_loss2 /= i + 1
    total_loss3 /= i + 1
    print(f"validation loss: {total_loss.item():>7f}  Epoch:{epoch:>d}")
    # ...log the val loss
    writer.add_scalar('validation mixed loss',
                      total_loss,
                      epoch)

    writer.add_scalar('validation photometric loss',
                      total_loss1,
                      epoch)

    writer.add_scalar('validation disparity smoothness',
                      total_loss2,
                      epoch)

    writer.add_scalar('validation depth consistency loss',
                      total_loss3,
                      epoch)

    writer.add_image('val Dispnet Output Normalized',
                     tensor2array(1 / tgt_depth[0], max_value=None, colormap='magma'),
                     epoch)
    writer.add_image('val Depth Output',
                     tensor2array(tgt_depth[0], max_value=10),
                     epoch)

    return [total_loss, total_loss1, total_loss2, total_loss3], ['Total loss', 'Photo loss', 'Smooth loss',
                                                                 'Consistency loss']


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
