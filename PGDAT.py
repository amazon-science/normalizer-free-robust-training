import os, sys, argparse, time, socket
from functools import partial
sys.path.append('./')
import numpy as np 

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import Subset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from timm.models import create_model
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import PGDAttack, FGSM

from dataloaders.imagenet import imagenet_datasets

from utils.utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', '--cpus', default=24, type=int)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='IN', choices=['IN'], help='which dataset to use')
    parser.add_argument('--data_dir', '--dd', default='/ssd1/haotao/datasets/imagenet', help='ImageNet dataset path.')
    parser.add_argument('--model', '--md', default='resnet26', choices=['resnet26', 'resnet50', 'nf_resnet26', 'nf_resnet50'], help='which model to use')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=90, help='Number of epochs to train.')
    parser.add_argument('--decay_epochs', '--de', default=[30,60], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=100)
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--wd', type=float, default=5e-5, help='Weight decay (L2 penalty).')
    parser.add_argument('--drop', type=float, default=0.25, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: None)')
    parser.add_argument('--drop_block', type=float, default=None, metavar='PCT', help='Drop block rate (default: None)')
    # attack
    parser.add_argument('--steps', default=7, type=int, help='attack iteration number')
    parser.add_argument('--val_steps', default=5, type=int, help='attack iteration number for validation')
    parser.add_argument('--tau', default=1, type=int, help='attack iteration number') # here tau is one larger than the tau in paper.
    parser.add_argument('--steps_list', '--sl', default=[3,5,7], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--tau_list', '--tl', default=[0,1,2], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--eps', default=8, type=int, help='eps for pgd attack')
    parser.add_argument('--alpha', default=-1, type=float, help='attack step size in uint8; -1 means auto step size.')
    parser.add_argument('--targeted', action='store_true', help='If true, use targeted attack in PGDAT')
    parser.add_argument('--Lambda', type=float, default=0.5)
    # others:
    parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
    parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao/')
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    args = parser.parse_args()
    return args

def create_save_path():
    # mkdirs:
    dataset_str = args.dataset
    model_str = args.model
    if args.Lambda != 0:
        steps_str = '%d' % args.steps
        args.alpha = args.alpha if args.alpha > 0 else min(1.25*args.eps, args.eps+4)/args.steps # here alpha is unint8
        attack_str = '%s-%s-%d-%.2f-Lambda%s' % ('PGD', steps_str, args.eps, args.alpha, args.Lambda)
    else:
        attack_str = 'Normal'
    if args.opt == 'sgd':
        opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s-smoothing%s-dropout%s-droppath%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd, args.smoothing, args.drop, args.drop_path)
    elif args.opt == 'adam':
        opt_str = 'e%d-b%d_adam-lr%s-wd%s-smoothing%s-dropout%s-droppath%s' % (args.epochs, args.batch_size, args.lr, args.wd, args.smoothing, args.drop, args.drop_path)
    if args.decay == 'cos':
        decay_str = 'cos'
    elif args.decay == 'multisteps':
        decay_str = 'multisteps-' + '-'.join(map(str, args.decay_epochs)) 

    if args.Lambda == 0:
        method_str = 'Normal'
    else:
        method_str = 'PGDAT'
    save_folder = os.path.join(args.save_root_path, 'nf_robustness_results', dataset_str, method_str, model_str, '%s_%s_%s' % (attack_str, opt_str, decay_str))
    if args.resume:
        print('Loading pretrained model from %s' % save_folder)
    else:
        create_dir(save_folder)
        print('saving to %s' % save_folder)

    return save_folder

def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(gpu_id, ngpus_per_node, args):

    save_folder = args.save_folder
    
    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    if gpu_id == 0:
        print(args)

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True # set cudnn.benchmark in each worker, as done in https://github.com/pytorch/examples/blob/b0649dcd638eb553238cdd994127fd40c8d9a93a/imagenet/main.py#L199

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data loader:
    num_classes = 1000
    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).to(device)
    train_data, val_data = imagenet_datasets(data_dir=args.data_dir, num_classes=num_classes)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model:
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d if 'nf' not in args.model else None,
        zero_init_last_bn=True if 'nf' not in args.model else None, 
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        # drop_block_rate=args.drop_block,
    ).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False)
    else:
        model = torch.nn.DataParallel(model)

    # attacker:
    attacker = PGDAttack(NormalizedModel(model, mean, std), eps=args.eps/255., nb_iter=args.steps, eps_iter=args.alpha/255., targeted=args.targeted) 
    val_steps = args.val_steps
    # val_steps_weak = 3
    attacker_val = PGDAttack(NormalizedModel(model, mean, std), eps=args.eps/255., nb_iter=val_steps, eps_iter=args.alpha/255., targeted=False) 
    # attacker_val_weak = PGDAttack(NormalizedModel(model, mean, std), eps=args.eps/255., nb_iter=val_steps_weak, eps_iter=args.alpha/255., targeted=False) 

    # adjust learning rate:
    args.lr *= args.batch_size / 256. # linearly scaled to batch size

    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        # scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t=5, warmup_lr_init=0.0001)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # loss_fn:
    loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # load ckpt:
    if args.resume:
        last_epoch, best_SA, best_RA, training_loss, val_SA, val_RA \
            = load_ckpt_adv(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_SA = 0
        best_RA = 0
        # training curve lists:
        training_loss, val_SA, val_RA = [], [], []

    # train:
    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        fp_train = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
        fp_val = open(os.path.join(save_folder, 'val_log.txt'), 'a+')
        start_time = time.time()

        ## training:
        model.train()
        requires_grad_(model, True)
        accs, accs_adv, losses, losses_clean, losses_adv = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        current_lr = scheduler.get_last_lr()[0]
        for i, (images, labels) in enumerate(train_loader):

            batch_start_time = time.time()

            # get batch:
            images = images.to(device)
            labels = labels.to(device)

            # logits for clean imgs:
            logits = model(images)

            # loss:
            loss_clean = loss_fn(logits, labels)

            # generate adv images:
            if args.Lambda != 0 :

                # Generate adv imgs:
                with ctx_noparamgrad_and_eval(model):
                    images = images * std + mean # range back to [0,1]
                    # print('images:', torch.max(images), torch.min(images))
                    if args.targeted:
                        attack_y = (labels + torch.randint_like(labels, 1, num_classes)) % num_classes
                        assert torch.sum(attack_y == labels) == 0
                    else:
                        attack_y = labels
                    # Generate adv imgs:
                    images_adv = attacker.perturb(images, attack_y) 
                    labels_reranked = labels
                    # print('images_adv:', torch.max(images_adv), torch.min(images_adv))
                    images_adv = (images_adv-mean)/std
                
                # get adv loss:
                logits_adv = model(images_adv)
                loss_adv = loss_fn(logits_adv, labels_reranked)
                loss = (1-args.Lambda) * loss_clean + args.Lambda * loss_adv
            
            else: # Normal training
                loss = loss_clean

            # update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_end_time = time.time()

            # metrics:
            accs.append((logits.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())
            if args.Lambda != 0:
                accs_adv.append((logits_adv.argmax(1) == labels_reranked).float().mean().item())
                losses_clean.append(loss_clean.item())
                losses_adv.append(loss_adv.item())

            if i % 50 == 0:
                if args.Lambda != 0:
                    train_str = 'Epoch %d-%d | Train | Loss: %.4f (%.4f, %.4f), SA: %.4f, RA: %.4f | time: %.2f (sec/batch) | lr: %s' % (epoch, i, losses.avg, losses_clean.avg, losses_adv.avg, accs.avg, accs_adv.avg, batch_end_time-batch_start_time, current_lr)
                else:
                    train_str = 'Epoch %d-%d | Train | Loss: %.4f, SA: %.4f | time: %.2f (sec/batch) | lr: %s' % (epoch, i, losses.avg, accs.avg, batch_end_time-batch_start_time, current_lr)
                if gpu_id == 0:
                    print(train_str)
                    fp_train.write(train_str + '\n')
                    fp_train.flush()
        
        # lr schedualr update at the end of each epoch:
        scheduler.step()

        ## validation:
        if rank == 0:
            model.eval()
            requires_grad_(model, False)
            print(model.training)

            eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.9*args.epochs)) # boolean

            val_SAs, val_RAs, val_RAs_weak = AverageMeter(), AverageMeter(), AverageMeter()
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_SAs.append((logits.argmax(1) == labels).float().mean().item())
                if args.Lambda != 0 and eval_this_epoch:
                    with ctx_noparamgrad_and_eval(model):
                        images = images * std + mean # range back to [0,1]
                        images_adv = attacker_val.perturb(images, labels) # always validate on untargeted attack.
                        images_adv = (images_adv-mean)/std
                        # images_adv_weak = attacker_val_weak.perturb(images, labels)
                        # images_adv_weak = (images_adv_weak-mean)/std
                    logits_adv = model(images_adv)
                    val_RAs.append((logits_adv.argmax(1) == labels).float().mean().item())
                    # logits_adv_weak = model(images_adv_weak)
                    # val_RAs_weak.append((logits_adv_weak.argmax(1) == labels).float().mean().item())
            # append to list:
            training_loss.append(losses.avg)
            val_SA.append(val_SAs.avg) 
            if args.Lambda != 0:
                val_RA.append(val_RAs.avg if eval_this_epoch else val_RA[-1]) 

            if args.Lambda != 0:
                if eval_this_epoch:
                    val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f | RA (steps %d): %.4f' % (
                        epoch, (time.time()-start_time), current_lr, val_SA[-1], val_steps, val_RA[-1])
                else:
                    val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f' % (
                        epoch, (time.time()-start_time), current_lr, val_SA[-1])
            else:
                val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f' % (epoch, (time.time()-start_time), current_lr, val_SA[-1])
            print(val_str)
            fp_val.write(val_str + '\n')

            # save loss curve:
            plt.plot(training_loss)
            plt.grid(True)
            plt.title("training loss")
            plt.savefig(os.path.join(save_folder, 'training_loss.png'))
            plt.close()

            plt.plot(val_SA, 'g', label='SA')
            if args.Lambda != 0:
                plt.plot(val_RA, 'r', label='RA')
            plt.grid(True)
            plt.title("val acc")
            plt.legend()
            plt.savefig(os.path.join(save_folder, 'val_acc.png'))
            plt.close()

            # save pth:
            if val_SAs.avg >= best_SA:
                best_SA = val_SAs.avg
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))
            if args.Lambda != 0 and eval_this_epoch:
                if val_RAs.avg >= best_RA:
                    best_RA = val_RAs.avg
                    torch.save(model.state_dict(), os.path.join(save_folder, 'best_RA.pth'))
            save_ckpt_adv(epoch, model, optimizer, scheduler, best_SA, best_RA, training_loss, val_SA, val_RA,
                os.path.join(save_folder, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_folder = create_save_path()
    args.save_folder = save_folder
    
    # set CUDA:
    if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)