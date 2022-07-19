import os, sys, argparse, time, socket, functools

import torchvision
sys.path.append('./')
import numpy as np 
from tqdm import tqdm

import torch
from torch.utils.data import Subset, DataLoader, Subset

from timm.models import create_model

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import PGDAttack, MomentumIterativeAttack, LinfBasicIterativeAttack
from torchattacks import APGD, UPGD

from dataloaders.imagenet import imagenet_datasets, imagenet_sketch_dataset, imagenet_c_dataset, imagenet_r_dataset, imagenet_a_dataset
from dataloaders.thousand_to_200 import indices_in_1k, imagenet_r_mask

from utils.utils import *

num_classes = 1000

def test_imagenet(model, data_dir, fp, ckpt_version, test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    start_time = time.time()

    val_SAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        logits = model(images)
        val_SAs.append((logits.argmax(1) == labels).float().mean().item())

    val_str = '[%s] clean acc: %.4f | time: %.4f' % (ckpt_version, val_SAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

def test_imagenet_bim(model, data_dir, fp, ckpt_version, steps, eps, alpha, targeted, test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).cuda()
    attacker = LinfBasicIterativeAttack(NormalizedModel(model, mean, std), eps=eps, nb_iter=steps, eps_iter=alpha, targeted=targeted) 

    start_time = time.time()

    val_RAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(model):
            images = images * std + mean # range back to [0,1]
            if targeted:
                attack_y = (labels + torch.randint_like(labels, 1, num_classes)) % num_classes
                assert torch.sum(attack_y == labels) == 0
            else:
                attack_y = labels
            images_adv = attacker.perturb(images, attack_y) 
            Linf = torch.mean(torch.amax(torch.abs(images-images_adv), dim=(1,2,3)), dim=0).item()
            images_adv = (images_adv-mean)/std
        logits_adv = model(images_adv)
        RA = (logits_adv.argmax(1) == labels).float().mean().item()
        val_RAs.append(RA)
        print('RA %.4f, Linf %.4f' % (RA, Linf))

    if targeted:
        val_str = '[%s] BIM-targeted-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    else:
        val_str = '[%s] BIM-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

def test_imagenet_pgd(model, data_dir, fp, ckpt_version, steps, eps, alpha, targeted, test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).cuda()
    attacker = PGDAttack(NormalizedModel(model, mean, std), eps=eps, nb_iter=steps, eps_iter=alpha, targeted=targeted) # Following Xie et al., attack step size fixed to 1.    

    start_time = time.time()

    val_RAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(model):
            images = images * std + mean # range back to [0,1]
            if targeted:
                attack_y = (labels + torch.randint_like(labels, 1, num_classes)) % num_classes
                assert torch.sum(attack_y == labels) == 0
            else:
                attack_y = labels
            images_adv = attacker.perturb(images, attack_y) 
            Linf = torch.mean(torch.amax(torch.abs(images-images_adv), dim=(1,2,3)), dim=0).item()
            images_adv = (images_adv-mean)/std
        logits_adv = model(images_adv)
        RA = (logits_adv.argmax(1) == labels).float().mean().item()
        val_RAs.append(RA)
        print('RA %.4f, Linf %.4f' % (RA, Linf))

    if targeted:
        val_str = '[%s] PGD-targeted-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    else:
        val_str = '[%s] PGD-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

def test_imagenet_mia(model, data_dir, fp, ckpt_version, steps, eps, alpha, targeted, test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).cuda()
    attacker = MomentumIterativeAttack(NormalizedModel(model, mean, std), eps=eps, nb_iter=steps, eps_iter=alpha, targeted=targeted) # Following Xie et al., attack step size fixed to 1.    

    start_time = time.time()

    val_RAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(model):
            images = images * std + mean # range back to [0,1]
            if targeted:
                attack_y = (labels + torch.randint_like(labels, 1, num_classes)) % num_classes
                assert torch.sum(attack_y == labels) == 0
            else:
                attack_y = labels
            images_adv = attacker.perturb(images, attack_y) 
            Linf = torch.mean(torch.amax(torch.abs(images-images_adv), dim=(1,2,3)), dim=0).item()
            images_adv = (images_adv-mean)/std
        logits_adv = model(images_adv)
        RA = (logits_adv.argmax(1) == labels).float().mean().item()
        val_RAs.append(RA)
        print('RA %.4f, Linf %.4f' % (RA, Linf))

    if targeted:
        val_str = '[%s] MIA-targeted-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    else:
        val_str = '[%s] MIA-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

def cwloss(logits, target, confidence=50, num_classes=1000):
    # CW attack (untargeted) loss.
    target_onehot = torch.zeros(target.size() + (num_classes,)).to(target.device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    real = (target_onehot * logits).sum(1)
    other = ((1. - target_onehot) * logits - target_onehot * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  
    loss = torch.sum(loss)
    return loss

def test_imagenet_cwlinf(model, data_dir, fp, ckpt_version, steps, eps, alpha, test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).cuda()
    attacker = PGDAttack(NormalizedModel(model, mean, std), eps=eps, nb_iter=steps, eps_iter=alpha, targeted=False, loss_fn=functools.partial(cwloss, confidence=50, num_classes=1000)) # Following Xie et al., attack step size fixed to 1.    

    start_time = time.time()

    val_RAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(model):
            images = images * std + mean # range back to [0,1]
            if args.targeted:
                attack_y = (labels + torch.randint_like(labels, 1, num_classes)) % num_classes
                assert torch.sum(attack_y == labels) == 0
            else:
                attack_y = labels
            images_adv = attacker.perturb(images, attack_y) 
            Linf = torch.mean(torch.amax(torch.abs(images-images_adv), dim=(1,2,3)), dim=0).item()
            images_adv = (images_adv-mean)/std
        logits_adv = model(images_adv)
        RA = (logits_adv.argmax(1) == labels).float().mean().item()
        val_RAs.append(RA)
        print('RA %.4f, Linf %.4f' % (RA, Linf))

    val_str = '[%s] CWlinf-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

def test_imagenet_apgd(model, data_dir, fp, ckpt_version, steps, eps, loss='ce', test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).cuda()
    attacker = APGD(NormalizedModel(model, mean, std), eps=eps, steps=steps, loss=loss) # Following Xie et al., attack step size fixed to 1.    

    start_time = time.time()

    val_RAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(model):
            images = images * std + mean # range back to [0,1]
            _, images_adv = attacker.perturb(images, labels) 
            Linf = torch.mean(torch.amax(torch.abs(images-images_adv), dim=(1,2,3)), dim=0).item()
            images_adv = (images_adv-mean)/std
        logits_adv = model(images_adv)
        RA = (logits_adv.argmax(1) == labels).float().mean().item()
        val_RAs.append(RA)
        print('RA %.4f, Linf %.4f' % (RA, Linf))

    val_str = '[%s] APGD-%s-%d-%.4f: acc %.4f | time %.4f' % (ckpt_version, loss, steps, eps, val_RAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

def test_imagenet_mipgd(model, data_dir, fp, ckpt_version, steps, eps, alpha, test_batch_size=100, num_workers=12):
    _, val_data = imagenet_datasets(data_dir=data_dir)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).cuda()
    attacker = UPGD(NormalizedModel(model, mean, std), eps=eps, steps=steps, alpha=alpha, random_start=True, decay=1)    

    start_time = time.time()

    val_RAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(model):
            images = images * std + mean # range back to [0,1]
            images_adv = attacker(images, labels) 
            Linf = torch.mean(torch.amax(torch.abs(images-images_adv), dim=(1,2,3)), dim=0).item()
            images_adv = (images_adv-mean)/std
        logits_adv = model(images_adv)
        RA = (logits_adv.argmax(1) == labels).float().mean().item()
        val_RAs.append(RA)
        print('RA %.4f, Linf %.4f' % (RA, Linf))

    val_str = '[%s] MIPGD-%d-%.4f-%.4f: acc %.4f | time %.4f' % (ckpt_version, steps, eps, alpha, val_RAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

## ImageNet-Sketch
def test_imagenet_sketch(model, data_dir, fp, ckpt_version, test_batch_size=100, num_workers=12):
    sketch_data = imagenet_sketch_dataset(data_dir=data_dir)
    sketch_loader = DataLoader(sketch_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    start_time = time.time()

    val_SAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(sketch_loader)):
        images, labels = images.cuda(), labels.cuda()
        logits = model(images)
        val_SAs.append((logits.argmax(1) == labels).float().mean().item())

    val_str = '[%s] ImageNet-Sketch acc: %.4f | time: %.4f' % (ckpt_version, val_SAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

## ImageNet-R
def test_imagenet_r(model, data_dir, fp, ckpt_version, test_batch_size=100, num_workers=12):
    sketch_data = imagenet_r_dataset(data_dir=data_dir)
    sketch_loader = DataLoader(sketch_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    start_time = time.time()

    val_SAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(sketch_loader)):
        images, labels = images.cuda(), labels.cuda()
        logits = model(images)[:,imagenet_r_mask]
        val_SAs.append((logits.argmax(1) == labels).float().mean().item())

    val_str = '[%s] ImageNet-R acc: %.4f | time: %.4f' % (ckpt_version, val_SAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

## ImageNet-A
def test_imagenet_a(model, data_dir, fp, ckpt_version, test_batch_size=100, num_workers=12):
    sketch_data = imagenet_a_dataset(data_dir=data_dir)
    sketch_loader = DataLoader(sketch_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    start_time = time.time()

    val_SAs = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(sketch_loader)):
        images, labels = images.cuda(), labels.cuda()
        logits = model(images)[:,indices_in_1k]
        val_SAs.append((logits.argmax(1) == labels).float().mean().item())

    val_str = '[%s] ImageNet-A acc: %.4f | time: %.4f' % (ckpt_version, val_SAs.avg, (time.time()-start_time))
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()

## ImageNet-C
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]

def find_mCE(target_model_c_CE, anchor_model_c_CE=ALEXNET_ERR):
    '''
    Args:
        target_model_c_CE: np.ndarray. shape=(15). CE of each corruption type of the target model.
        anchor_model_c_CE: np.ndarray. shape=(15). CE of each corruption type of the anchor model (normally trained ResNet18 as default).
    '''
    assert len(target_model_c_CE) == 15 # a total of 15 types of corruptions
    mCE = 0
    for target_model_CE, anchor_model_CE in zip(target_model_c_CE, anchor_model_c_CE):
        mCE += target_model_CE/anchor_model_CE
    mCE /= len(target_model_c_CE)
    return mCE

def test_imagenet_c(model, data_dir, fp, ckpt_version, test_batch_size=100, num_workers=12):
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]
    corruption_loader_list = []
    for corruption in CORRUPTIONS:
        each_corruption_loader_list = []
        for severity in range(1,6):
            corruption_data = imagenet_c_dataset(corruption=corruption, severity=severity, data_dir=data_dir)
            corruption_loader = DataLoader(corruption_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            each_corruption_loader_list.append(corruption_loader)
        corruption_loader_list.append(each_corruption_loader_list)

    test_CE_c_list = []
    for corruption, each_corruption_loader_list in zip(CORRUPTIONS, corruption_loader_list):
        print('Evaluation on %s' % corruption)
        test_c_CE_c_s_list = []
        start_time = time.time()
        # alexnet_CE_c_s_list = alexnet_CE_dict[corruption]
        for severity in range(1,6):
            test_c_loader_c_s = each_corruption_loader_list[severity-1]
            test_c_CE_meter = AverageMeter()
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_c_loader_c_s):
                    images, targets = images.cuda(), targets.cuda()
                    logits = model(images)
                    pred = logits.data.max(1)[1]
                    CE = (~pred.eq(targets.data)).float().mean()
                    test_c_CE_meter.append(CE.item())
            # test acc of each type of corruptions:
            test_c_CE_c_s = test_c_CE_meter.avg
            test_c_CE_c_s_list.append(test_c_CE_c_s)
        test_CE_c = np.mean(test_c_CE_c_s_list)
        test_CE_c_list.append(test_CE_c)

        # print
        print('%s test time: %.2fs' % (corruption, time.time()-start_time))
        corruption_str = '%s CE: %.4f' % (corruption, test_CE_c)
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of corruptions:
    test_c_acc = 1-np.mean(test_CE_c_list)
    test_mCE = find_mCE(test_CE_c_list)

    val_str = '[%s] ImageNet-C acc: %.4f | mCE: %.4f' % (ckpt_version, test_c_acc, test_mCE)
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test on differen ImageNet benchmark datasets')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', '--cpus', default=12, type=int)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='IN', choices=['IN'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='root path for datasets.')
    parser.add_argument('--model', '--md', default='resnet26', choices=['nf_resnet26', 'nf_resnet50', 'resnet26', 'resnet50'], help='which model to use')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=100)
    parser.add_argument('--norm', default='bn', choices=['bn', 'in', 'gn', 'ln'], help='which normalization to use')
    # attack
    parser.add_argument('--targeted', action='store_true', help='If true, use targeted attack in PGDAT')
    parser.add_argument('--steps', default=10000, type=int, help='attack iteration number')
    parser.add_argument('--eps', default=8, type=int, help='eps for pgd attack')
    parser.add_argument('--alpha', default=-1, type=float, help='attack step size in uint8; -1 means auto step size.')
    parser.add_argument('--apgd_loss', default='ce', choices=['ce', 'dlr'], help='Loss fn for APGD attack')
    parser.add_argument('--restarts', default=1, type=int, help='Loss fn for APGD attack')
    # mode:
    parser.add_argument('--mode', default='all', choices=['all', 'clean', 'pgd', 'bim', 'mia', 'mipgd', 'apgd', 'cw', 'sketch', 'c', 'r', 'a'], help='which evaluation set to use')
    parser.add_argument('--ckpt_path', help='ckpt_path')
    parser.add_argument('--ckpt_version', '--ckpt', default='best_RA', choices=['best_SA', 'best_RA'], type=str, help='')
    args = parser.parse_args()
    args.alpha = args.alpha if args.alpha > 0 else min(1.25*args.eps, args.eps+4)/args.steps # here alpha is unint8

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    # model:
    if 'nf' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=num_classes
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            norm_layer=nn.BatchNorm2d,
            num_classes=num_classes
        )
    if 'DeepAug_Official' in args.ckpt_path or 'FastAT_Official' in args.ckpt_path or 'Robustness_Library' in args.ckpt_path:
        model = torchvision.models.resnet50()
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, '%s.pth' % args.ckpt_version)))
    except FileNotFoundError:
        print('Checkpoint file does not exist %s, skipping...' % os.path.join(args.ckpt_path, '%s.pth' % args.ckpt_version))
        exit(0)
    
    model.eval()
    requires_grad_(model, False)
    print(model.training)

    fp = open(os.path.join(args.ckpt_path, 'test_results.txt'), 'a+')

    if args.mode in ['clean', 'all']:
        test_imagenet(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers)
    if args.mode in ['pgd']:
        test_imagenet_pgd(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers, 
                        steps=args.steps, eps=args.eps/255., alpha=args.alpha/255., targeted=args.targeted, )
    if args.mode in ['bim']:
        test_imagenet_bim(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers, 
                        steps=args.steps, eps=args.eps/255., alpha=args.alpha/255., targeted=args.targeted, )
    if args.mode in ['mia']:
        test_imagenet_mia(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers, 
                        steps=args.steps, eps=args.eps/255., alpha=args.alpha/255., targeted=args.targeted, )
    if args.mode in ['mipgd']:
        test_imagenet_mipgd(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers, 
                        steps=args.steps, eps=args.eps/255., alpha=args.alpha/255.)
    if args.mode in ['cw']:
        test_imagenet_cwlinf(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers, 
                        steps=args.steps, eps=args.eps/255., alpha=args.alpha/255.)
    if args.mode in ['apgd']:
        test_imagenet_apgd(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers,
                        steps=args.steps, eps=args.eps/255., loss=args.apgd_loss)
    if args.mode in ['sketch', 'all']:
        test_imagenet_sketch(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet-sketch'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers)
    if args.mode in ['r', 'all']:
        test_imagenet_r(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet-r'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers)
    if args.mode in ['a', 'all']:
        test_imagenet_a(model=model, data_dir=os.path.join(args.data_root_path, 'imagenet-a'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers)
    if args.mode in ['c']:
        test_imagenet_c(model=model, data_dir=os.path.join(args.data_root_path, 'ImageNet-C'), fp=fp, ckpt_version=args.ckpt_version, 
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers)
       
    fp.close()
