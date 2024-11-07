import os
import sys

import torchbearer
from torchbearer.callbacks import EarlyStopping, on_end_epoch, CSVLogger
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18, ResNet101_Weights, resnet101
from torchvision.models.vgg import vgg16, VGG16_Weights
from torch.utils.data import DataLoader

from datasets.singlepixel import MalhotraSinglePixelDataset, SinglePixelDataset, PixelClusterDataset
from utils.modelfitting import fit_model, evaluate_model, set_seed

import argparse
import torch.nn as nn
import torch


def load_data(data_dir, size, seed, batch_size, num_workers, pos_sd, col_sd, augment, presize, dataset_cls="MalhotraSinglePixelDataset"):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if augment:
        train_tf = transforms.Compose([transforms.ToTensor(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        train_tf = tf

    shared_args_ds = {'download': True, 'seed': seed, 'size': size, 'position_sd': pos_sd, 'colour_sd': col_sd,
                      'presize': presize}

    train_ds = dataset_cls(data_dir, train=True, transform=train_tf, **shared_args_ds)
    same_ds = dataset_cls(data_dir, train=False, condition='same', transform=tf, **shared_args_ds)
    diff_ds = dataset_cls(data_dir, train=False, condition='diff', transform=tf, **shared_args_ds)
    nopix_ds = dataset_cls(data_dir, train=False, condition='nopix', transform=tf, **shared_args_ds)
    train_nopix_ds = dataset_cls(data_dir, train=True, condition='nopix', transform=train_tf, **shared_args_ds)

    shared_args_loader = {'pin_memory': True, 'batch_size': batch_size, 'num_workers': num_workers}

    return {'train': DataLoader(train_ds, shuffle=True, **shared_args_loader),
            'val_same': DataLoader(same_ds, shuffle=False, **shared_args_loader),
            'val_diff': DataLoader(diff_ds, shuffle=False, **shared_args_loader),
            'val_nopix': DataLoader(nopix_ds, shuffle=False, **shared_args_loader),
            'train_nopix': DataLoader(train_nopix_ds, shuffle=False, **shared_args_loader)}


def val_callback(datasets):
    @on_end_epoch
    def cb(state):
        trial = torchbearer.Trial(state[torchbearer.MODEL], state[torchbearer.OPTIMIZER], state[torchbearer.CRITERION],
                                  metrics=state[torchbearer.METRIC_LIST].metric_list, verbose=2)

        for name, ds in datasets.items():
            if name.startswith('val_'):
                res = trial.with_val_generator(ds).to(state[torchbearer.DEVICE]).evaluate()
                for k, v in res.items():
                    if k.startswith('val_'):
                        nk = k.replace('val_', f'{name}_')
                        state[torchbearer.METRICS][nk] = v

    return cb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default='224', choices=[32, 224])
    parser.add_argument('--model', type=str, default='vgg16', choices=['vgg16', 'resnet18', 'resnet18_3x3', 'resnet101'])
    parser.add_argument('--pretrained', action="store_true", default=False)
    parser.add_argument('--freeze', action="store_true", default=False)
    parser.add_argument('--data-seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default="/scratch/jsh2/datasets")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--checkpoint-period', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--position-sd', type=float, default=1.0)
    parser.add_argument('--colour-sd', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--augment', action="store_true", default=False)
    parser.add_argument('--post-size', action="store_true", default=False)
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument("--dataset-class", type=str, default="MalhotraSinglePixelDataset",
                        choices=["MalhotraSinglePixelDataset", "SinglePixelDataset", "PixelClusterDataset"],)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model == 'vgg16':
        if args.pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
            net = vgg16(weights=weights)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 10)

            if args.freeze:
                for param in net.features:
                    param.requires_grad = False
        else:
            net = vgg16(num_classes=10)
    elif args.model == 'resnet18':
        if args.pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            net = resnet18(weights=weights)
            if args.freeze:
                for param in net:
                    param.requires_grad = False
            net.fc = nn.Linear(net.fc.in_features, 10)
        else:
            net = resnet18(num_classes=10)
    elif args.model == 'resnet18_3x3':
        if args.pretrained:
            raise Exception("Not Implemented")
        else:
            net = resnet18(num_classes=10)
            net.conv1 = nn.Conv2d(3, 64,
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif args.model == 'resnet101':
        if args.pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V1
            net = resnet101(weights=weights)
            if args.freeze:
                for param in net:
                    param.requires_grad = False
            net.fc = nn.Linear(net.fc.in_features, 10)
        else:
            net = resnet101(num_classes=10)
    else:
        sys.exit(f"Unsupported model {args.model}")

    loss = nn.CrossEntropyLoss()

    lr = 1e-5 if args.pretrained else 1e-3
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    data = load_data(args.data_dir, args.size, args.data_seed, args.batch_size, args.num_workers,
                     args.position_sd, args.colour_sd, args.augment, not args.post_size,
                     dataset_cls=globals()[args.dataset_class])

    ds = f"-{args.dataset_class}" if args.dataset_class != 'MalhotraSinglePixelDataset' else ''

    run_id = f"{args.model}-{args.size}{'-pre' if args.pretrained else ''}-seed_{args.seed}" + \
             f"-dataseed_{args.data_seed}-pos_sd_{args.position_sd}-col_sd_{args.colour_sd}" + \
             f"-momentum_{args.momentum}-wd_{args.weight_decay}{'-frozen' if args.freeze else ''}" + \
             f"{'-augment' if args.augment else ''}{'-post-size' if args.post_size else ''}" + \
             f"{ds}"

    if args.early_stop:
        extra_callbacks = [EarlyStopping(monitor='acc', patience=5, mode='max')]
    else:
        extra_callbacks = []

    if args.log_dir:
        args.log_dir = f"{args.log_dir}/{run_id}"
        models_dir = f"{args.log_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        model_file = models_dir + "/weights_{epoch:03d}.pt"
        extra_callbacks.append(CSVLogger(f"{args.log_dir}/log.csv"))

        torch.save({"model": net.state_dict()}, f"{models_dir}/weights_init.pt")
    else:
        model_file = None

    pre_extra_callbacks = [val_callback(data)]

    fit_model(net, loss, opt, data['train'], None, epochs=args.max_epochs, device='auto',
              verbose=2, acc='acc', model_file=model_file, run_id=run_id, log_dir=args.log_dir, resume=args.resume,
              pre_extra_callbacks=pre_extra_callbacks, extra_callbacks=extra_callbacks, period=args.checkpoint_period)
