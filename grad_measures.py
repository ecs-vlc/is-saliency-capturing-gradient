import glob
import sys
from functools import partial

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, resnet18
from tqdm import tqdm

import utils.modelfitting as modelfitting
from single_pixel_experiments import load_data
from utils.modelfitting import get_device
from datasets.singlepixel import SinglePixelDataset, MalhotraSinglePixelDataset, PixelClusterDataset


def compute_grad(input, output, y_true=None, softmax=True):
    if softmax and output.shape[1] != 1:
        output = torch.softmax(output, dim=1)

    # select the "biggest" output
    if y_true is None:
        if output.shape[1] == 1:
            idx = 0
            out = output[:, idx]
        else:
            out = output.max(dim=1)[0]
    else:
        out = output.gather(1, y_true.unsqueeze(1)).squeeze(1)

    return torch.autograd.grad(out, input, grad_outputs=torch.ones(output.shape[0], device=input.device),
                               create_graph=True)[0].detach()


class PixInfoDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ret = *self.dataset[idx], self.dataset.diag_pixels[idx]

        cond = self.dataset.condition
        self.dataset.condition = 'nopix'
        nopix = self.dataset[idx][0]
        self.dataset.condition = cond

        return *ret, nopix, idx


def predict_importance_grad(loader, model, device, use_y_true=False, softmax=False):
    def fcn(input, output, y_true):
        return compute_grad(input, output, y_true if use_y_true else None, softmax).mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance_inp(loader, model, device, use_y_true=False, softmax=False):
    def fcn(input, output):
        return input.mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance_absinp(loader, model, device, use_y_true=False, softmax=False):
    def fcn(input, output):
        return input.abs().mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance_integrated_gradient(loader, model, device, steps=100, mode='black', use_y_true=True, softmax=True):
    def fcn(input, output, y_true):
        init = None
        if mode == 'zeros':
            init = torch.zeros_like(input)
        if mode == 'black':
            init = (torch.zeros_like(input) - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        if mode == 'white':
            init = (torch.ones_like(input) - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        output_idx = None
        if not use_y_true:
            # use the target based on the first prediction
            output_idx = output.argmax(dim=1)

        grads = torch.zeros_like(input)
        for i in range(steps):
            inp = torch.lerp(init, input, i/(steps-1))
            inp.detach().requires_grad = True
            output = model(inp)  # rerun model
            grads += compute_grad(inp, output, y_true if use_y_true else output_idx, softmax) / steps
        return ((input - init) * grads).mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance_absgrad(loader, model, device, use_y_true=False, softmax=False):
    def fcn(input, output, y_true):
        return compute_grad(input, output, y_true if use_y_true else None, softmax).abs().mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance_grad_inp(loader, model, device, use_y_true=False, softmax=False):
    def fcn(input, output, y_true):
        return (compute_grad(input, output, y_true if use_y_true else None, softmax) * input).mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance_absgrad_absinp(loader, model, device, use_y_true=False, softmax=False):
    def fcn(input, output, y_true):
        return (compute_grad(input, output, y_true if use_y_true else None, softmax).abs() * input.abs()).mean(dim=1)

    return predict_importance(loader, model, device, fcn)


def predict_importance(loader, model, device, function):
    model = model.to(device)
    model.eval()

    loader = DataLoader(PixInfoDatasetWrapper(loader.dataset), batch_size=256, shuffle=False, drop_last=False)

    stats = {'overall.withpix_model_accuracy': 0,
             'overall.attribution_correct_accuracy': 0,
             'overall.nopix_model_accuracy': 0,
             'overall.attribution_incorrect': 0,
             'overall.truebg_withblackpix_model_accuracy': 0,
             'overall.importance_maps': [],
             'attribution_incorrect.nopix_model_accuracy': 0,
             'attribution_incorrect.withpix_model_accuracy': 0,
             'attribution_incorrect.blackbg_withpix_model_accuracy': 0,
             'attribution_incorrect.blackbg_withattrpix_model_accuracy': 0,
             'attribution_incorrect.truebg_withblackattrpix_model_accuracy': 0,
             'attribution_incorrect.truebg_withblackpix_model_accuracy': 0,
             'attribution_incorrect.truebg_withblackattrpix_preserved_accuracy': 0,
             'attribution_incorrect.truebg_withblackpix_preserved_accuracy': 0,
             'attribution_incorrect.pixel_info': []}
    for x, y, pixinfo, nopix, idx in tqdm(loader):
        x = x.to(device)
        x.requires_grad = True
        y = y.to(device)

        # Compute the prediction and importance map
        pred = model(x)
        importance_map = function(x, pred, y)
        stats['overall.importance_maps'].append(importance_map)

        # compute prediction without the pixel
        nopix_pred = model(nopix.to(device))

        # max-importance maps (False everywhere, except True in the position of most important)
        gmax = torch.stack([(importance_map[i] == torch.max(importance_map[i]))
                            for i in range(importance_map.shape[0])], dim=0)

        # Make a black input and set the true predictive pixel to its value & make the prediction
        zeroed_x = (torch.zeros_like(x) - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        for i in range(importance_map.shape[0]):
            zeroed_x[i, :, pixinfo[i, 0], pixinfo[i, 1]] = x[i, :, pixinfo[i, 0], pixinfo[i, 1]]
        pred_black_truepixel = model(zeroed_x)

        # Make a black input, set the predicted most important pixel to its value & make the prediction
        zeroed_x = (torch.zeros_like(x) - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        mask = gmax.unsqueeze(1).repeat(1, 3, 1, 1)
        pred_black_predpixel = model(x * mask + zeroed_x * ~mask)

        # Make a copy of the input and set the predicted most important pixel to black & make the prediction
        pred_predpixel_black = model(x * ~mask + zeroed_x * mask)

        # Make a copy of the input and set the discriminative pixel to black & make the prediction
        zeroed_x = (torch.zeros_like(x) - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        truepixel_black = x.clone()
        for i in range(importance_map.shape[0]):
            truepixel_black[i, :, pixinfo[i, 0], pixinfo[i, 1]] = zeroed_x[i, :, pixinfo[i, 0], pixinfo[i, 1]]
        pred_truepixel_black = model(truepixel_black)

        # Accumulate statistics
        for i in range(importance_map.shape[0]):
            # model prediction accuracy
            if pred[i].argmax() == y[i]:
                stats['overall.withpix_model_accuracy'] += 1

            if nopix_pred[i].argmax() == y[i]:
                stats['overall.nopix_model_accuracy'] += 1

            if pred_truepixel_black[i].argmax() == y[i]:
                stats['overall.truebg_withblackpix_model_accuracy'] += 1

            # is the predicted important pixel the true one? accumulate stats related to this
            if gmax[i, pixinfo[i][0], pixinfo[i][1]]:
                stats['overall.attribution_correct_accuracy'] += 1
            else:
                stats['overall.attribution_incorrect'] += 1  # no incorrect

                rc = torch.nonzero(gmax[i])[0]
                pv = x[i, :, rc[0], rc[1]].detach()
                pv *= torch.tensor([0.229, 0.224, 0.225], device=device)
                pv += torch.tensor([0.485, 0.456, 0.406], device=device)
                pv *= 255
                stats['attribution_incorrect.pixel_info'].append({'image': idx[i].item(),
                                                                  'predicted_pixel': torch.cat((rc, pv)).int(),
                                                                  'true_pixel': pixinfo[i]})

                # was the true pixel important? look at acc with and without it
                if nopix_pred[i].argmax() == y[i]:
                    stats['attribution_incorrect.nopix_model_accuracy'] += 1
                if pred[i].argmax() == y[i]:
                    stats['attribution_incorrect.withpix_model_accuracy'] += 1

                # was the predicted pixel important by itself on a black bg?
                if pred_black_predpixel[i].argmax() == y[i]:
                    stats['attribution_incorrect.blackbg_withattrpix_model_accuracy'] += 1

                # was the true pixel important by itself on a black bg?
                if pred_black_truepixel[i].argmax() == y[i]:
                    stats['attribution_incorrect.blackbg_withpix_model_accuracy'] += 1

                # TODO: was the predicted pixel important by itself by setting it to black
                # did the classification change when we set the predicted pixel to black?
                if pred_predpixel_black[i].argmax() == y[i]:
                    stats['attribution_incorrect.truebg_withblackattrpix_model_accuracy'] += 1

                if pred_truepixel_black[i].argmax() == y[i]:
                    stats['attribution_incorrect.truebg_withblackpix_model_accuracy'] += 1

                if pred_predpixel_black[i].argmax() == pred[i].argmax():
                    stats['attribution_incorrect.truebg_withblackattrpix_preserved_accuracy'] += 1

                if pred_truepixel_black[i].argmax() == pred[i].argmax():
                    stats['attribution_incorrect.truebg_withblackpix_preserved_accuracy'] += 1

    stats['overall.withpix_model_accuracy'] /= len(loader.dataset)
    stats['overall.nopix_model_accuracy'] /= len(loader.dataset)
    stats['overall.attribution_correct_accuracy'] /= len(loader.dataset)
    stats['overall.truebg_withblackpix_model_accuracy'] /= len(loader.dataset)

    stats['overall.attribution_incorrect'] += 1e-10
    stats['attribution_incorrect.nopix_model_accuracy'] /= stats['overall.attribution_incorrect']
    stats['attribution_incorrect.withpix_model_accuracy'] /= stats['overall.attribution_incorrect']

    stats['attribution_incorrect.blackbg_withattrpix_model_accuracy'] /= stats['overall.attribution_incorrect']
    stats['attribution_incorrect.blackbg_withpix_model_accuracy'] /= stats['overall.attribution_incorrect']
    stats['attribution_incorrect.truebg_withblackattrpix_model_accuracy'] /= stats['overall.attribution_incorrect']
    stats['attribution_incorrect.truebg_withblackpix_model_accuracy'] /= stats['overall.attribution_incorrect']
    stats['attribution_incorrect.truebg_withblackattrpix_preserved_accuracy'] /= stats['overall.attribution_incorrect']
    stats['attribution_incorrect.truebg_withblackpix_preserved_accuracy'] /= stats['overall.attribution_incorrect']

    stats['overall.attribution_incorrect'] -= 1e-10
    stats['overall.importance_maps'] = torch.cat(stats['overall.importance_maps'])

    return stats


def predict_importance_integrated_gradient_white(loader, model, device, steps=100, use_y_true=True, softmax=True):
    return predict_importance_integrated_gradient(loader, model, device, steps=100, use_y_true=use_y_true,
                                                  softmax=softmax, mode="white")


if __name__ == '__main__':
    model_dir = sys.argv[1]

    attribution_methods = [predict_importance_absgrad,
                           predict_importance_grad_inp,
                           predict_importance_absgrad_absinp,
                           predict_importance_integrated_gradient,
                           predict_importance_integrated_gradient_white]

    data_root = "./data"
    output_file = "attribution_results.csv"

    use_y_true = [False, True]
    softmax = [True, False]
    test_stdevs = [(0., 0.), (0., 5.), (3., 5.), (3., 3.), (1., 3.)]

    modelfitting.FORCE_MPS = True
    device = get_device('auto')
    print(device)

    parts = model_dir.split("/")[-1]
    if len(parts) == 0:
        parts = model_dir.split("/")[-2]
    parts = parts.split("-")
    mt = parts[0]
    seed = int(parts[2].replace("seed_", ""))
    data_seed = int(parts[3].replace("dataseed_", ""))

    train_pos_sd = float(parts[4].replace("pos_sd_", ""))
    train_col_sd = float(parts[5].replace("col_sd_", ""))
    if "SinglePixelDataset" in model_dir:
        dsc = SinglePixelDataset
    elif "PixelClusterDataset" in model_dir:
        dsc = PixelClusterDataset
    else:
        dsc = MalhotraSinglePixelDataset

    all_results = []
    model_path = f"{model_dir}/models/*_last.pt"
    model_files = glob.glob(model_path)
    if len(model_files) == 0:
        print("Skipping", model_path)
        sys.exit(0)
    assert len(model_files) == 1
    model_file = model_files[0]
    print("Probing", model_file)

    # Load the model
    if mt == 'vgg16':
        model = vgg16(num_classes=10)
    else:
        model = resnet18(num_classes=10)
        # TODO make this not assume 3x3
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.load_state_dict(torch.load(model_file, map_location="cpu")['model'])

    for test_pos_sd, test_col_sd in test_stdevs:
        data = load_data(data_root, 32, data_seed, 32,
                         0, test_pos_sd, test_col_sd, False, False,
                         dataset_cls=dsc)

        for f in attribution_methods:
            for uyt in use_y_true:
                for sm in softmax:
                    f2 = partial(f, use_y_true=uyt, softmax=sm)
                    res = f2(data['val_same'], model, device)

                    filtered_res = {k: v for k, v in res.items() if "accuracy" in k}
                    filtered_res['overall.attribution_incorrect'] = res['overall.attribution_incorrect']
                    filtered_res['dataset'] = dsc.__name__
                    filtered_res['data_seed'] = data_seed
                    filtered_res['model_type'] = mt
                    filtered_res['model_seed'] = seed
                    filtered_res['train_pos_sd'] = train_pos_sd
                    filtered_res['train_col_sd'] = train_col_sd
                    filtered_res['test_pos_sd'] = test_pos_sd
                    filtered_res['test_col_sd'] = test_col_sd
                    filtered_res['attribution_method'] = f.__name__
                    filtered_res['use_y_true'] = uyt
                    filtered_res['use_softmax'] = sm
                    filtered_res['model'] = model_file

                    all_results.append(filtered_res)

                df = pd.DataFrame.from_records(all_results)
                df.to_csv(f"{model_dir}/{output_file}", index=False)
