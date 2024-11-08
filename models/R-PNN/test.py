import argparse
import gc
import os

import numpy as np
import scipy.io as io
import torch
from torch import nn

from tqdm import tqdm

from network import R_PNN_model
from loss import SpectralLoss, StructuralLoss

from tools.spectral_tools import gen_mtf, normalize_prisma, denormalize_prisma

from dataset import open_mat
from config_dict import config
from tools.cross_correlation import local_corr_mask


def test_r_pnn(args):

    # Paths and env configuration
    basepath = args.input
    method = 'R-PNN'
    out_dir = os.path.join(args.out_dir, method)

    gpu_number = args.gpu_number
    use_cpu = args.use_cpu

    # Training hyperparameters

    if args.learning_rate != -1:
        learning_rate = args.learning_rate
    else:
        learning_rate = config['learning_rate']

    # Satellite configuration
    sensor = config['satellite']
    ratio = config['ratio']

    # Environment Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # Devices definition
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

    if sensor == 'PRISMA':
        normalize = normalize_prisma
        denormalize = denormalize_prisma
    else:
        raise 'Satellite not supported'

    # Open the image

    pan, ms_lr, ms, _, wl = open_mat(basepath)

    ms_lr = normalize(ms_lr)
    ms = normalize(ms)
    pan = normalize(pan).to(device)

    net_scope = config['net_scope']
    pad = nn.ReflectionPad2d(net_scope)

    # Torch configuration
    net = R_PNN_model()
    net.load_state_dict(torch.load(os.path.join('weights', 'R-PNN_' + sensor + '.tar')))
    net = net.to(device)

    criterion_spec = SpectralLoss(gen_mtf(ratio, sensor, kernel_size=61, nbands=1), ratio, device).to(device)
    criterion_struct = StructuralLoss(ratio).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    history_loss_spec = []
    history_loss_struct = []

    alpha = config['alpha_1']

    fused = []

    for band_number in range(ms.shape[1]):

        band = ms[:, band_number:band_number + 1, :, :].to(device)
        band_lr = ms_lr[:, band_number:band_number + 1, :, :].to(device)

        # Aux data generation
        inp = torch.cat([band, pan], dim=1)
        inp = pad(inp)

        threshold = local_corr_mask(inp, ratio, sensor, device, config['semi_width'])

        if wl[band_number] > 700:
            alpha = config['alpha_2']

        if band_number == 0:
            ft_epochs = config['first_iter']
        else:
            ft_epochs = int(min(((wl[band_number].item() - wl[band_number - 1].item()) // 10 + 1) * config['epoch_nm'], config['sat_val']))

        min_loss = torch.inf
        print('Band {} / {}'.format(band_number + 1, ms.shape[1]))
        pbar = tqdm(range(ft_epochs))

        for epoch in pbar:

            pbar.set_description('Epoch %d/%d' % (epoch + 1, ft_epochs))

            net.train()
            optim.zero_grad()

            outputs = net(inp)

            loss_spec = criterion_spec(outputs, band_lr)
            loss_struct, loss_struct_without_threshold = criterion_struct(outputs,
                                                                          pan,
                                                                          threshold[:, :, net_scope:-net_scope, net_scope:-net_scope])

            loss = loss_spec + alpha * loss_struct

            loss.backward()
            optim.step()

            running_loss_spec = loss_spec.item()
            running_loss_struct = loss_struct_without_threshold

            history_loss_spec.append(running_loss_spec)
            history_loss_struct.append(running_loss_struct)

            if loss.item() < min_loss:
                min_loss = loss.item()
                if not os.path.exists('temp'):
                    os.makedirs(os.path.join('temp'))
                torch.save(net.state_dict(), os.path.join('temp', 'R-PNN_best_model.tar'))

            pbar.set_postfix(
                {'Spec Loss': running_loss_spec, 'Struct Loss': running_loss_struct})

        net.load_state_dict(torch.load(os.path.join('temp', 'R-PNN_best_model.tar')))

        net.eval()
        fused.append(net(inp).detach().cpu())

    fused = torch.cat(fused, 1)
    fused = denormalize(np.moveaxis(torch.squeeze(fused, 0).numpy(), 0, -1))
    fused = np.round(fused)
    fused = np.clip(fused, 0, 2 ** 16)
    fused = fused.astype(np.uint16)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(out_dir, basepath.split(os.sep)[-1].split('.')[0] + '_R-PNN.mat')
    io.savemat(save_path, {'I_MS': fused})
    history = {'loss_spec': history_loss_spec, 'loss_struct': history_loss_struct}
    io.savemat(os.path.join(out_dir, basepath.split(os.sep)[-1].split('.')[0] + '_R-PNN_stats.mat'), history)

    torch.cuda.empty_cache()
    gc.collect()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='R-PNN Training code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='R-PNN is an unsupervised deep learning-based pansharpening '
                                                 'method.',
                                     epilog='''\
Reference: 
Band-wise Hyperspectral Image Pansharpening using CNN Model Propagation
G. Guarino, M. Ciotola, G. Vivone, G. Scarpa 

Authors: 
- Image Processing Research Group of University of Naples Federico II ('GRIP-UNINA')
- National Research Council, Institute of Methodologies for Environmental Analysis (CNR-IMAA)
- University of Naples Parthenope

For further information, please contact the first author by email: giuseppe.guarino2[at]unina.it '''
                                     )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required named arguments')

    required.add_argument("-i", "--input", type=str, required=True,
                          help='The path of the .mat file'
                               'For more details, please refer to the GitHub documentation.')

    optional.add_argument("-o", "--out_dir", type=str, default='Outputs',
                          help='The directory in which save the outcome.')

    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')

    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    test_r_pnn(arguments)
