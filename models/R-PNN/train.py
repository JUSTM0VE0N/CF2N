import argparse
import gc
import os
import numpy as np

import scipy.io as io
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from network import R_PNN_model
from loss import SpectralLoss, StructuralLoss

from tools.spectral_tools import gen_mtf, normalize_prisma

from dataset import TrainingDatasetFR
from config_dict import config
from tools.cross_correlation import local_corr_mask


def training_r_pnn(args):

    # Paths and env configuration
    basepath = args.input
    method = 'R-PNN'
    out_dir = os.path.join(args.out_dir, method)
    train_root = os.path.join(basepath, 'Training')
    val_root = os.path.join(basepath, 'Validation')

    gpu_number = args.gpu_number
    use_cpu = args.use_cpu

    # Training hyperparameters

    net_scope = config['net_scope']
    ms_scope = config['ms_scope']
    batch_sz = config['batch_size']

    epochs = args.epochs

    if args.learning_rate != -1:
        learning_rate = args.learning_rate
    else:
        learning_rate = config['learning_rate']

    if epochs == -1:
        epochs = config['epochs']

    # Satellite configuration
    sensor = config['satellite']
    ratio = config['ratio']

    # Environment Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # Devices definition
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

    # Data Management
    train_path = []
    val_path = []

    # Store the paths of the images for training set

    img_names = sorted(next(os.walk(train_root))[2])
    for i in range(len(img_names)):
        train_path.append(os.path.join(train_root, img_names[i]))

    # Store the paths of the images for validation set
    if config['validation']:
        img_names = sorted(next(os.walk(val_root))[2])
        for i in range(len(img_names)):
            val_path.append(os.path.join(val_root, img_names[i]))

    # Dataset definition
    dataset = TrainingDatasetFR(train_path, normalize_prisma)
    val_dataset = TrainingDatasetFR(val_path, normalize_prisma)

    # DataLoader definition
    train_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True, num_workers=8, pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)

    # Network definition
    net = R_PNN_model().to(device)

    # Optimizer definition
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(config['beta_1'], config['beta_2']))

    # Losses definition
    criterion_spec = SpectralLoss(gen_mtf(ratio, sensor, kernel_size=61, nbands=1), ratio, device).to(device)
    criterion_struct = StructuralLoss(ratio).to(device)

    # Best model path implementation
    custom_weights_path = config['save_weights_path']
    if not os.path.exists(custom_weights_path):
        os.mkdir(custom_weights_path)
    path_min_loss = os.path.join(custom_weights_path, 'weights_' + 'Training' + '_' + method + '_' + sensor + '.tar')

    # History variables initialization
    history_loss_spec = []
    history_loss_struct = []
    history_val_loss_spec = []
    history_val_loss_struct = []

    pbar = tqdm(range(epochs))

    min_loss = np.inf

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, epochs))
        running_loss_spec = 0.0
        running_loss_struct = 0.0
        running_val_loss_spec = 0.0
        running_val_loss_struct = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            pan, ms_lr, ms = data
            pan_band = pan.to(device)
            band = ms[:, 0:1, :, :].to(device)
            band_lr = ms_lr[:, 0:1, :, :].to(device)

            # Aux data generation

            inp = torch.cat([band, pan_band], dim=1)
            threshold = local_corr_mask(inp, ratio, sensor, device, config['semi_width'])

            outputs = net(inp)

            loss_spec = criterion_spec(outputs, band_lr[:, :, ms_scope:-ms_scope, ms_scope:-ms_scope])
            loss_struct, loss_struct_without_threshold = criterion_struct(outputs,
                                                                          pan_band[:, :, net_scope:-net_scope, net_scope:-net_scope],
                                                                          threshold[:, :, net_scope:-net_scope, net_scope:-net_scope])

            loss = loss_spec + config['alpha_1'] * loss_struct

            loss.backward()
            optimizer.step()

            running_loss_spec += loss_spec.item()
            running_loss_struct += loss_struct_without_threshold

        running_loss_spec = running_loss_spec / len(train_loader)
        running_loss_struct = running_loss_struct / len(train_loader)
        running_loss = running_loss_spec + config['alpha_1'] * running_loss_struct

        if running_loss < min_loss:
            torch.save(net.state_dict(), path_min_loss)
            min_loss = running_loss

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    pan, ms_lr, ms = data
                    pan = pan.to(device)
                    ms = ms[:, 0:1, :, :].to(device)
                    ms_lr = ms_lr[:, 0:1, :, :].to(device)

                    inp = torch.cat([ms, pan], dim=1)
                    threshold = local_corr_mask(inp, ratio, sensor, device,
                                                config['semi_width'])

                    outputs = net(inp)

                    val_loss_spec = criterion_spec(outputs, ms_lr[:, :, ms_scope:-ms_scope, ms_scope:-ms_scope])
                    _, val_loss_struct_without_threshold = criterion_struct(outputs,
                                                                            pan[:, :, net_scope:-net_scope, net_scope:-net_scope],
                                                                            threshold[:, :, net_scope:-net_scope, net_scope:-net_scope])

                    running_val_loss_spec += val_loss_spec.item()
                    running_val_loss_struct += val_loss_struct_without_threshold

            running_val_loss_spec = running_val_loss_spec / len(val_loader)
            running_val_loss_struct = running_val_loss_struct / len(val_loader)

        history_loss_spec.append(running_loss_spec)
        history_loss_struct.append(running_loss_struct)
        history_val_loss_spec.append(running_val_loss_spec)
        history_val_loss_struct.append(running_val_loss_struct)

        pbar.set_postfix(
            {'Spec Loss': running_loss_spec, 'Struct Loss': running_loss_struct, 'Val Spec Loss': running_val_loss_spec,
             'Val Struct Loss': running_val_loss_struct})

    history = {'loss_spec': history_loss_spec,
               'loss_struct': history_loss_struct,
               'val_loss_spec': history_val_loss_spec,
               'val_loss_struct': history_val_loss_struct
               }

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if config['save_training_stats']:
        io.savemat(os.path.join(out_dir, method + '_history' + '.mat'), history)

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
                          help='The path of the root containing Training and Validation folders. '
                               'For more details, please refer to the GitHub documentation.')

    optional.add_argument("-o", "--out_dir", type=str, default='Training',
                          help='The directory in which save the outcome.')

    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')

    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')

    optional.add_argument("--epochs", type=int, default=-1, help='Number of the epochs with which perform the '
                                                                 'training of the algorithm.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    training_r_pnn(arguments)
