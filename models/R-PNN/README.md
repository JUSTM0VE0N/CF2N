# Band-wise Hyperspectral Image Pansharpening using CNN Model Propagation

[Band-wise Hyperspectral Image Pansharpening using CNN Model Propagation](https://ieeexplore.ieee.org/document/10341305) (  [ArXiv](https://www.google.com/url?q=http://arxiv.org/abs/2311.06510&source=gmail&ust=1700026852089000&usg=AOvVaw0bA5hS9pYGvnMPpmkPF2LT) ) introduces **Rolling Pansharpening Neural Network** (R-PNN), a Deep Learning method that use a simple CNN model propagation strategy for Hyperspectral Pansharpening. Starting from an initial weights configuration, each band is pansharpened refining the model tuned on the preceding one. In this way, R-PNN is able to work with hyperspectral images with an arbitrary number of bands. The proposed method has been tested on real hyperspectral images (PRISMA Dataset), both Full and Reduced resoution and then compared with several Pansharpening methods, both model-based and Deep-Learning.

## Cite R-PNN

If you use R-PNN in your research, please use the following BibTeX entry.

```
@ARTICLE{10341305,
  author={Guarino, Giuseppe and Ciotola, Matteo and Vivone, Gemine and Scarpa, Giuseppe},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Band-wise Hyperspectral Image Pansharpening using CNN Model Propagation}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3339337}}
```

## Team members

*   Giuseppe Guarino (giuseppe.guarino2@unina.it);

*   Matteo Ciotola (matteo.ciotola@unina.it);

*   Gemine Vivone;

*   Giuseppe Scarpa  (giuseppe.scarpa@uniparthenope.it).

## License

Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/giu-guarino/R-PNN/blob/main/LICENSE.txt)
(included in this package)

## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

*   Python 3.10.10
*   PyTorch 2.0.0
*   Cuda 11.7 or 11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the algorithm
*   Download the algorithm and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/giu-guarino/R-PNN

*   Create the virtual environment with the `r_pnn_env.yaml`

<!---->

    conda env create -n r_pnn_env -f r_pnn_env.yaml

*   Activate the Conda Environment

<!---->

    conda activate r_pnn_env

*   Test it

<!---->

    python test.py -i example/PRISMA_example.mat -o ./Output_folder/ 

## Usage

### Before to start

The easiest way for testing this algorithm is to create a `.mat` file. It must contain:

*   `I_MS_LR`: Original Hyperspectral Stack in channel-last configuration (Dimensions: H x W x B);
*   `I_MS`: Upsampled version of original Hyperspectral Stack in channel-last configuration (Dimensions: HR x WR x B);
*   `I_PAN`: Original Panchromatic band, without the third dimension (Dimensions: HR x WR).
*   `Wavelengths`: Array of wavelengths (Dimensions: B x 1)

where R is the ratio of the sensor.

Please refer to `--help` for more details.

### Testing

This project provide a set of weights obtained from a pre-training procedure described in the paper. You can directly start from this configuration using the command:

    python test.py -i path/to/file.mat
    
where `path/to/file.mat` can be any dataset (with any number of bands) organized as described before.

Several options are possible. Please refer to the parser help for more details:

    python test.py -h

### Training

If you want to train this network by yourself, you can do it providing in input the path of a `Dataset` folder that has to contain two subfolders: `Training` and `Validation`.


## Dataset

You can find the dataset used in this work at this [link](https://openremotesensing.net/knowledgebase/panchromatic-and-hyperspectral-image-fusion-outcome-of-the-2022-whispers-hyperspectral-pansharpening-challenge/)
