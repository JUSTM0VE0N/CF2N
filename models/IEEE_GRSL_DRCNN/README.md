# IEEE_GRSL_DRCNN

## [Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening](https://ieeexplore.ieee.org/document/8874962)

## [Hyperspectral Pansharpening Based on Guided Filter and Deep Residual Learning](https://ieeexplore.ieee.org/document/8899015)

**Tensorflow implementation of our proposed DRCNN method for hyperspectral pansharpening.**

![Framework](https://github.com/yxzheng24/IEEE_GRSL_DRCNN/blob/main/Framework_GRSL20.png "Framework of the proposed method for hyperspectral pansharpening.")

## Usage
Here we take the experiments conducted on the [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/) data set as an example for illustration.

*   Training:
1.   Download the [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/) data set and put the data into the __./data_process/cave_ref/__ folder, which are served as the reference images. Note that you can also download the CAVE data set (in .mat format) from the Baidu Cloud links: https://pan.baidu.com/s/1tmuanIjXLXTEdaAsRxq1fw (Access Code: 3e53).
2.   Run *"get_hspan_cave.m"* to obtain LR-HSI and HR-PAN images.
3.   Run *"get_Hini_Hres_cave.m"* to genertate the initialized HSI and the residual HSI.
4.   Randomly select 22 HSI pairs from __./data_process/cave_Hini/__ and __./data_process/cave_Hres/__ folders to form the training set.
5.   Run *"get_traindata_h5.m"* to produce the HDF5 file for training.
6.   Run *"train_DRCNN.py"*.

*   Testing: 
    
    Run *"test_DRCNN.py"* by utilizing the pretrained model __./models/model_cave.h5__ to obtain the fused HSIs.

    Note that due to the large size of testing images, you may need to first manually download the 10 initialized HSIs (*.mat files) to your local folder under path __./data_process/cave_Hini/test_10Hini/__ by the given links of Baidu Cloud below.

    Baidu Cloud links: https://pan.baidu.com/s/1fdy4SBryu6I8o-rt_1jZuA
    
    Access Code: c838

## Requirements
Latest version was tested on Ubuntu 16.04, using Python 3.6.10, Tensorflow 1.10.0, Keras 2.2.4 and Matlab R2017a.

## Citation
If you find this code helpful, please kindly cite the following papers:

(1) Y. Zheng, J. Li, Y. Li, K. Cao and K. Wang, "Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening," IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 8, pp. 1435-1439, Aug. 2020, doi: 10.1109/LGRS.2019.2945424.

(2) Y. Zheng, J. Li and Y. Li, "Hyperspectral Pansharpening Based on Guided Filter and Deep Residual Learning," 2019 IEEE International Geoscience and Remote Sensing Symposium, Jul. 2019, pp. 616-619, doi: 10.1109/IGARSS.2019.8899015.

    @ARTICLE{Zheng2020GRSL,
    author={Y. {Zheng} and J. {Li} and Y. {Li} and K. {Cao} and K. {Wang}},
    journal={IEEE Geosci. Remote Sens. Lett.}, 
    title={Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening}, 
    year={2020},
    month={Aug.},
    volume={17},
    number={8},
    pages={1435-1439},
    doi={10.1109/LGRS.2019.2945424}}
    
    @INPROCEEDINGS{Zheng2019IGARSS,
    author={Zheng, Yuxuan and Li, Jiaojiao and Li, Yunsong},
    booktitle={2019 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2019)}, 
    title={Hyperspectral Pansharpening Based on Guided Filter and Deep Residual Learning}, 
    year={2019},
    month={Jul.},
    pages={616-619},
    doi={10.1109/IGARSS.2019.8899015}}

## Contact Information
If you have any problem, please do not hesitate to contact Yuxuan Zheng (e-mail: yxzheng24@163.com).

Yuxuan Zheng is with the State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an 710071, China.
