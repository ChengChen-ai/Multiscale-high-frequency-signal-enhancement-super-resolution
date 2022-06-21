# Multiscale-high-frequency-signal-enhancement-super-resolution
We use information distillation for multi-scale high-frequency signal enhancement to achieve real-time image super-resolution.
* Multi-scale High-Frequency Signal Enhancement Super-Resolution model structure diagram<br>
![image](https://github.com/ChengChen-ai/Multiscale-high-frequency-signal-enhancement-super-resolution/tree/main/images/超分辨率.png)


## Environment
* Python 3.6 <br>
* PyTorch 1.8.0 <br>
* CUDA 11.1 <br>
* Ubuntu 20.04 <br>

## Datasets
Baidu network disk：https://pan.baidu.com/s/1p2hlvfoi4FXi74Ar2qPfhA 
Extraction code：CcSs  
Note: the data contains paired training sets and pre-trained models

## Training
The downloaded training data is placed in the following file  
>data
>>images  
>>labels

    python ./train.py


## Testing
The pre-trained models are placed in the following file
>results  

The testing data is placed in the following file  
>data
>>test
>>>images  

    python ./test.py  

* Test Results  
![image](https://github.com/ChengChen-ai/Sky-Segmentation/blob/main/data/MAG/test_1.jpg)  
![image](https://github.com/ChengChen-ai/Sky-Segmentation/blob/main/data/MAG/test_2.jpg)
## Acknowledgments
Code is inspired by [Retinex](https://github.com/weichen582/RetinexNet) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
