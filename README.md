# Multiscale-high-frequency-signal-enhancement-super-resolution
We use information distillation for multi-scale high-frequency signal enhancement to achieve real-time image super-resolution.
* Multi-scale High-Frequency Signal Enhancement Super-Resolution model structure diagram<br>
![image](https://github.com/ChengChen-ai/Multiscale-high-frequency-signal-enhancement-super-resolution/blob/main/images/%E8%B6%85%E5%88%86%E8%BE%A8%E7%8E%87.png)


## Environment
* Python 3.6 <br>
* PyTorch 1.8.0 <br>
* CUDA 11.1 <br>
* Ubuntu 20.04 <br>

## Datasets
We use DIV2K[Retinex](https://data.vision.ee.ethz.ch/cvl/DIV2K) datasets for model training and validation

## Training
The downloaded training data is placed in the following file  
>datasets
>>trainData  

    python ./train.py


## Testing
The pre-trained models are placed in the following file
>pretrained_model  
>>Super_Resolution
>>>models
>>>>MMAFNet

The testing data is placed in the following file  
>datasets
>>testData

    python ./test.py  

* Test Results  
![image](https://github.com/ChengChen-ai/Multiscale-high-frequency-signal-enhancement-super-resolution/blob/main/datasets/testData/0915.png)  
![image](https://github.com/ChengChen-ai/Multiscale-high-frequency-signal-enhancement-super-resolution/blob/main/results/test/0915.png)
## Acknowledgments
Code is inspired by [Retinex](https://github.com/Zheng222/IMDN) and [CycleGAN](https://data.vision.ee.ethz.ch/cvl/DIV2K).
