
## To Reproduce Our Action Localization Results
### 1.Download Features and Annotations
<!-- * Download *mammalnet_fearure.tar.gz* from [[Google Drive]()] [[Amazon S3](https://s3.us-east-2.amazonaws.com/animal-net.com/mammalnet_feature.tar.gz)] [[百度网盘](https://pan.baidu.com/s/171Zd8E-qkoyLf70Wm19tSg) 提取码:yk0m] and [[mammalnet_detection_json]()]. -->
Download the mammalnet_features for localization
```bash
wget https://mammalnet.s3.amazonaws.com/full_video_features/mammalnet_feature.tar.gz
```

The annotation file can be downloaded from here:

```bash
wget https://mammalnet.s3.amazonaws.com/annotation.tar.gz
```

**Details**: The file includes I3D features and action annotations in json format (similar to ActivityNet annotation format). To produce the features for our MammalNet videos, we firstly format the all videos to 25 FPS, then finetune a two-stream I3D model, that is originally pretrained on ImageNet and Kinetics 400, on our dataset, and finally extract the RGB and optical flow features for each video. We concatenate these two features together as the model input. Feature extraction can refer to [mmaction2](https://github.com/open-mmlab/mmaction2) and [I3D Feature Extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction).

* The feature folder structure is
```
Feature folder
└───mammalnet_feature/
│    └───RGB_feature/
│    │	 └───SMN6WFVy-Ys.npy
│    │	 └───Di4eEBZjkA4.npy   
│    │	 └───...
│    └───Flow_feature/
│    │	 └───SMN6WFVy-Ys.npy
│    │	 └───Di4eEBZjkA4.npy   
│    │	 └───...
│    └───Concatenate_feature/
│    │	 └───SMN6WFVy-Ys.npy
│    │	 └───Di4eEBZjkA4.npy   
│    │	 └───...
```


### 2.Model Code
* We trained all detection models using their officially released code: [ActionFormer](https://github.com/happyharrycn/actionformer_release), [TAGS](https://github.com/sauradip/TAGS), and [CoLA](https://github.com/zhang-can/CoLA).
* You only need to convert our annotation json and set the feature path based on the official folder and annotation file configuration.

<!-- ### 3.Training Details 
* For training ActionFormer model, we apply the base learning rate 0.001, cosine decay learning rate scheduler, 30 training epochs, 5 warmup epochs, and the batch size 16. 
* For training TAGS model, we apply the base learning rate of 0.0004, step decay learning rate scheduler, 20 training epochs, and the batch size 200. 
* For training CoLA model, we apply the base learning rate of 0.0001, 50 training epochs, and the batch size 256.  -->
