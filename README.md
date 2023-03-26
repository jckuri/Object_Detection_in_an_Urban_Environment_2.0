# Object Detection in an Urban Environment 2.0

In this project, you will learn how to train an object detection model using the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and [AWS Sagemaker](https://aws.amazon.com/sagemaker/).

This link has the former documentation: [FORMER_README.md](FORMER_README.md)

In this current documentation, we will focus on the rubric.

# RUBRIC:

## [Model training and Evaluation]

## Test at least two pretrained models (other than EfficientNet)

> * Tried two models other than EfficientNet.
> * Update and submit the `pipeline.config` file and notebooks associated with all the pretrained models.

I tried 2 models other than EfficientNet:
1. SSD MobileNet V2 FPNLite 640x640 [[Pretrained Model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)] [[Jupyter Notebook](1_model_training_SSD_MobileNet/1_train_model_SSD_MobileNet.ipynb)] [[pipeline.config](1_model_training_SSD_MobileNet/source_dir/pipeline.config)] [[Training Directory](1_model_training_SSD_MobileNet/)]
2. SSD ResNet50 V1 FPN 640x640 (RetinaNet50) [[Pretrained Model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)] [[Jupyter Notebook](1_model_training_RetinaNet50/1_train_model_RetinaNet50.ipynb)] [[pipeline.config](1_model_training_RetinaNet50/source_dir/pipeline.config)] [[Training Directory](1_model_training_RetinaNet50/)]


## Choosing the best model for deployment

> Write a brief summary of your experiments and suggest the best model for this problem. This should include the accuracy (mAP) values of the models you tried. Also, discuss the following:
> 
> * How does the validation loss compare to the training loss?
> * Did you expect such behavior from the losses/metrics?
> * What can you do to improve the performance of the tested models further?

> [INSTRUCTIONS FROM THE JUPYTER NOTEBOOK]<br/>
> **Improve on the initial model**
> 
> Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the pipeline.config file to improve this model. One obvious change consists in improving the data augmentation strategy. The preprocessor.proto file contains the different data augmentation method available in the Tf Object Detection API. Justify your choices of augmentations in the writeup.
> 
> Keep in mind that the following are also available:
> 
> * experiment with the optimizer: type of optimizer, learning rate, scheduler etc
> * experiment with the architecture. The Tf Object Detection API model zoo offers many architectures. Keep in mind that the pipeline.config file is unique for each architecture and you will have to edit it.
> * visualize results on the test frames using the 2_deploy_model notebook available in this repository.
> 
> In the cell below, write down all the different approaches you have experimented with, why you have chosen them and what you would have done if you had more time and resources. Justify your choices using the tensorboard visualizations (take screenshots and insert them in your writeup), the metrics on the evaluation set and the generated animation you have created with this tool.

## [Model Deployment]

## Deploy the best model and run inference.

> * Deploy the best model in AWS by completing and `running 2_deploy_model.ipynb`.
> * Should be able to generate an output video with vehicle, pedestrian, and cyclist detections using the deployed model.


