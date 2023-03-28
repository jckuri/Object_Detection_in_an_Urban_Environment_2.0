# Object Detection in an Urban Environment 2.0

In this project, you will learn how to train an object detection model using the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and [AWS Sagemaker](https://aws.amazon.com/sagemaker/).

This link has the former documentation: [FORMER_README.md](FORMER_README.md)

In this current documentation, we will focus on the rubric.

# RUBRIC:

You can click on the [rubric for the project Object Detection in an Urban Environment 2.0](https://review.udacity.com/#!/rubrics/5089/view).

## [Model training and Evaluation]

## 1. Test at least two pretrained models (other than EfficientNet)

> * Tried two models other than EfficientNet.
> * Update and submit the `pipeline.config` file and notebooks associated with all the pretrained models.

I tried 2 models other than EfficientNet:
1. SSD MobileNet V2 FPNLite 640x640 [[Pretrained Model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)] [[Jupyter Notebook](1_model_training_SSD_MobileNet/1_train_model_SSD_MobileNet.ipynb)] [[pipeline.config](1_model_training_SSD_MobileNet/source_dir/pipeline.config)] [[Training Directory](1_model_training_SSD_MobileNet/)]
2. SSD ResNet50 V1 FPN 640x640 (RetinaNet50) [[Pretrained Model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)] [[Jupyter Notebook](1_model_training_RetinaNet50/1_train_model_RetinaNet50.ipynb)] [[pipeline.config](1_model_training_RetinaNet50/source_dir/pipeline.config)] [[Training Directory](1_model_training_RetinaNet50/)]

Helpful links:
- []()
- []()

## 2. Choosing the best model for deployment

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

**Baseline Experiment - EfficientDet D1 640x640**

```
I0323 02:05:31.501080 140013260003136 model_lib_v2.py:1015] Eval metrics at step 2000
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP: 0.079465
I0323 02:05:31.514246 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP: 0.079465
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP@.50IOU: 0.199936
I0323 02:05:31.515697 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP@.50IOU: 0.199936
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP@.75IOU: 0.048818
I0323 02:05:31.517093 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP@.75IOU: 0.048818
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (small): 0.034636
I0323 02:05:31.518450 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (small): 0.034636
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (medium): 0.309154
I0323 02:05:31.519841 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (medium): 0.309154
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (large): 0.241418
I0323 02:05:31.521204 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (large): 0.241418
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@1: 0.020105
I0323 02:05:31.522563 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@1: 0.020105
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@10: 0.090442
I0323 02:05:31.524003 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@10: 0.090442
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100: 0.126141
I0323 02:05:31.525356 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100: 0.126141
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (small): 0.068426
I0323 02:05:31.526715 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (small): 0.068426
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (medium): 0.441800
I0323 02:05:31.528115 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (medium): 0.441800
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (large): 0.344867
I0323 02:05:31.529497 140013260003136 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (large): 0.344867
INFO:tensorflow:#011+ Loss/localization_loss: 0.024434
I0323 02:05:31.530541 140013260003136 model_lib_v2.py:1018] #011+ Loss/localization_loss: 0.024434
INFO:tensorflow:#011+ Loss/classification_loss: 0.441609
I0323 02:05:31.531654 140013260003136 model_lib_v2.py:1018] #011+ Loss/classification_loss: 0.441609
INFO:tensorflow:#011+ Loss/regularization_loss: 0.030694
I0323 02:05:31.532732 140013260003136 model_lib_v2.py:1018] #011+ Loss/regularization_loss: 0.030694
INFO:tensorflow:#011+ Loss/total_loss: 0.496737
I0323 02:05:31.533815 140013260003136 model_lib_v2.py:1018] #011+ Loss/total_loss: 0.496737
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.049
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.090
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.126
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.068
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.345
```

![IMAGES/baseline_experiment/1_map_precision.png](IMAGES/baseline_experiment/1_map_precision.png)

![IMAGES/baseline_experiment/2_map_recall.png](IMAGES/baseline_experiment/2_map_recall.png)

![IMAGES/baseline_experiment/3_loss.png](IMAGES/baseline_experiment/3_loss.png)

![IMAGES/baseline_experiment/4_other.png](IMAGES/baseline_experiment/4_other.png)

![2_run_inference/output.gif](2_run_inference/output.gif)

**Experiment 1 - SSD MobileNet V2 FPNLite 640x640**

```
```

```
```

![IMAGES/experiment1_SSD_MobileNet/1_map_precision.png](IMAGES/experiment1_SSD_MobileNet/1_map_precision.png)

![IMAGES/experiment1_SSD_MobileNet/2_map_recall.png](IMAGES/experiment1_SSD_MobileNet/2_map_recall.png)

![IMAGES/experiment1_SSD_MobileNet/3_loss.png](IMAGES/experiment1_SSD_MobileNet/3_loss.png)

![IMAGES/experiment1_SSD_MobileNet/4_other.png](IMAGES/experiment1_SSD_MobileNet/4_other.png)

![2_run_inference_SSD_MobileNet/output.gif](2_run_inference_SSD_MobileNet/output.gif)

**Experiment 2 - SSD ResNet50 V1 FPN 640x640 (RetinaNet50)**

```
```

```
```

![IMAGES/experiment2_RetinaNet50/1_map_precision.png](IMAGES/experiment2_RetinaNet50/1_map_precision.png)

![IMAGES/experiment2_RetinaNet50/2_map_recall.png](IMAGES/experiment2_RetinaNet50/2_map_recall.png)

![IMAGES/experiment2_RetinaNet50/3_loss.png](IMAGES/experiment2_RetinaNet50/3_loss.png)

![IMAGES/experiment2_RetinaNet50/4_other.png](IMAGES/experiment2_RetinaNet50/4_other.png)

![2_run_inference_RetinaNet50/output.gif](2_run_inference_RetinaNet50/output.gif)

## [Model Deployment]

## 3. Deploy the best model and run inference.

> * Deploy the best model in AWS by completing and `running 2_deploy_model.ipynb`.
> * Should be able to generate an output video with vehicle, pedestrian, and cyclist detections using the deployed model.


