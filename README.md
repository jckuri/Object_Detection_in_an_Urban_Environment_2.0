# Object Detection in an Urban Environment 2.0

In this project, you will learn how to train an object detection model using the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and [AWS Sagemaker](https://aws.amazon.com/sagemaker/).

This link has the former documentation: [FORMER_README.md](FORMER_README.md)

In this current documentation, we will focus on the rubric.

# RUBRIC:

## Model training and Evaluation

## Test at least two pretrained models (other than EfficientNet)

> * Tried two models other than EfficientNet.
> * Update and submit the `pipeline.config` file and notebooks associated with all the pretrained models.



## Choosing the best model for deployment

> Write a brief summary of your experiments and suggest the best model for this problem. This should include the accuracy (mAP) values of the models you tried. Also, discuss the following:

> * How does the validation loss compare to the training loss?
> * Did you expect such behavior from the losses/metrics?
> * What can you do to improve the performance of the tested models further?



## Model Deployment

## Deploy the best model and run inference.

> * Deploy the best model in AWS by completing and `running 2_deploy_model.ipynb`.
> * Should be able to generate an output video with vehicle, pedestrian, and cyclist detections using the deployed model.


