# Pneumonia-Detection
Pneumonia Detection- Using Transfer learning to process chest X-ray scans

This project involves the use of Transfer learning to classify whether a patient is pneumonatic or not using his/her chest X-ray images. For this project I have implemented VGG19 and RESNET52 using transfer learning.
## DATASET
The datasets used for the pneumonia detection is provided as under:
1. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Transfer Learning
Transfer learning is a deep learning method where a model developed earlier is reused as the starting point for the model on second task, i.e. classifying the X-ray images as in our case; along with its pretrained weights. I have used the VGG19 and RESNET52 in our case. These models are extensively tested in the ImageNet 
classification event. In this event every year there are numerous models tested and the one with the best model is awarded. While, VGG16 and RESNET are one of the break through models in deep learning in their respective years for image processing. These are very efficient models. They have been trained over millions of images belonging to 1000 classes. (Link to ImageNet : https://www.image-net.org)
## Methodology
The models have been imported through keras.applications.resnet52 and keras.applications.vgg19 libraries respectively. As mentioned earlier, these models have been trained over millions of images belonging to 1000 classes. But we do not need the last dense (output) layer because we are working with only two classes- Pneumonia or Non-Pneumonia. Hence, we use an attribute during the respective constructor calling, known as the include_top attribute. This is assigned as false to exclude the output layer and then we include a layer according to our number of classes, here it is 2. An object of the class Model is created. The image data is augmented, since there is very limited resources available. Then the dataset is batch processed to make it ready to be fitted in the model. The model is fit through 10 epochs in my example.
## Results
My VGG19 architecture has achieved an accuracy of 91.3 while the RESNET architecture has achieved an accuracy above 95 percent
