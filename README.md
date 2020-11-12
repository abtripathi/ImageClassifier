In this project,transfer learning using Pytorch is applied to build an Image classifier to recognize different species of flowers using the [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories

A pretrained neural network for e.g. Vgg16 which is already trained on [ImageNet](http://www.image-net.org/), a massive dataset with over 1 million labeled images in 1000 categories.This dataset is [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html) and is used to extract general features from input images

An Image classifier is build with fully connected and activation units which classifies the species of flowers based on the features extracted by pretrained network




