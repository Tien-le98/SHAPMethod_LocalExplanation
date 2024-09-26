# Explaining Predictions of Used Vehicle Prices by SHAP methods

_Author: Clara Le_

_Date: 10/08/2024_

___

## INTRODUCTION
Image classification are becoming more important in various industry fields. For example, in autonomous automobile manufacture, image classification are necessary because it can support self-driving cars in detecting and recognizing other objects such as other vehicles, pedestrians, animals, traffic lights on the road. In healthcare system, image classification can support health professionals in diagnosing diseases such as lung disease, heart disease, and cancer. In addition, image classification can be used to automatically categorize products on e-commerce websites, which can result in better customer’s experience because customers can find relevant products quickly. Image classification also can promote the development of security sector such as surveillance, and facial recognition, since it can be used to identify person, objects, and then potential threats can be addressed quickly. Due to various applications of image classification, this project aims at developing a Convolution Neural Network (CNN) which can perform image classification task well, using CIFAR-10 dataset. Particularly, several CNN architectures were conducted, and a comparison between these architecture’s performance was carried out to find out a CNN model which can achieve the best performance, and generate good prediction on unknown datasets.

The CIFAR-10 dataset comprises 60000 32x32x3 images, meaning that each image has 1024 pixels, along with three channels for red, green, and blue (RGB). The first layer of 1024 entries is the red channel, the second layer of 1024 entries represents for the green channel, and the last layer of 1024 entries represents for the blue channel. Among these images, 50000 images were used for training models, stored in 5 training batches, and 1 testing batch included 10000 images. Both training and testing dataset included 3072 columns, representing for 3072 image’s pixels, and 1 column showing image’s labels. Ten classes comprised in this dataset are vehicles such as airplane, automobile, ship, truck, and animals such as bird, cat, deer, dog, frog, and horse. Each class had 10000 images, their respective labels were shown in the below table, and their images were also shown in the below figure.
|Label|Class|Label|Class|
|:--|:--|:--|:--| 
|0|airplane|5|dog|
|1|automobile|6|frog|
|2|bird|7|horse|
|3|cat|8|ship|
|4|deer|9|truck|

<a href="url"><img src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/image_label.png" align="center" ></a>

This CIFAR-10 dataset was employed to perform data cleaning, data pre-processing, EDA, and data splitting before applying CNN models. Besides, various data augmentation methods for images such as flipping, rotation, and contrast were also utilised. In addition, many CNN architectures were applied on the dataset, including Visual Geometry Group (VGG16), GoogLeNet, InceptionV3, and ResNet-50.

## EXPERIMENTS

### Experiment 1

The first experiment was carried out to compare model’s performance on the original dataset and the pre-processed dataset, in order to figure out if pre-processing steps are necessary for models to perform better.

A baseline CNN model was built with 1 input layer, 1 output layer with 10 neurons, using Softmax function and Sparse categorical crossentropy loss function. This baseline model contained 1 convolutional layer with 64 filters and kernel size of 3x3, followed by a 2x2 max pooling layer, a fully connected layer with 256 neurons and two Dropout layers. SGD optimizer with learning rate of 0.001 was employed in this baseline CNN model. Additionally, this model ran through 300 epochs, and batch size of 100. Early stopping method was also employed to prevent models from potential overfitting problem. The baseline model was trained on the original dataset, a dataset pre-processed with Standard scaling method, Max-Min scaling method, and other data augmentation methods (flipping, rotation, and contrast). The training accuracy score and validation accuracy score of the baseline model were shown in the below figure.

<a href="url"><img src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/baseline.png" align="center"></a>

+ In terms of the original dataset, the baseline CNN model converged after around 5 epochs with the accuracy score on the training set and the validation set of only 10%. However, this baseline model performed better on the dataset pre-processed by only Standard scaling method, with the maximum accuracy score on the training set was about 81%, and the maximum accuracy score on the validation set was nearly 67%, after 43 epochs. Because this accuracy score on the training set was around 14% higher than the figure for the validation set, this gap can raise a signal for potential overfitting problem. On the pre-processed data using only Max-Min scaling method, after 127 epochs, this baseline model obtained the accuracy score on the training dataset of about 79%, and the figure for the validation dataset of nearly 66%. Through this experiment, the dataset should be pre-processed before training CNN models in order to improve the model’s performance because accuracy score of models trained on the pre-processed dataset were significantly higher than the figure for the original dataset.
+ Besides, according to this above table, when a CNN model only contains several layers, its capability is low, data augmentation methods applied on this model can make it perform worse, which was shown by the decrease in the training accuracy score and validation accuracy score. Hence, these data augmentation methods only should be considered in deeper and more complicated neural networks.
+ Because the accuracy score of models trained on standard scaled dataset was the highest among other methods, the pre-processed training data using Standard scaling method was used to train other CNN architectures and analyze further.

### Experiment 2

The second experiment was implemented to find out the CNN architecture that achieved highest accuracy score for this image classification task, using the CIFAR-10, among several different architectures. Architectures considered in this project are VGG16, GoogLeNet, GoogLeNet with auxiliary classifiers, InceptionV3, InceptionV3 with fine tuning, and Resnet-50. These architectures employed Sparse categorical crossentropy loss function, SGD optimizer with 0.001, and batch size of 100. Early stopping method also employed to mitigate potential overfitting problem. The output layer of these architecture used Softmax function for classification since classes were encoded as numbers from 0 to 9. All these architectures were set to run through 50 epochs. The performance of all architectures on the training dataset and validation dataset were shown in the below table.

<a href="url"><img src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/CNN.png" align="center"></a>

After comparing model’s performance of different CNN architectures on the above table, Resnet-50 (with Relu activation function and SGD optimizer) obtained the highest accuracy score on both the training dataset (above 99%) and the validation dataset (nearly 92%), hence, data augmentation methods were considered in this Resnet-50 architecture. 

### Experiment 3

The third experiment was executed by applying data augmentation methods on the Resnet-50 architecture to figure out if these methods can improve model’s performance. Flipping method was used to flip the input images horizontally and vertically. Rotation method was employed to rotate the input images with a factor of 0.2 while contrast method modifies the difference between the darkest and brightest areas in the input images with a factor of 0.2. The below table shows the performance of this architecture, with different methods of data augmentation. 

<a href="url"><img src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/resnet50_dataaugmentation.png" align="center"></a>

It can be seen that data augmentation techniques can perform better on Resnet-50 architecture than the baseline model since this Resnet-50 is more complicated and deeper with the greater number of layers and parameters. These methods can mitigate the overfitting problem, which was shown through the decrease in the difference between training accuracy score and testing accuracy score. This gap of the original Resnet-50 architecture was nearly 7%, while the figure for Resnet-50 with random flip was just about 5%. However, using many data augmentation techniques was not always appropriate since including all methods, such as random flip, random rotation and random contrast methods in this Resnet-50, reduced the model’s performance significantly, with the training accuracy score of only 58.71% and the validation accuracy score of only 23.16%. In addition, through this experiment, the best architecture yielding the highest accuracy score on validation dataset (92.30%) was Resnet-50 with random flip method, however, while validation accuracy score of this Resnet-50 was only 0.7% higher than the figure for the original Resnet-50, it required much longer time to converge. Therefore, the Resnet-50 without data augmentation was still selected to implement the experiment of tuning hyperparameters.

### Experiment 4

The last experiments was carried out by training this Resnet-50 architecture with various values of chosen hyperparameter, to define the optimal hyperparamenters which can lead to better model’s performance. There were 7 combinations of hyperparameters which were tuned in this experiment, keeping other parameters remain unchanged, as shown in the below table. 

<a href="url"><img src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/resnet_tuning.png" align="right" height="300" width="350" ></a>

Resnet-50 architecture obtained high validation accuracy score, regardless of different combinations of activation function and optimizer. In general, their validation accuracy scores were about 80% or above. The best model is the Resnet-50 architecture with SELU activation function, SGD optimizer, and its learning rate is 0.001. The below figure showed changes in training loss, training accuracy, validation loss, and validation accuracy of this best model through each epoch. In addition, this model converged after 17 epochs with the training accuracy score of 99.56% and the validation accuracy score of 91.70%, which was just slightly higher than the figure for the original Resnet-50 using ReLU and SGD optimizer. However, in comparison with the baseline model, this best model’s validation accuracy score was nearly 25% higher than the figure for the baseline CNN. In addition, the difference between training accuracy score and validation accuracy score of this best model was only about 8%, which was 6% lower than the figure for the baseline model. Several reasons that can lead to this improvement, as belows: 

+ First, this best Resnet-50 has a deeper network with a greater number of layers than the baseline model, which can help model to learn more information of the input data.
+ Second, because vanishing gradient and non-zero mean are two of common problems that can negatively affect model’s performance, the SELU activation function also can contributed in this improvement of accuracy scores since it can mitigate vanishing gradient problem and push the mean of activation function closer to 0.
+ In addition, different from ReLU activation function which converted all negative values to 0, SELU creates a smooth curve going through these negative numbers, hence it still can keep information of these input data, and improve model’s performance.

<a href="url"><img src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/resnet_tuning_plot.png" align="center" height="400" width="700"></a>

This best Resnet-50 was used to evaluate model’s performance on the testing dataset. In particular, its testing loss was about 0.258, and its accuracy score and precision score was around 91.3%. In terms of individual classes, each label was predicted accurately more than 90% of its number of images, except for label 2, 3 and 5 since they were only predicted precisely about 87%, 81% and 85% respectively, as shown in the below figure. In addition, in terms of predictions, around 90% of predicted labels were the same with true labels, except for predictions of label 3 and 5, since only 82% labels predicted as 3 were correct, and 88% labels predicted as 5 were precise. 

<p align="center" width="100%">
    <img width="49%" src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/confmat_truelabel.png">
    <img width="49%" src="https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/confmat_predictedlabel.png">
</p>

## CONCLUSION

Through results of these experiments, there were several main points as belows:
+ Preprocessed data steps are important in improving model’s performance. In particular, Standard scaling method can improve model’s performance better than other pre-processing methods.
+ Although data augmentation techniques can have positive effects on model’s performance, if the model does not have a large number of layers, these techniques can make it performs worse. It was shown through the decreased accuracy score of the baseline CNN model when data augmentation methods were employed. On the contrary, if the neural network is deeper, these data augmentation techniques can promote its performance and mitigate overfitting problem, leading to higher accuracy scores, which was shown in the performance of Resnet-50 architecture when applying different data augmentation methods. However, choosing appropriate number and type of data augmentation methods should be considered thoroughly since employing many techniques can lead to worse model’s performance.
+ Among different architectures which are appropriate in image classification, Resnet-50 were seen as the best architecture, achieving the highest accuracy score on the training dataset and the validation dataset. After tuning hyperparameters, the Resnet-50 with SGD optimizer, and SELU activation function obtained the highest accuracy score on the validation dataset because SELU activation function can alleviate vanishing gradient, and non-zero mean problems. Therefore, hyperparameters can affect model’s performance, tuning hyperparameters steps can contribute significantly in choosing the best model, which can result in better overall accuracy, meaning that better model’s performance.

> Please refer to this code file for more details: [code](https://github.com/Tien-le98/CIFAR-10-Image-Classification/blob/main/code_file.ipynb)
