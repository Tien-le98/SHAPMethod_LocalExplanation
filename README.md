# Explaining Predictions of Used Vehicle Prices by SHAP methods

_Author: Clara Le_

_Date: 10/08/2024_

___

## INTRODUCTION
In recent years, the demands of pre-owned vehicles in Australia have significantly increased due to two main reasons. First, negative events such as the COVID-19 pandemic, Red Sea shipping attacks, and Russia-Ukraine war, have negatively impacted the prices and availability of new vehicles. During the COVID-19 pandemic, the production and delivery of new vehicles were delayed due to serious lock-downs and social distancing requirements, which resulted in the shortage of available new cars in the automobile market. Besides, arising Red Sea shipping attacks compelled carriers to travel through the longer Cape of Good Hope route, which led to soaring transport costs and delivery delays for new vehicles. In addition, the supply chain of wiring harnesses, which is an important material used in manufacturing vehicles, was disrupted due to the Russia-Ukraine war, which also raised the prices of new vehicles. Second, in light of the government’s proposed New Vehicle Efficiency Standards, along with fierce competition from many rivals, automobile brands tend to apply more technological advancements to their products, making future vehicles more expensive. Due to this low availability and high prices of new vehicles, customers’ demands were shifted toward second-hand vehicles, causing an increase in used vehicles’ demands and their prices. As shown in the below figure, the average prices of pre-owned vehicles continuously rose from 2010 to 2023, and the changes in their prices even became more significant during and after the Covid-19 pandemic, which was from 2020 to 2023.

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/price_YoY.png" width = "70%" align="center" ></a>

This project focuses on 4 main tasks, as shown below:
+ Building a model that can accurately predict the prices of used vehicles based on their characteristics. Many machine learning models such as linear regression, generalized additive model, tree-based models, and support vector machine, were built and fine-tuned. Their performance were evaluated to figure out the best model, by using RMSE and R-squared.
+ Computing features’ importance, and the average magnitude of effect that each feature can have on the predictions, to identify which vehicle features can significantly impact the estimated prices. Particularly, KernelSHAP and TreeSHAP were considered on the best model to calculate Shapley values for each vehicle feature. The average Shapley value of each feature across the whole dataset was seen as its feature importance.
+ Evaluating the effect of each feature’s value ranges or categories on the price predictions, to support customers in tailoring their needs for suitable used vehicles.
+ Using BayesSHAP to evaluate the uncertainty of Shapley values. Lower uncertainty can indicate higher confidence in using Shapley values for local explanations and measuring features’ importance.

## PROJECT WORKFLOW

### Data collection

The dataset employed in this paper was Australian Vehicle Prices dataset, gathered from Kaggle. The dataset included 16734 vehicles sold from 1978 to 2023. Several used cars’ features were: 
+ Transmission type: Automatic and Manual
+ Drive type: Front-wheel/Front, Four-wheel/4WD, All-wheel/AWD, and Rear-wheel/Rear
+ Fuel type: Unleaded, Diesel, Premium, Hybrid, and Liquefied Petroleum Gas/LPG
+ Body type: SUV, Hatch-back, Ute/Tray, Sedan, Wagon, Commercial, Coupe, Convertible, People Mover, and Other
+ The volume of displacement (Disp)
+ Fuel consumption rate (FuelConsumption)
+ Kilometres traveled (Kilometres)
+ The number of cylinders in engine (CylindersinEngine/Cyl)
+ The quantity of doors (Doors), and
+ The number of seats (Seats).

### Data pre-processing and data splitting
The dataset went through various pre-processing steps such as data type correctness, variable transformation, removing invalid values, handling outliers and missing values. The final dataset only comprised 11914 pre-owned vehicles sold from 2010 to 2023. 

Because Disp, FuelConsumption, and Cyl variables were highly correlated to each other, this can indicate a multicollinearity problem, which can negatively affect model performance. In addition, since Cyl feature had the highest correlation coefficient with the price, only Cyl was remained, Disp and FuelConsumption were removed from the final dataset. Therefore, the final dataset had 10 features and 1 dependent variable (Price).

### Exploratory Data Analysis (EDA)
Several main relationships can be extracted as below:
<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/boxplot_price_transmission.png" align="right" width = "50%" ></a>
+ Automatic used vehicles tended to have higher average price than manual vehicles, holding other features unchanged. It may be because an automatic transmission system is more advanced and complicated than a manual system, it usually requires higher production costs, which can result in higher prices.

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/boxplot_price_fueltype.png" align="right" width = "50%" ></a>
+ Keeping other variables unchanged, hybrid second-hand vehicles witnessed the highest average price, while the figure for LPG vehicles was the lowest. Particularly, hybrid cars usually have higher prices than other petrol-powered cars due to its better fuel efficiency. On the contrary, the average price of LPG vehicles was the lowest because LPG cars have a lower energy density per litre, which means that when travelling the same distance, LPG cars usually require more fuel than petrol-powered vehicles. Hence, customers tend to shift their demands toward more efficient cars, which lead to a decline in the number of LPG cars used in Australia as well as their prices.

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/boxplot_price_drivetype.png" align="right" width = "48%" ></a>
+ Pre-owned vehicles with 4WD and AWD tended to have higher average prices than the other drive types. In addition, front-wheel drive type had the lowest average price, compared to other drive types, keeping other features fixed. Similar to transmission type, because AWD and 4WD types employ advanced systems to divert power to all four wheels, their production costs are usually higher than the figure for two-wheel drive systems (2WD), thus their prices also tend to be higher.

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/boxplot_price_cyl.png" align="right" width = "45%" ></a>
+ In terms of Cyl variable only, used vehicles having more than 8 cylinders in engine tended to have significant higher average prices than the others. In addition, second-hand cars having more engine cylinders tended to have higher average prices. Similarly, when displacement volume rose, the average prices of used cars also increased. The same trend also can be seen in FuelConsumption factor. According to below figure, engines having higher displacement volume and more engine cylinders can generate more power and consume more fuel, therefore, the prices of those powerful and bigger engines tend to be higher.
<p align="center" width="100%">
    <img width="45%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/scatterplot_price_displacement.png">
    <img width="45%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/scatterplot_price_fuelconsumption.png">
</p>

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/scatterplot_price_kilo.png" align="right" width = "50%" ></a>
+ If we only consider kilometres run, pre-owned vehicles that have traveled more kilometres tended to have lower prices than those with fewer kilometres run. It may be because vehicles have run more kilometres tend to be older than the other used vehicles.

+ Considering the number of doors only, used vehicles with 2 doors had the highest average price because they had higher number of engine cylinders and higher displacement volume. Following 2-door vehicles, vehicles with 4 doors had the second highest average price because it was the most popular vehicle type sold in the period 2010-2023, as indicated in the below Tab. 1. Therefore, it can be seen that higher demands in used vehicles with 4 doors can lead to their higher prices.

Number of doors | Number of vehicles sold | Percentage (%)
--- | --- | ---
2 | 765 | 6.4
3 | 132 | 1.1
4 | 8934 | 75
5 | 2083 | 17.5

+ According to the correlation plot of the training dataset, only Cyl feature had a positive medium linear relationship with the Price variable, and its correlation coefficient was 0.46. By contrast, Kilometres run had the highest negative impact on the prices, with linear correlation coefficient of around -0.375. However, other features only had weak linear relationships with the price. These variables might have non-linear relationships with the outcome variable instead of linear relationships, hence their relationships can not be correctly indicated by using these linear correlation coefficients, which was confirmed by GAM plots later.
<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/corr_coef_1.png" align="center" width = "60%" ></a>


### Training and fine-tuning ML models

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
