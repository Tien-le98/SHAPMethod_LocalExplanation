# Explaining Predictions of Used Vehicle Prices by SHAP methods

_Author: Clara Le_

_Date: 10/08/2024_

___

## INTRODUCTION
In recent years, the demands of pre-owned vehicles in Australia have significantly increased due to two main reasons. 

First, negative events such as the COVID-19 pandemic, Red Sea shipping attacks, and Russia-Ukraine war, have negatively impacted the prices and availability of new vehicles. 
+ During the COVID-19 pandemic, the production and delivery of new vehicles were delayed due to serious lock-downs and social distancing requirements, which resulted in the shortage of available new cars in the automobile market.
+ Besides, arising Red Sea shipping attacks compelled carriers to travel through the longer Cape of Good Hope route, which led to soaring transport costs and delivery delays for new vehicles.
+ In addition, the supply chain of wiring harnesses, which is an important material used in manufacturing vehicles, was disrupted due to the Russia-Ukraine war, which also raised the prices of new vehicles.

Second, in light of the government’s proposed New Vehicle Efficiency Standards, along with fierce competition from many rivals, automobile brands tend to apply more technological advancements to their products, making future vehicles more expensive.

Due to this low availability and high prices of new vehicles, customers’ demands were shifted toward second-hand vehicles, causing an increase in used vehicles’ demands and their prices. As shown in the below figure, the average prices of pre-owned vehicles continuously rose from 2010 to 2023, and the changes in their prices even became more significant during and after the Covid-19 pandemic, which was from 2020 to 2023.

<p align="center" width="100%">
    <img width="70%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/price_YoY.png">
</p>

This project focuses on 4 main tasks, as shown below:
+ Building a model that can accurately predict the prices of used vehicles based on their characteristics. Many machine learning models such as linear regression, generalized additive model, tree-based models, and support vector machine, were built and fine-tuned. Their performance were evaluated to figure out the best model, by using RMSE and R-squared.
+ Computing features’ importance, and the average magnitude of effect that each feature can have on the predictions, to identify which vehicle features can significantly impact the estimated prices. Particularly, KernelSHAP and TreeSHAP were considered on the best model to calculate Shapley values for each vehicle feature. The average Shapley value of each feature across the whole dataset was seen as its feature importance.
+ Evaluating the effect of each feature’s value ranges or categories on the price predictions, to support customers in tailoring their needs for suitable used vehicles.
+ Using BayesSHAP to evaluate the uncertainty of Shapley values. Lower uncertainty can indicate higher confidence in using Shapley values for local explanations and measuring features’ importance.

## PROJECT WORKFLOW

### Data collection

The dataset employed in this project was Australian Vehicle Prices dataset, gathered from Kaggle. The dataset included 16734 vehicles sold from 1978 to 2023. Several used cars’ features were: 
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

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/boxplot_price_cyl.png" align="right" width = "48%" ></a>
+ In terms of Cyl variable only, used vehicles having more than 8 cylinders in engine tended to have significant higher average prices than the others. In addition, second-hand cars having more engine cylinders tended to have higher average prices. Similarly, when displacement volume rose, the average prices of used cars also increased. The same trend also can be seen in FuelConsumption factor. According to below figure, engines having higher displacement volume and more engine cylinders can generate more power and consume more fuel, therefore, the prices of those powerful and bigger engines tend to be higher.

<p align="center" width="100%">
    <img width="49%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/scatterplot_price_displacement.png">
    <img width="49%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/scatterplot_price_fuelconsumption.png">
</p>

<a href="url"><img src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/scatterplot_price_kilo.png" align="right" width = "50%" ></a>
+ If we only consider kilometres run, pre-owned vehicles that have traveled more kilometres tended to have lower prices than those with fewer kilometres run. It may be because vehicles have run more kilometres tend to be older than the other used vehicles.

+ Considering the number of doors only, used vehicles with 2 doors had the highest average price because they had higher number of engine cylinders and higher displacement volume. Following 2-door vehicles, vehicles with 4 doors had the second highest average price because it was the most popular vehicle type sold in the period 2010-2023, as indicated in the below table. Therefore, it can be seen that higher demands in used vehicles with 4 doors can lead to their higher prices.

Number of doors | Number of vehicles sold | Percentage (%)
--- | --- | ---
2 | 765 | 6.4
3 | 132 | 1.1
4 | 8934 | 75
5 | 2083 | 17.5

+ According to the correlation plot of the training dataset, only Cyl feature had a positive medium linear relationship with the Price variable, and its correlation coefficient was 0.46. By contrast, Kilometres run had the highest negative impact on the prices, with linear correlation coefficient of around -0.375. However, other features only had weak linear relationships with the price. These variables might have non-linear relationships with the outcome variable instead of linear relationships, hence their relationships can not be correctly indicated by using these linear correlation coefficients, which was confirmed by GAM plots below.

<p align="center" width="100%">
    <img width="45%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/corr_coef_1.png">
</p>

<p align="center" width="100%">
    <img width="45%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/gam_plot_splited.png">
    <img width="45%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/gam_plot_splited_1.png">
</p>

### Training and fine-tuning ML models

Performance of various ML models were showed in the below table.

Model | RMSE | R-squared
--- | --- | ---
Decision tree | 14695.143 | 0.676
$${\color{green}Random forest}$$ | $${\color{green}10933.343}$$ |  $${\color{green}0.821}$$
XGBoost Regressor | 11158.176 | 0.813 
Support vector machine | 23842.403 | 0.148 
Linear regression | 18114.083 | 0.508 
RuleFit | 15224.721 | 0.653 
Lasso regression | 18114.019 | 0.508 
Ridge regression | 18114.028 | 0.508  
GAM | 15708.113 | 0.63

+ SVM achieved the lowest R-squared value, which means that this model was the worst model for the Australian Vehicle Prices dataset.
+ Since relationships between vehicle features and the prices were mostly non-linear, and the interaction terms between features were not considered, linear regression models such as linear regression, lasso regression, and ridge regression also achieved low R-squared (only about 0.508).
+ Taking non-linear relationships into consideration, the GAM performance was pretty higher than these linear regressors, with R-squared of about 0.63.
+ Besides, by combining interaction terms extracted from decision rules and existing features, RuleFit also acquired a higher R- squared of around 0.653.
+ The best model was random forest with 200 trees, and its maximum number of features to consider when looking for the best split was log2(p) with p is the total number of features. Its RMSE value was about 10933.343 and R-squared value was up to 0.821 (82.1%). 

### Evaluating Shapley values for vehicle features
+ Because random forest is a tree-based model, TreeSHAP was employed to compute Shapley values for used vehicle features. Shapley value plots were presented in the below figures. 
+ Several vehicle features had pretty clear relationships with the estimated prices. For example, low values of Kilometres, which were presented in blue points, tended to increase the predictions, while high values negatively affected the predicted prices. Similarly, low values of DriveType (AWD, 4WD) raised the predictions, while high values of FuelType (Unleaded) negatively impacted the estimated prices.

<p align="center" width="100%">
    <img width="49%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/tree_shap_intervention.png">
    <img width="49%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/drivetype_shap.png">
</p>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/avg_shap_intervention.png">
</p>

+ Kilometres had the highest average Shapley value across the whole dataset, meaning that this feature had the largest average magnitude of effect on the price predictions.
+ Among 10 vehicle features considered in this project, Seats, State, and Transmission variables had very small impact on the estimated prices of pre-owned vehicles.
+ There were also several interaction effects between features and the predicted prices. For example, when considering BodyType only, to vehicles which are SUV, Sedan, Ute/Tray, and Wagon, their price predictions tended to be lower than the average price, because Shapley values of these body types were less than 0. Additionally, the lower numbers of engine cylinders in these body types even decreased the estimated prices more significantly than the higher values of Cyl variable.
+ When considering the effect of Brands variable only, to some brands such as Audi, BMW, and Ford, their price predictions tended to be lower than the average price due to their negative Shapley values. In addition, with higher numbers of engine cylinders, their estimated prices even became considerably lower than the average price, as shown in the bottom left corner of the plot. By contrast, other brands witnessed the opposite trend because their predictions tended to be higher than the average price, and these figures became much higher when they have higher number of cylinders in engines.

<p align="center" width="100%">
    <img width="49%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/bodytype_cylinderinengine.png">
    <img width="49%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/brand_cyl_shap.png">
</p>

+ In terms of kilometres traveled, no matter which fuel type were employed, the prices of second-hand vehicles which traveled less than 50000km were higher than the average price, while the figure for vehicles traveled more than 50000km was lower.

<p align="center" width="100%">
    <img width="50%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/kilometre_shap.png">
</p>

+ Average Shapley values for each feature’s category or value range were also presented in the below figure. This figure can support Australian customers in estimating the difference between the predicted price of a second-hand vehicle and the average price when this vehicle’s features are available. For example, considering Kilometres traveled only, with a value of Kilometres less than 50000km, used vehicle’s predicted price tended to be nearly 4570AUD higher on average than the average price. Similarly, the estimated price of a second-hand vehicle using Unleaded tended to be 2645AUD lower than the average price. In terms of drive type only, a pre-owned vehicle using 4WD or AWD can have a predicted price over 2750AUD higher than the average price.

<p align="center" width="100%">
    <img width="50%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/shap_each_valuerange.png">
</p>

### Evaluating uncertainty of Shapley values
+ BayesSHAP package was leveraged to measure the uncertainty of Shapley values for a specific observation as well as the whole testing dataset. The local explanation plot of the first observation was presented in the below figure. On average, Drivetype of 4WD increased the price prediction by about 33923AUD, and Bodytype of Coupe raised this figure by over 7445AUD. On the contrary, Kilometres traveled of 80861km decreased the estimated price by around 14602AUD, and having only 4 cylinders in engine also negatively affected the price prediction with the magnitude of effect was around 38623AUD.
+ In addition, the error bars also presented the uncertainty of Shapley value for each feature of this first observation. The uncertainty seemed to be very small, in comparison with the respective average Shapley values, and the uncertainty even became much smaller when considering the whole testing set, instead of only one record.

<p align="center" width="100%">
    <img width="50%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/bayes_data0.png">
</p>

+ The Shapley value uncertainty of all variables in the whole testing set only ranged from 13AUD to 18AUD on average, which were very small, in comparison with the average Shapley values. Therefore, these low values of uncertainty can indicate the high quality and high consistency of the Shapley values, which can lead to high confidence in utilising them for making local explanations, measuring effects of independent features on the target variable, and evaluating feature importance.

<p align="center" width="100%">
    <img width="50%" src="https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/bayes_global.png">
</p>

## CONCLUSION

Through results of the project, there were several main points as belows:

+ Random forest can outweigh other ML models, and generate good predictions with the R-squared value up to 82%.
+ Only the number of kilometres traveled, the number of cylinders in engine, fuel type, and drive type had significant impact on the estimated prices. The remaining features seem not to have huge influence on the price predictions.
+ Several interaction effects between predictors and the estimated prices were also examined through analyzing SHAP output, but these interaction effects were not very clear. This project also measured the average Shapley value that each category or value range of features can have on the price predictions. For example, on average, some used vehicle features, such as kilometres traveled of more than 50000km, and unleaded fuel type, can have negative contribution to the difference between the price prediction of a specific observation and the average price.
+ According to BayesSHAP output, Shapley values of all features were seen to have high quality and high consistency since their uncertainty values were very small, which can lead to high confidence in employing them.

However, this project still has several limitations that should be addressed in further improvements. 
+ First, observations of minority classes in this dataset should be gathered more, in order to prevent model outputs from being biased. For example, the number of automatic second-hand vehicles were 5 times higher than the figures for manual used vehicles. In addition, the quantity of pre-owned cars with 4 doors were at least 4 times higher than the figures for other categories. This current imbalance of the dataset can negatively affect model performance, and may lead to biased results.
+ Second, as shown in the correlation plots, most features only have weak and medium linear relationships with the dependent variable, hence, more important features should be added to the model in order to increase its ability in correctly predicting used vehicle prices.
+ Third, applying DL models, particularly neural network models, can achieve better performance than traditional ML models. In addition, text variables such as vehicles’ titles and colors were not considered in this project since these variables comprised a lot of words, letters, and categories, which were difficult to be handled by traditional ML models. Therefore, DL models should be considered in the future to figure out the best model for this regression task, and address the problem of textual variables.

> Please refer to this code file for more details: [code](https://github.com/Tien-le98/SHAPMethod_LocalExplanation/blob/main/code_file.ipynb)
