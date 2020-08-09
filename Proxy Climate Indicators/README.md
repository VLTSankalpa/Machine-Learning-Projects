# Proxy climate indicators

**PROBLEM DEFINITION AND APPROACH**

The World Data Center (WDC) for Paleoclimatology maintains the ice core data from polar and low-latitude mountain glaciers and ice caps throughout the world. Proxy climate indicators related to glaciers and ice caps include oxygen isotopes, methane concentrations, dust content, as well as many other parameters. As a one of important climate indicator correlation between CO2 level of ice core and age of the ice core, need to be investigate.

**IMPLEMENTATION DETAILS**

Aim of implementing this simple linear regression model is to predict the CO2 level of ice cores based on their age. Therefore, this model has consider **age** (years) of ice cores as the **independent variable** and the **CO2 level** (ppmv: -parts per million by volume) of ice core as the **dependent variable**. A data set consist with 1096 observations on the above two variables. [Note: check **Q4.py** for more implementation details of the model]

![](RackMultipart20200809-4-qh2ziz_html_af26aa5c57cf0839.png)

**ACCURACY IMPROVEMENTS**

![](RackMultipart20200809-4-qh2ziz_html_7c446a3fb176458f.gif)

According to above graphs data set shows nonlinear characteristics, simple linear regression gives very poor accurate predictions. Therefore, improvements in predictions results has achieved using random forest regression model.

**Note:**

- Red colour dots represent actual data points in training and testing dataset
- Blue line represent the fitted model

Below figure, compare the values among,

**y\_test** : - testing data set points

**y\_pred\_SLR** : - predictions of simple linear regression model

**y\_pred\_RFR** : - predictions of random forest regression model

![](RackMultipart20200809-4-qh2ziz_html_89f45449c221f0c9.png)

**PERFORMANCE OPTIMIZATIONS**

**Feature scaling** can used to normalize the range of independent variables or features of dataset in order to optimize the training or learning process of a above random forest regression model. In addition, this can considerably reduce the processing power and time required to plot graphs in above-mentioned Q4.py. This is also known as data **normalization** and is generally performed during the data pre-processing step. However, in the other hand feature scaling can harmfully effect to the interpretability of the machine learning model. Therefore in above model feature scaling has not been applied and particular **StandardScaler**  code section has included as a comment. Other than StandardScaler below, scalars also can be used for this purpose.

1. MinMaxScaler
2. RobustScaler
3. Normalizer

**SIMPLE LINEAR CLASSIFIER**

Above simple linear regression model can used to classify glaciers and ice caps in to two categorize as CO2 saturated and unsaturated glaciers. All the data point above the best fitting line of simple linear classifier represent **CO2 saturated glaciers** while all the data point below the best fitting line represent **CO2 unsaturated glaciers**. Therefore, above simple linear regression model can utilized as simple linear classifier to identify any give glacier as CO2 saturated or unsaturated based on their age.

**[Note: - The entire python programme has created using Anaconda 2019.03 and Python 3.7.3 64-bit****]**