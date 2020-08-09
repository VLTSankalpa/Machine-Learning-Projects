# Machine-Learning-Projects

## Blood Transfusion Service

The concept called **Gini importance** of variables and features in random forest classifiers can used to calculate the variable importance and then select most importance variables among other large number of variables or features in the random forest problem.

To explain the process an example dataset related to blood transfusion service centre has used as follow **[Dataset.csv].** To demonstrate the RFMTC marketing model (a modified version of RFM), this study adopted the donor database of blood transfusion service centre in Hsin-Chu city in Taiwan. The RFM ( **Recency, Frequency, Monetary** ) analysis is a marketing technique used to determine quantitatively which customers are the best ones by examining how recently a customer has purchased (recency), how often they purchase (frequency), and how much the customer spends (monetary). To build a FRMTC model, it has selected 748 donors from the donor database. These 748 donor data, each one included following variables,

1. **Recency** - months since last donation
2. **Frequency** - total number of donation
3. **Monetary** - total blood donated
4. **Time** - months since first donation
5. **Binary variable** - represent whether he/she donated blood in March 2007 ( **1** stand for donating blood; **0** stands for not donating blood)

A random forest classifier has used to build the RFMTC marketing model by considering above one to four variables as **independent variables** and fifth variable as **dependent variable** as in **ML Model.py** file.

## Fashion MNIST-SageMaker

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

Zalando seeks to replace the original MNIST dataset
Content

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

## Google Stock Price Trend Prediction


## IOT Network Traffic Prediction


## Proxy Climate Indicators


## Stock Market Anomaly Detection

#### part 1 :- Identifing outlier trades based on Executed Price & Executed Qty using Hierarchical Clustering¶

An outlier is a data point that differs significantly from other observations. Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar and data points in different groups are more differs. Therefore, when clusters form outlier data points will be cluster in to an outlier clusters.

#### part 2 :- Identifing outlier traders based on sum of Executed Qty using Hierarchical Clustering

Form above part 1 it’s possible to conclude that outlier trades form due to executed quantity of trades (the trade that enrol executed qty between 184 and 201) Therefore its possible to identify the outlier traders using hierarchical clustering based on the sum of executed qty for each buyer.
