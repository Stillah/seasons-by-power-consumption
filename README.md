# Project for Machine Learning course

## Team
Gatiatullin Ruslan ES05334 (the only team member).

## Introduction
In a world increasingly driven by data, the patterns of energy consumption and production provide a hidden insight into the rhythms of nature and human behavior. By examining hourly power production and consumption we can detect subtle patterns that reveal the time of year, effectively predicting the current season. This ability is based on the cyclical relationship between energy systems and seasonal changes: during winter, the early sunsets increase the demand for evening lighting, while during summer, the long daylight hours boost solar production. The transitional seasons, such as spring and autumn, exhibit unique load profiles as heating and cooling needs fluctuate.
## Problem
To predict which season of the year (winter, spring, summer, autumn) corresponds to a 24-hour record of power consumption and production of a *european* country.
## Example of problem applications
* Adjust electricity prices in real-time based on predicted seasonal demand patterns, encouraging off-peak usage
* Estimate seasonal emissions from power grids and suggest cleaner energy alternatives
* Compare with future data to analyze the extend of adoptation of renewable energy

## Dataset
Open Power System Data dataset. 

Dataset used is a subset of the Open Power System Data, which contains various hourly power
related measurements (e.g., consumption, wind generation, solar capacity) across EU 
countries from 2015 to approximately 2020.

## EDA
![Winter](images/winter.png)
![Spring](images/spring.png)
![Summer](images/summer.png)
![Autumn](images/autumn.png)

These 3 columns were chosen because they are the most descriptive of the season. 

Observations:
1. Wind generation is almost constant throughout 1 season
2. Wind generation differs significantly from season to season
3. Solar electricity is generated between 4 am and 6 pm and is very different for each season

## Potential solutions
Various ML techniques can be applied to solve this problem.

The most simple choice is linear/polynomial regression. Due to their simplicity, these algorithms should give comparatively low accuracy.

Another option could be k-means++. It also shouldn't give great results because some data fields have more predictive weight than others, but k-means++ treats them equally in terms of distance.

This leads us to neural networks.

## Proposed solutions
We propose Multi-layer Perceptron (MLP), 1D and 2D convolutional neural networks (CNN) as our solutions to the problem. We think they are particularly well-suited for this task for the following reasons:

* MLP performs well when the data is well-structured because it directly learns from raw features without relying on spacial relationships
* 1D CNN is designed to capture patterns in sequential data such as time-series (which we have in this task)
* 2D CNN is likely going to perform well, but worse than the previous 2 algorithms because of small 'image' size (24 pixels while it usually needs 1000+). It's included mainly for comparison

## Results
We have chosen Denmark for testing the models. The results on a random launch are as follows:

* MLP accuracy: 0.8349
* 1D CNN accuracy: 0.8540
* 2D CNN accuracy: 0.8540

The accuracy is quite high for all methods with 1D and 2D CNN being the best. Furthermore, accuracy for 1D CNN can be improved to 0.9+ if StandardScaler is used instead of MinMaxScaler. We used MinMaxScaler because it leads to significantly better results for 2D CNN with GAF transformation. 1D CNN being the best (with StandardScaler) is expected since it's specifically designed to work with time-series. MLP is slightly worse than the other 2 models because it doesn't capture more distant patterns.

Below confusion matrices and training graphs are presented.
### MLP
![MLP Confusion matrix](images/MLP_confusion.png)
![MLP training](images/MLP_graphs.png)

### 1D CNN
![1D](images/1D-CNN_confusion.png)
![1D](images/1D-CNN_graphs.png)

### 2D CNN
![2D](images/2D-CNN_confusion.png)
![2D](images/2D-CNN_graphs.png)


## How to use
First clone the repository and install requirements.txt.

Then enter desired values for n number of days to input.csv and launch predict.py. Values should correspond to Denmark as it was used for training, for other countries training dataset needs to be changed in main.ipynb.

## Reflection
During this project we have recieved hands on experience in machine learning. A lot (90%) of the expertise needed to implement this project wasn't covered in the course, so we had to put significant effort into understanding new material ourselves. Luckily, it was fun and engaging. As a result, we have learned in-depth how MLPs & CNNs work in theory and how to implement them in practice using pytorch. Besides that we have learned a lot of smaller ML aspects, e.g. how to perform exploratory data analysis (EDA) and ensure no data leakage.
