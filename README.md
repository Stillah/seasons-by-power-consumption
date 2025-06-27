# Project for Machine Learning course

## Introduction
In a world increasingly driven by data, the patterns of energy consumption and production provide a hidden insight into the rhythms of nature and human behavior. By examining hourly power production and consumption we can detect subtle patterns that reveal the time of year, effectively predicting the current season. This ability is based on the cyclical relationship between energy systems and seasonal changes: during winter, the early sunsets increase the demand for evening lighting, while during summer, the long daylight hours boost solar production. The transitional seasons, such as spring and autumn, exhibit unique load profiles as heating and cooling needs fluctuate.
## Problem
To predict which season of the year (winter, spring, summer, autumn) corresponds to a 24-hour record of power consumption and production of a *european* country.
## Example of problem applications
* Adjust electricity prices in real-time based on predicted seasonal demand patterns, encouraging off-peak usage
* Estimate seasonal emissions from power grids and suggest cleaner energy alternatives
* Compare with future data to analyze the effect of global warming

## Dataset
Open Power System Data dataset. 

Dataset used is a subset of the Open Power System Data, which contains various hourly power
related measurements (e.g., consumption, wind generation, solar capacity) across EU 
countries from 2015 to approximately 2020.

## EDA

## Potential solutions
Various ML techniques can be applied to solve this problem.

The most simple choice is linear/polynomial regression. Due to their simplisity, these algorithms should give the worst accuracy.

Another option could be k-means++. It also shouldn't give great results because some data fields have more predictive weight than others, but k-means++ treats them equally in terms of distance.

This leads us to neural networks.

## Proposed solutions
We propose Multi-layer Perceptron (MLP), 1D and 2D convolutional neural networks (CNN) as our solutions to the problem. We think they are particularly well-suited for this task for the following reasons:

* MLP performs well when the data is well-structured because it directly learns from raw features without relying on spacial relationships
* 1D CNN is designed to capture patterns in sequential data such as time-series (which we have in this task)
* 2D CNN is likely going to perform well, but worse than the previous 2 algorithms because of small 'image' size (24 pixels while it usually needs 1000+). It's included mainly for comparison



