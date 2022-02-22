# Pruning CNN models

## Summary

The goal of this CNN implementation is three fold:
1. Train my own CNN architecture to classify the images
2. Use Transfer Learning to improve my model scores
3. Use pruning to simplify the TL model while maintaining my scores

My aim is to implement this excellent research paper on training sparse neural networks using the [Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf).

## Dataset

There are ~24,000 images of Natural scences around the world and is present on [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification). The images are of size 150x150 distributed under 6 categories which are as follows:

|Label| Landscape|
|-|-|
|0| buildings|
|1| forest|
|2| glacier|
|3| mountain|
|4| sea|
|5| street|

The train, test and prediction data is separated in each zip files. I would be using the first two sets for this analysis.
This data was initially published on [AnalyticsVidhya](https://datahack.analyticsvidhya.com) by Intel to host a Image classification Challenge.

## Results

Since the problem is a multi-label image classification problem, CrossEntropyLoss is used with the Adam Optimizer. The number of epochs is 30 and default seed is as follows:

```{python}
torch.manual_seed(2018)
```
The simpleCNN is my baseline model implemenetation with 2 CNN layers using MaxPool and Dropout some layers in between. The following scores are of the last epoch.

| Implementation| Learning Rate| Train Accuracy | Valid Accuracy |
|-|-|-|-|
| simpleCNN - 2 CNN layers| 0.001| 0.98| 0.74|
| densenet121 with default parameters| 0.001| 0.99| 0.89|
| resnet18 with default parameters| 0.001| 0.98| 0.87|

## Dependencies

numpy=1.20.3 <br>
pandas=1.3.5 <br>
torch=1.9.1 <br>
torchvision=0.10.1 <br>

## License

This analysis was created by Rohit Rawat. Feel free to fork/clone and build on this code with attribution. It is licensed under the terms of the MIT license. 
