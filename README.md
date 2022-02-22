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

Since the problem is a multi-label image classification problem, CrossEntropyLoss is used with the Adam Optimizer.

| Implementation| Learning Rate| Train Accuracy | Valid Accuracy |
|-|-|-|-|
| simpleCNN - 2 CNN layers| 0.001| 0.97| 0.72|
| simpleCNN - 2 CNN layers| 0.0005| 0.87| 0.70|
| DenseNet with default parameters| 0.0005| 0.98| 0.89|

## Dependencies

numpy=1.20.3 <br>
pandas=1.3.5 <br>
torch=1.9.1 <br>
torchvision=0.10.1 <br>

## License

This analysis was created by Rohit Rawat. Feel free to fork/clone and build on this code with attirbution to this repository. It is licensed under the terms of the MIT license. 
