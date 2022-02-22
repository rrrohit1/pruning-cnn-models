# Pruning CNN models

## Summary

The goal of this CNN implementation is three fold:
1. Train my own CNN architecture to classify the images
2. Use Transfer Learning to improve my model scores
3. Use pruning to simplify the TL model while maintaining my scores

My aim is to implement this excellent research paper on training sparse neural networks using the [Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf).

## Dependencies

numpy=1.20.3
pandas=1.3.5
torch=1.9.1
torchvision=0.10.1 
