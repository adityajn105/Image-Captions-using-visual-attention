# Image-Captions-using-visual-attention [WIP]
Implemenetation of 2016 paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" on Flickr8k dataset.


## Paper Summary

1. Overview

	* Image Captioning using high level VGG19 features
	* Soft and Hard Attention
	* Focus will be on decoder

2. Encoder
	
	* VGG19 will take 224 * 224 * 3 image and generate 512 feature maps of size 14 * 14
	* Each pixel of 14x14 feature map represents a region of image, so we have 512 features for a region of image.
	* We will represent 196 regions with 512 features for each region.

3. Attention
	
	* Soft attention is when we calculate the context vector as a weighted sum of the encoder features. 
	* Hard attention is when, instead of weighted average of all encoder features, we use attention scores to select a single hidden state.
	* Here we will use Soft Attention.
	* 