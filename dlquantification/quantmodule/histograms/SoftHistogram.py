# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHistogram(nn.Module):
    """
    This class computes a histogram based in convolutional layers. I have followed the paper:
    Wang, Zhe, et al. "Learnable histogram: Statistical context features for deep neural networks." European
    Conference on Computer Vision. Springer, Cham, 2016.
    We use two convolutions, one for computing the center and the other for computing the vote from each value to each
    bin.
    Bin centers and witdh is initialized but are learned during the process.
    """

    def __init__(self, n_features, num_bins, quantiles=False):

        # inherit nn.module
        super(SoftHistogram, self).__init__()

        # The number of features will be the number of channels to use in the convolution
        self.in_channels = n_features
        self.num_bins = num_bins
        self.quantiles = quantiles

        # in_channels: number of features that we have
        # out_channels: number of bins*number of features
        # kernel_size: 1
        # groups: we group by channel, each input channel is convolved with its own set of filters
        # in_channels: correspond with the number of input features
        # out_channels: n_features*n_bins
        # bias terms: we will have as many bias terms as the out channels. This is ok because these are the bin centers
        self.bin_centers_conv = nn.Conv1d(
            self.in_channels, self.num_bins * self.in_channels, kernel_size=1, groups=self.in_channels, bias=True
        )
        # All weights to one and we do not want to learn these.
        # We are modeling the centering operation, we only work with bias
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False

        self.bin_widths_conv = nn.Conv1d(
            self.num_bins * self.in_channels,
            self.num_bins * self.in_channels,
            kernel_size=1,
            groups=self.num_bins * self.in_channels,
            bias=True,
        )
        # In this case, bias fixed with ones and we do not learn it
        self.bin_widths_conv.bias.data.fill_(1)
        self.bin_widths_conv.bias.requires_grad = False

        # self.hist_pool = nn.AvgPool1d(n_examples, stride=1, padding=0)  # average by row
        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

        # Compute initial bin centers and bin sizes, this can change during backpropagation
        bin_centers = -1 / self.num_bins * (torch.arange(self.num_bins).float() + 0.5)
        self.bin_centers_conv.bias = torch.nn.Parameter(torch.cat(self.in_channels * [bin_centers]), requires_grad=True)
        bin_width = -self.num_bins * 2
        self.bin_widths_conv.weight.data.fill_(bin_width)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        result = torch.empty((input.shape[0], self.num_bins * self.in_channels), device=input.device)
        # Histograms do not work with batches of samples, process sample by sample
        for i, sample in enumerate(input):
            # Pass through first convolution to learn bin centers: |x-center|
            sample = self.bin_centers_conv(sample.transpose(0, 1).unsqueeze(0))
            sample = torch.abs(sample)

            # Pass through second convolution to learn bin widths 1-w*|x-center|
            sample = self.bin_widths_conv(sample)

            # Pass through relu
            sample = F.relu(sample)

            # Enforce sum to one constraint
            # Add small positive constant in case sum is zero
            # if(self.normalize_bins):
            #    xx = self.constrain_bins(xx)

            # input = self.hist_pool(input)
            sample = torch.mean(sample, dim=2)

            if self.quantiles:
                sample = input.view(-1, self.num_bins).cumsum(dim=1)

            # The output is a Tensor with size n_bins*n_features
            result[i, :] = sample.flatten()
        return result

    # def constrain_bins(self, xx):
    #     # Enforce sum to one constraint across bins
    #     n, c, l = xx.size()
    #     xx_sum = xx.reshape(n, c // self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
    #     xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
    #     xx = xx / xx_sum
    #     return xx


class SoftHistogramRBF(nn.Module):
    """
    This class computes a histogram based in convolutional layers. I have followed the paper:
    Peeples, Joshua, Weihuang Xu, and Alina Zare. "Histogram Layers for Texture Analysis."
    arXiv preprint arXiv:2001.00215 (2020).
    We use two convolutions, one for computing the center and the other for computing the vote from each value to
    each bin.
    Bin centers and witdh is initialized but are learned during the process.
    """

    def __init__(self, n_features, num_bins, quantiles=False):

        # inherit nn.module
        super(SoftHistogramRBF, self).__init__()

        # The number of features will be the number of channels to use in the convolution
        self.in_channels = n_features
        self.num_bins = num_bins
        self.quantiles = quantiles

        # in_channels: number of features that we have
        # out_channels: number of bins*number of features
        # kernel_size: 1
        # groups: we group by channel, each input channel is convolved with its own set of filters
        # in_channels: correspond with the number of input features
        # out_channels: n_features*n_bins
        # bias terms: we will have as many bias terms as the out channels. This is ok because these are the bin centers
        self.bin_centers_conv = nn.Conv1d(
            self.in_channels, self.num_bins * self.in_channels, kernel_size=1, groups=self.in_channels, bias=True
        )
        # All weights to one and we do not want to learn these. We are modeling the centering operation,
        # we only work with bias
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False

        self.bin_widths_conv = nn.Conv1d(
            self.num_bins * self.in_channels,
            self.num_bins * self.in_channels,
            kernel_size=1,
            groups=self.num_bins * self.in_channels,
            bias=False,
        )

        # self.hist_pool = nn.AvgPool1d(n_examples, stride=1, padding=0)  # average by row
        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

        # Compute initial bin centers and bin sizes, this can change during backpropagation
        bin_centers = -1 / self.num_bins * (torch.arange(self.num_bins).float() + 0.5)
        self.bin_centers_conv.bias = torch.nn.Parameter(torch.cat(self.in_channels * [bin_centers]), requires_grad=True)
        bin_width = self.num_bins * 2
        self.bin_widths_conv.weight.data.fill_(bin_width)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        result = torch.empty((input.shape[0], self.num_bins * self.in_channels), device=input.device)
        # Histograms do not work with batches of samples, process sample by sample
        for i, sample in enumerate(input):
            # Pass through first convolution to learn bin centers: |x-center|
            sample = self.bin_centers_conv(sample.transpose(0, 1).unsqueeze(0))
            sample = torch.abs(sample)

            # Pass through second convolution to learn bin widths
            sample = self.bin_widths_conv(sample)

            # Pass through relu (this simulates the max operation in the paper)
            sample = torch.exp(-(sample**2))

            # Enforce sum to one constraint
            # Add small positive constant in case sum is zero
            # if(self.normalize_bins):
            #    xx = self.constrain_bins(xx)

            # input = self.hist_pool(input)
            sample = torch.mean(sample, dim=2)

            if self.quantiles:
                sample = sample.view(-1, self.num_bins).cumsum(dim=1)

            result[i, :] = sample.flatten()

            # The output is a Tensor with size n_bins*n_features
        return result
