import torch


class BasicCount(torch.nn.Module):
    """
    This class allows to build a histogram that is differentiable. This histogram is an aproximation, not the real one
    Following: https://discuss.pytorch.org/t/differentiable-torch-histc/25865
    """

    def __init__(self, n_classes):
        super(BasicCount, self).__init__()
        self.n_classes = n_classes
        self.num_bins = 1

    def forward(self, input):
        n_examples = input.shape[0]
        max = torch.argmax(input, dim=1)
        counts = torch.bincount(max, minlength=self.n_classes)
        frequs = counts / n_examples
        return frequs
