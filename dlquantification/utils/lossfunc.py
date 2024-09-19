import torch.nn.functional as F


def JSD_Loss(p, p_hat):
    m = 0.5 * (p + p_hat)
    # compute the JSD Loss
    return 0.5 * (F.kl_div(p.log(), m) + F.kl_div(p_hat.log(), m))


class MRAE:
    def __init__(self, eps, n_classes):
        self.eps = eps
        self.n_classes = n_classes

    def MRAE(self, p, p_hat):
        """MRAE implementation following Sebastiani et al. (2021) Evaluation Measures

        Args:
            p (torch.Tensor): True prevalences
            p_hat (torch.Tensor): Predicted prevalences
            eps ([type]): value for smothing. Suggested value in the paper is 1/2*n_samples

        Returns:
            [type]: [description]
        """
        if len(p.shape) != len(p_hat.shape):
            raise ValueError("The shape does not match")
        else:
            dims = len(p.shape)
        p_s = (p + self.eps) / (self.eps * self.n_classes + 1)
        p_hat_s = (p_hat + self.eps) / (self.eps * self.n_classes + 1)
        return (abs(p_s - p_hat_s) / p_s).mean(dims - 1).mean()
