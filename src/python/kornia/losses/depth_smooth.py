import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}


    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples::

        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = kornia.losses.DepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)
    """

    def __init__(self, reduction='mean') -> None:
        super(InverseDepthSmoothnessLoss, self).__init__()
        self.reduction = reduction

    @staticmethod
    def gradient_x(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    @staticmethod
    def gradient_y(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def forward(  # type: ignore
            self,
            idepth: torch.Tensor,
            image: torch.Tensor,
            mask=None, grad_weight=True) -> torch.Tensor:
        if not torch.is_tensor(idepth):
            raise TypeError("Input idepth type is not a torch.Tensor. Got {}"
                            .format(type(idepth)))
        if not torch.is_tensor(image):
            raise TypeError("Input image type is not a torch.Tensor. Got {}"
                            .format(type(image)))
        if not len(idepth.shape) == 4:
            raise ValueError("Invalid idepth shape, we expect BxCxHxW. Got: {}"
                             .format(idepth.shape))
        if not len(image.shape) == 4:
            raise ValueError("Invalid image shape, we expect BxCxHxW. Got: {}"
                             .format(image.shape))
        if not idepth.shape[-2:] == image.shape[-2:]:
            raise ValueError("idepth and image shapes must be the same. Got: {}"
                             .format(idepth.shape, image.shape))
        if not idepth.device == image.device:
            raise ValueError(
                "idepth and image must be in the same device. Got: {}".format(
                    idepth.device, image.device))
        if not idepth.dtype == image.dtype:
            raise ValueError(
                "idepth and image must be in the same dtype. Got: {}".format(
                    idepth.dtype, image.dtype))
        # compute the gradients
        idepth_dx: torch.Tensor = self.gradient_x(idepth)
        idepth_dy: torch.Tensor = self.gradient_y(idepth)
        image_dx: torch.Tensor = self.gradient_x(image)
        image_dy: torch.Tensor = self.gradient_y(image)

        # compute image weights
        if grad_weight:
            weights_x: torch.Tensor = torch.exp(
                -torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
            weights_y: torch.Tensor = torch.exp(
                -torch.mean(torch.abs(image_dy), dim=1, keepdim=True))
        else:  # do not weight using gradient
            weights_x: torch.Tensor = torch.ones_like(idepth_dx)
            weights_y: torch.Tensor = torch.ones_like(idepth_dy)

        # apply image weights to depth
        smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
        smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)

        # -BH add mask
        if mask is not None:
            mask = mask.expand_as(idepth)
            # return torch.mean(smoothness_x[mask[..., 0:-1]]) + torch.mean(smoothness_y[mask[..., 0:-1, :]])
            num_pix_x = mask[..., 0:-1].numel()
            num_ones_x = mask[..., 0:-1].sum()
            num_pix_y = mask[..., 0:-1, :].numel()
            num_ones_y = mask[..., 0:-1, :].sum()
            return torch.mean(smoothness_x*mask[..., 0:-1])*num_ones_x/num_pix_x + torch.mean(smoothness_y*mask[..., 0:-1, :])*num_ones_y/num_pix_y
            # return torch.mean(smoothness_x*mask[..., 0:-1]) + torch.mean(smoothness_y*mask[..., 0:-1, :])
        else:
            if self.reduction == 'none':
               return  smoothness_x.mean(1).mean(1).mean(1) + smoothness_y.mean(1).mean(1).mean(1)
            else:
                return torch.mean(smoothness_x) + torch.mean(smoothness_y)


######################
# functional interface
######################


def inverse_depth_smoothness_loss(
        idepth: torch.Tensor,
        image: torch.Tensor) -> torch.Tensor:
    r"""Computes image-aware inverse depth smoothness loss.

    See :class:`~kornia.losses.InvDepthSmoothnessLoss` for details.
    """
    return InverseDepthSmoothnessLoss()(idepth, image)
