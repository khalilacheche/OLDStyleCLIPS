import torch
import clip


class CLIPLoss(torch.nn.Module):
    """
    Custom module for calculating the CLIP loss.

    Args:
        opts (argparse.Namespace): Options and settings for the CLIP loss.

    """

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        """
        Performs forward pass of the CLIP loss.

        Args:
            image (torch.Tensor): Generated image tensor.
            text (torch.Tensor): Text input tensor.

        Returns:
            torch.Tensor: Similarity score between the generated image and the text input.

        """
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity
