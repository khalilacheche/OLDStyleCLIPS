import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms


class FaceSegmentation:
    """
    This class is a wrapper for generating segmentation output from
    face images. It uses a pretrained BiSeNet model from this repository
    https://github.com/zllrunning/face-parsing.PyTorch
    """

    part_to_mask_idx = {
        "background": 0,
        "skin": 1,
        "l_brow": 2,
        "r_brow": 3,
        "l_eye": 4,
        "r_eye": 5,
        "eyeglass": 6,
        "l_ear": 7,
        "r_ear": 8,
        "earrings": 9,
        "nose": 10,
        "mouth": 11,
        "u_lip": 12,
        "l_lip": 13,
        "neck": 14,
        "necklace": 15,
        "cloth": 16,
        "hair": 17,
        "hat": 18,
    }

    preprocess_transformation = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def __init__(self, face_bisenet, device):
        """
        Parameters
        ----------
        face_bisenet: pretrained model for face segmentation
        device: torch device used for performing the computation
        """
        self.face_bisenet = face_bisenet
        self.device = device
        self.face_bisenet = self.face_bisenet.to(device)

    def zero_grad(self):
        self.face_bisenet.zero_grad()

    @torch.no_grad()
    def predict(self, pil_images, one_hot=False):
        """
        Parameters
        ----------
        pil_images: {list | pil Image}, list of pil images
            or a single to predict the face segmentation
        one_hot, bool, default=False
            Whether to return the result in one_hot mode or idx mode
        """
        if isinstance(pil_images, list):
            x = [self.preprocess_transformation(pil_image) for pil_image in pil_images]
        else:
            x = [self.preprocess_transformation(pil_images)]

        x = torch.stack(x, dim=0)
        x = x.to(self.device)
        # TODO: Add one_hot functionality
        # TODO: Change prediction to mini batches to avoid out of memory error
        y = self.face_bisenet(x)[0]
        y = y.detach()
        y = torch.argmax(y, dim=1, keepdim=True)
        return y
