import torch
from torch import nn
from criteria.utils.segmentation_utils import *
from models.facial_recognition.model_irse import Backbone
import torchvision
from PIL import Image


def _load_face_bisenet_model(model_path):
    """
    You can download the pretrained model from this repository
    https://github.com/zllrunning/face-parsing.PyTorch
    """
    from models.face_bisenet.model import BiSeNet

    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def raw_image_to_pil_image(raw_img):
    img = (raw_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_imgs = []
    for i in range(len(img)):
        pil_imgs.append(Image.fromarray(img[i].cpu().numpy(), "RGB"))
    return pil_imgs


class LocalizationLoss(nn.Module):
    """
    Custom module for calculating the localization loss.
    The localization loss is calculated by calculating the mean squared difference between the pixels that are outside of the semantic parts that we want to edit.
    The loss is higher the more the pixels outside of the semantic parts are changed.

    Args:
        opts (argparse.Namespace): Options and settings for the localization loss.

    """

    def __init__(self, opts):
        super(LocalizationLoss, self).__init__()
        self.opts = opts
        print("Loading Segmentation Models")
        self.device = torch.device("cuda")

        segmentation_model = _load_face_bisenet_model(
            "pretrained/face_bisenet/model.pth"
        )
        self.segmentation_model = FaceSegmentation(segmentation_model, self.device)
        self.semantic_parts = opts.semantic_parts

    def get_semantic_parts(self, text):
        # returns the semantic parts according to the given text
        parts = {
            "mouth": ["mouth", "u_lip", "l_lip"],
            "skin": ["skin"],
            "eyes": ["l_eye", "r_eye"],
            "nose": ["nose"],
            "ears": ["l_ear", "r_ear", "earrings"],
            "eye_brows": ["l_brow", "r_brow"],
            "hat": ["hair", "hat"],
            "hair": ["hair"],
            "neck": ["cloth", "neck", "necklace"],
        }
        semantic_parts = []
        for semantic_part in self.semantic_parts:
            if semantic_part in parts.keys():
                semantic_parts += parts[semantic_part]

        return semantic_parts

    ### Batch data should now be coming from the generator, instead of the direct image outoput of the gan
    def forward(self, batch_data, new_batch_data, text, i):
        """
        Performs forward pass of the localization loss.

        Args:
            batch_data (dict): Batch data from the generator.
            new_batch_data (dict): New batch data from the generator.
            text (str): Input text.
            i (int): Index in the optimization.

        Returns:
            torch.Tensor: Loss value.

        """
        last_layer_res = None
        localization_loss = 0
        localization_layers = list(range(1, 10))
        # Set the weights for the localization layers
        # for now, as we are only using the last layer, the weight is 1 for the last layer and 0 for all other layers
        localization_layer_weights = np.array(
            [0.0] * (len(localization_layers) - 1) + [1.0]
        )
        layer_to_resolution = {
            0: 4,
            1: 4,
            2: 8,
            3: 16,
            4: 32,
            5: 64,
            6: 128,
            7: 256,
            8: 512,
            9: 1024,
        }

        old_segmentation_output = self.segmentation_model.predict(
            raw_image_to_pil_image(batch_data["image"]), one_hot=False
        )
        segmentation_output_res = old_segmentation_output.shape[2]
        new_segmentation_output = self.segmentation_model.predict(
            raw_image_to_pil_image(new_batch_data["image"]), one_hot=False
        )

        semantic_parts = self.get_semantic_parts(text)

        part_ids = [
            self.segmentation_model.part_to_mask_idx[part_name]
            for part_name in semantic_parts
        ]

        old_mask = 0.0
        for part_idx in part_ids:
            old_mask += 1.0 * (old_segmentation_output == part_idx)

        new_mask = 0.0
        for part_idx in part_ids:
            new_mask += 1.0 * (new_segmentation_output == part_idx)

        combined_mask = old_mask

        mask = combined_mask.clone()

        # Calculate the loss for each layer of the generator
        # For now, the calculation is done for the last layer only, but it can be extended to all layers
        localization_layers = localization_layers[-1:]
        localization_layer_weights = [localization_layer_weights[-1]]
        for layer, layer_weight in zip(
            reversed(localization_layers), reversed(localization_layer_weights)
        ):
            layer_res = layer_to_resolution[layer]
            if last_layer_res != layer_res:
                if layer_res != segmentation_output_res:
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=(layer_res, layer_res),
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    mask = combined_mask.clone()
            last_layer_res = layer_res
            if layer_weight == 0:
                continue

            x1 = batch_data[f"layer_{layer}"].detach()
            x2 = new_batch_data[f"layer_{layer}"]

            diff = torch.mean(torch.square(x1 - x2), dim=1)

            indicator = 1 - mask[:, 0]

            localization_loss = (
                layer_weight * torch.sum(diff * indicator, dim=[1, 2])[0]
            )
            localization_loss = torch.mean(localization_loss)

        pil_to_tensor = torchvision.transforms.ToTensor()

        old_image = pil_to_tensor(raw_image_to_pil_image(batch_data["image"])[0])
        old_mask_image = (
            torch.nn.functional.interpolate(
                old_mask, size=(1024, 1024), mode="bilinear", align_corners=True
            )[0]
            .repeat(3, 1, 1)
            .detach()
            .cpu()
        )
        new_image = pil_to_tensor(raw_image_to_pil_image(new_batch_data["image"])[0])
        new_mask_image = (
            torch.nn.functional.interpolate(
                new_mask, size=(1024, 1024), mode="bilinear", align_corners=True
            )[0]
            .repeat(3, 1, 1)
            .detach()
            .cpu()
        )
        if (
            self.opts.export_segmentation_image
            and self.opts.save_intermediate_image_every > 0
            and (i + 1) % self.opts.save_intermediate_image_every == 0
        ):
            images = [old_image, old_mask_image, new_image, new_mask_image]
            image_grid = torchvision.utils.make_grid(images, nrow=2)
            torchvision.utils.save_image(
                image_grid,
                os.path.join(
                    self.opts.results_dir,
                    f"seg_{str(i).zfill(5)}.jpg",
                ),
            )

        return localization_loss
