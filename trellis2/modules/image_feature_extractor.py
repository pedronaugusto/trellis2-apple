from typing import *
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._device = 'cpu'

    def to(self, device):
        self._device = device
        self.model.to(device)

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    @property
    def device(self):
        return self._device

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self._device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).to(self._device)
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._device = 'cpu'

    def to(self, device):
        self._device = device
        self.model.to(device)

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    @property
    def device(self):
        return self._device

    @property
    def _encoder(self):
        """The module that owns `embeddings`, `rope_embeddings` and `layer`.

        transformers moved this: up to ~5.5 DINOv3ViTModel exposes them at the
        top level, while later 5.x (observed on 5.14) nests them one deeper
        under `.model`, leaving only `embeddings`/`rope_embeddings`/`model`/
        `norm` on the outer object. Resolve it by looking for `layer` rather
        than pinning a transformers version, so both layouts work.
        """
        if hasattr(self.model, 'layer'):
            return self.model
        inner = getattr(self.model, 'model', None)
        if inner is not None and hasattr(inner, 'layer'):
            return inner
        raise AttributeError(
            "Could not locate the transformer blocks on "
            f"{type(self.model).__name__}: expected `.layer` or `.model.layer`. "
            "This usually means a transformers version with a layout neither "
            "branch handles — please open an issue with your transformers version."
        )

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        encoder = self._encoder
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(encoder.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self._device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).to(self._device)
        features = self.extract_features(image)
        return features
