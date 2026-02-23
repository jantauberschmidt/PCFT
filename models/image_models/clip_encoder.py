import torch.nn as nn
import torchvision
from torchvision.transforms.functional import InterpolationMode

from transformers import CLIPVisionModelWithProjection
from huggingface_hub import PyTorchModelHubMixin


class ClipEncoder(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="imscore",
    repo_url="https://github.com/RE-N-Y/imscore"
):
    def __init__(self):
        super().__init__()

        self._exclude_from_saving = True

        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.resize = torchvision.transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)
        self.crop = torchvision.transforms.CenterCrop(224)
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                          std=[0.26862954, 0.26130258, 0.27577711])


    def forward(self, pixels):
        # assume pixels is between 0 and 1
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)

        pixels = self.resize(pixels)
        pixels = self.crop(pixels)
        pixels = self.normalize(pixels)
        embed = self.clip(pixel_values=pixels).image_embeds
        return embed

    def cosine_distance(self, img_a, img_b):
        ea = self.forward(img_a)
        ea = ea / ea.norm(dim=1, keepdim=True)

        eb = self.forward(img_b)
        eb = eb / eb.norm(dim=1, keepdim=True)
        return 1.0 - (ea * eb).sum(dim=1)



if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms.functional as TF

    clip = ClipEncoder().eval()

    img_path = "clock.jpg"
    img_pil = Image.open(img_path).convert("RGB")
    img = TF.to_tensor(img_pil)

    emb = clip(img)

