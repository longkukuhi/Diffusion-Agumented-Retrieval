import torch
import torch.nn.functional as F
from PIL import Image


class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """Project images to vectors and preprocess image files for the model."""
        self.model = model
        self.processor = preprocessor

def BLIP_BASELINE():
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    import sys
    sys.path.insert(0, './BLIP')
    from BLIP.models.blip_itm import blip_itm
    # blip_ckpt = 'model_base_retrieval_coco.pth'
    blip_ckpt = 'chatir_weights.ckpt'
    model = blip_itm(
        pretrained=blip_ckpt,
        med_config='BLIP/configs/med_config.json',
        image_size=224,
        vit='base'
    )

    print(f"BLIP model load {blip_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
    ])

    def blip_project_img(image):
        embeds = model.visual_encoder(image)
        projection = model.vision_proj(embeds[:, 0, :])
        return F.normalize(projection, dim=-1)

    def blip_prep_image(path):
        raw = Image.open(path).convert('RGB')
        return transform_test(raw)

    image_embedder = ImageEmbedder(blip_project_img, lambda path: blip_prep_image(path))

    def dialog_encoder(dialog):
        text = model.tokenizer(
            dialog,
            padding='longest',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        text_output = model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text'
        )

        shift = model.text_proj(text_output.last_hidden_state[:, 0, :])
        return F.normalize(shift, dim=-1)

    return dialog_encoder, image_embedder
