from src.models import ModelRegistry, BaseModel
import torch
from vit_pytorch import SimpleViT

@ModelRegistry("vit")
class VitWrapper(BaseModel):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop('led_inference')
        super().__init__(*args, **kwargs)

        self.vit = SimpleViT(
            image_size = 640,
            patch_size = 32,
            num_classes = 6,
            dim = 1024,
            depth = 3,
            heads = 16,
            mlp_dim = 2048
        )


    
    def forward(self, image: torch.Tensor) -> torch.Any:
        im = torch.nn.functional.pad(image, (0, 0, (640 - 360) // 2, (640 - 360) // 2))
        out = self.vit(im)
        out = torch.nn.functional.sigmoid(out)
        return out
   
    def predict_leds(self, outs, batch):
        return outs
    
    def predict_pos_from_outs(self, image, out):
        bs = image.shape[0]
        return torch.zeros((bs, 2)).numpy()

    def predict_orientation_from_outs(self, out):
        bs = out.shape[0]
        return torch.zeros((bs,)).numpy(), torch.zeros((bs,)).numpy(), torch.zeros((bs,)).numpy(),


    def loss(self, batch, model_out):
        supervised_label = batch["supervised_flag"].to(model_out.device)
        led_trues = batch["led_mask"].to(model_out.device)

        led_preds = model_out
        led_losses = torch.zeros_like(led_trues, device=model_out.device, dtype=torch.float32)
        for i in range(led_preds.shape[1]):
            led_losses[:, i] = torch.nn.functional.binary_cross_entropy(
                    led_preds[:, i], led_trues[:, i].float(), reduction='none')
        
        led_loss = led_losses.mean(-1) * ~supervised_label
        return torch.tensor([0], device = model_out.device, dtype=torch.float32),\
                torch.tensor([0], device = model_out.device, dtype=torch.float32),\
                torch.tensor([0], device = model_out.device, dtype=torch.float32),\
                led_loss, led_losses.detach().mean(0)


