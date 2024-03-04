import clip
from PIL import Image
import torch
from src.models import ModelRegistry, BaseModel

@ModelRegistry("clip")
class ClipHead(BaseModel):
    def __init__(self):
        super(ClipHead, self).__init__(task = '')

        self.linear = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(in_features=512, out_features = 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(out_features = 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.Sigmoid()

        )
        self.clip_model, self.clip_preprocessor = clip.load("ViT-B/32", download_root = "/tmp/clip/")
        self.clip_model.eval()
        self.device= 'cpu'

        self.uv_dims = torch.tensor([[640, 320]])
        self.uv_diag = torch.sqrt((self.uv_dims ** 2).sum(1))


    def forward(self, x):
        image_array = torch.zeros_like(x, dtype=torch.uint8)
        image_array = (x * 255).to(torch.uint8).split(1)
        pil_images = [Image.fromarray(img.cpu().numpy().squeeze(), mode="RGB") for img in image_array]
        out = torch.stack(
            [self.clip_preprocessor(img) for img in pil_images]
        ).to(self.device)
        embeddings = self.clip_model.encode_image(out).to(torch.float32).detach()
        out =  self.linear(embeddings)
        proj_out = torch.nn.functional.sigmoid(out[..., :2])
        ori_out = torch.nn.functional.tanh(out[..., 2:4])
        dist_out  = torch.nn.functional.sigmoid(out[..., :4]) * self.MAX_DIST_M

        out = torch.cat([proj_out, ori_out, dist_out], axis = 1)
        out[:, :2] *= self.uv_dims
        return out


    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        self.clip_model = self.clip_model.to(*args, **kwargs)
        self.uv_dims = self.uv_dims.to(*args, **kwargs)
        self.uv_diag = self.uv_diag.to(*args, **kwargs)
        self.device = next(res.parameters()).device
        return res

    def loss(self, batch, out):
        proj_out = out[..., :2]
        ori_cos_out = out[..., 2]
        ori_sin_out = out[..., 3]
        dist_out  = out[..., 4]
    
        proj_true = batch["proj_uvz"][..., :-1].to(self.device)
        dist_true = batch["distance_rel"].to(self.device)
        ori_true = batch["pose_rel"][:, -1].to(self.device)
        cos_true = torch.cos(ori_true)
        sin_true = torch.sin(ori_true)

        proj_loss = (torch.sqrt(torch.pow(proj_true - proj_out, 2).sum(1)).mean() / self.uv_diag).squeeze()
        ori_cos_loss = torch.nn.functional.mse_loss(ori_cos_out, cos_true)
        ori_sin_loss = torch.nn.functional.mse_loss(ori_sin_out, sin_true)
        ori_loss = (ori_cos_loss + ori_sin_loss) / 4
        dist_loss = torch.nn.functional.mse_loss(dist_out, dist_true) / self.MAX_DIST_M ** 2

        return proj_loss, dist_loss, ori_loss

    def save_checkpoint(self, path, **kwargs):
        torch.save({
            'model_state_dict' : self.linear.state_dict(),
            'model_name' : self.model_name,
            **kwargs
        }, path)

    def load_from_checkpoint(self, data):
        self.linear.load_state_dict(data["model_state_dict"])
        return data

    def optimizer(self, learning_rate):
        return torch.optim.Adam(self.linear.parameters(), lr=learning_rate)


    def predict_pos_from_outs(self, outs):
        return outs[..., :2].detach().cpu().numpy()
    
    def predict_orientation_from_outs(self, outs):
        return torch.atan2(outs[..., 4], outs[..., 5]).detach().cpu().numpy()[None, ...]
    
    def predict_dist_from_outs(self, outs):
        return outs[..., 2:3].detach().cpu().numpy()
    
    def predict_leds_from_outs(self, outs):
        return torch.ones(outs.shape[0])

    