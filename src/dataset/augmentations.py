from functools import reduce
from typing import Any
import torch
from torchvision.transforms.v2 import ColorJitter, functional as F, InterpolationMode
import math


class RandomHorizontalFlip():
    def __init__(self, size):
        self.size = size

    def __call__(self, batch):
        if torch.rand(1) < .5:
            batch['proj_uvz'][0] = batch['image'].shape[-1] - batch['proj_uvz'][0]
            batch['image'] = F.hflip(batch['image'])
            batch['pos_map'] = F.hflip(batch['pos_map'])
            batch["pose_rel"][-1] = -batch["pose_rel"][-1]
            batch["pose_rel"][1] = -batch["pose_rel"][1]
            
            swap = batch["led_bl"]
            batch["led_bl"] = batch["led_br"]
            batch["led_br"] = swap

            swap = batch["led_tl"]
            
            batch["led_tl"] = batch["led_tr"]
            batch["led_tr"] = swap
            
            swap = batch["led_mask"][1] 
            batch["led_mask"][1] = batch["led_mask"][2]
            batch["led_mask"][2] = swap

            swap = batch["led_mask"][-2] 
            batch["led_mask"][-2] = batch["led_mask"][-1]
            batch["led_mask"][-1] = swap

            if 'led_visibility_mask' in batch:
                swap = batch["led_visibility_mask"][1] 
                batch["led_visibility_mask"][1] = batch["led_visibility_mask"][2]
                batch["led_visibility_mask"][2] = swap

                swap = batch["led_visibility_mask"][-2] 
                batch["led_visibility_mask"][-2] = batch["led_visibility_mask"][-1]
                batch["led_visibility_mask"][-1] = swap

        return batch
    


def simplex(images, frequency,
            fade=lambda t: t ** 3 * (t * (6 * t - 15) + 10)):
    # note: frequency must be int and a divisor of largest image dimension
    bs, c, h, w = images.size()
    side = max(h, w)
    size = float(frequency) / side
    invsize = int(1 / size)
    grid = torch.stack(torch.meshgrid(
        [torch.arange(side, device=images.device) * size] * 2, indexing='ij'), dim=-1) % 1
    angles = 2 * torch.pi * \
        torch.rand(int(side * size) + 1, int(side * size) +
                   1, device=images.device)
    # gradients
    gradients = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    gradients = torch.repeat_interleave(gradients, invsize, 0)
    gradients = torch.repeat_interleave(gradients, invsize, 1)
    g00 = gradients[:-invsize, :-invsize]
    g10 = gradients[+invsize:, :-invsize]
    g01 = gradients[:-invsize, +invsize:]
    g11 = gradients[+invsize:, +invsize:]
    # ramps
    n00 = (torch.stack([grid[..., 0], grid[..., 1]],
                       dim=-1) * g00).sum(dim=-1)
    n10 = (torch.stack([grid[..., 0] - 1, grid[..., 1]],
                       dim=-1) * g10).sum(dim=-1)
    n01 = (torch.stack([grid[..., 0], grid[..., 1] - 1],
                       dim=-1) * g01).sum(dim=-1)
    n11 = (torch.stack(
        [grid[..., 0] - 1, grid[..., 1] - 1], dim=-1) * g11).sum(dim=-1)
    # interpolation
    t = fade(grid)
    n0 = n00 * (1 - t[..., 0]) + t[..., 0] * n10
    n1 = n01 * (1 - t[..., 0]) + t[..., 0] * n11
    grid = math.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)
    grid = grid.repeat(bs, c, 1, 1)

    # note: noise is in [-1, 1], but is not normalized here
    return grid


class SimplexNoiseTransform():
    def __init__(self, size) -> None:
        self.size = size
        self.length = 1260
        self.n_maps = 3
        self.blends = torch.tensor([.6, .4, .4])
        self.frequencies = torch.tensor([20, 42, 126])
        # valid frequencies for 1260: 1 2 4 5 7 9 10 14 15 18 20 21 28 30 35 36 42 45
        #                             60 63 70 84 90 105 126 140 180 210 252 315 420 630

        self.noise_maps = [
            1 - simplex(torch.zeros(1, 1, self.length, self.length), f)[0, 0].clamp(0, 1)
            for f in self.frequencies]

    def __call__(self, batch):
        u_indices = torch.randint(0, self.length - self.size[1],
                                (self.n_maps, 2))
        v_indices = torch.randint(0, self.length - self.size[0],
                                (self.n_maps, 2))
        
        blends = torch.rand(self.n_maps) * self.blends
        noise = [n[v[0]:v[0] + self.size[0], u[1]:u[1] + self.size[1]] ** b
                 for (n, b, u, v) in zip(self.noise_maps, blends, u_indices, v_indices)]
        batch['image'] *= reduce(lambda x, y: x * y, noise, 1)
        return batch

class RandomRotTranslTransform():
    def __init__(self, max_angle, max_translate, bound):
        self.max_angle = max_angle
        self.max_translate = max_translate
        self.bound = bound

    def __call__(self, batch):
        image = batch['image']
        size = torch.tensor(image.shape[-2:]) # [V,U]

        angle = (2 * torch.rand(1) - 1) * self.max_angle

        translate = (2 * torch.rand(2) - 1) * self.max_translate * size

        sin = torch.sin(angle * torch.pi / 180).item()
        cos = torch.cos(angle * torch.pi / 180).item()
        u = batch["proj_uvz"][0] - size[1] / 2
        v = batch["proj_uvz"][1] - size[0] / 2

        batch["proj_uvz"][0] = u * cos - v * sin + translate[0].item() + size[1] / 2
        batch["proj_uvz"][1] = u * sin + v * cos + translate[1].item() + size[0] / 2

        angle = angle.float().item()
        translate = translate.tolist()

        image = F.affine(image, angle, translate, scale=1, shear=(0, 0),
                         interpolation=InterpolationMode.BILINEAR)
        pos_map = F.affine(batch['pos_map'][None, ...], angle, translate, scale=1, shear=(0, 0),
                         interpolation=InterpolationMode.BILINEAR)

        batch['image'] = image
        batch['pos_map'] = pos_map.squeeze()

        return batch
    

class ColorJitterAugmentation(ColorJitter):

    def __call__(self, batch) -> Any:
        transformed_image = super().__call__(batch['image'])
        batch['image'] = transformed_image
        return batch
    
