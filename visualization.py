import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

from dataset import MarineCreatureDataset


def visualize(opt, dataset):
    random.seed(170)
    model = torch.load(opt.pretrained_model)
    model.to(opt.device)
    model.eval()
    if opt.heatmap:
        if opt.layer:
            cam_extractor = SmoothGradCAMpp(model, opt.layer)
        else:
            cam_extractor = SmoothGradCAMpp(model)
    for i in range(1, opt.image_per_file+1):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        img, img_path, label_str = sample['img'], sample['img_path'], sample['label_str']
        img = img.to(opt.device).unsqueeze(0)
        label_pred = model(img)
        label_pred_np = torch.argmax(label_pred, dim=1).detach().cpu().numpy()
        label_str_pred = MarineCreatureDataset.label_map.inverse[label_pred_np[0]]
        original_img = Image.open(img_path)
        if not opt.heatmap:              
            plt.subplot(1, opt.image_per_file, i)
            plt.axis('off')
            plt.title(f'pred: {label_str_pred}')
            plt.imshow(original_img)
        else:
            activation_map = cam_extractor(torch.argmax(label_pred, dim=1).item(), label_pred)
            result = overlay_mask(original_img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.subplot(2, opt.image_per_file, i)
            plt.axis('off')
            plt.title(f'pred: {label_str_pred}\n real: {label_str}', fontsize='small')
            plt.imshow(original_img)
            plt.subplot(2, opt.image_per_file, opt.image_per_file + i)
            plt.axis('off')
            plt.title(f'Heatmap')
            plt.imshow(result)

    plt.tight_layout()
    plt.savefig(os.path.join(opt.save_dir, opt.image_basename))
