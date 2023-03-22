import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import MarineCreatureDataset
from options import InferenceOptions
from visualization import visualize


def inference():
    
    opt = InferenceOptions().parse()
    
    test_dataset = MarineCreatureDataset(image_dir=opt.image_dir, image_size=opt.image_size, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    loaders = test_loader
    print(f'Finish loading dataset.')

    model = torch.load(opt.pretrained_model)
    model.to(opt.device)

    criterion = opt.criterion
    
    model.eval()

    inference_loss = []
    count_label = {l:0 for l in MarineCreatureDataset.label_map}
    acc_pred_label = {l:0 for l in MarineCreatureDataset.label_map}

    for data in tqdm(loaders, unit='batch'):
        img, label = data['img'], data['label']
        img, label = img.to(opt.device), label.to(opt.device)

        with torch.no_grad():
            label_pred = model(img)
            loss = criterion(label_pred, label)
            inference_loss.append(loss.item())
            label_pred_np = torch.argmax(label_pred, dim=1).detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            for l in MarineCreatureDataset.label_map:
                id = MarineCreatureDataset.label_map[l]
                count_label[l] += np.sum(label_np == id)
                acc_pred_label[l] += np.sum(np.logical_and(label_pred_np == label_np, label_np == id))

    avg_loss = np.mean(np.array(inference_loss))
    acc_label = {l:(acc_pred_label[l] / count_label[l]) for l in MarineCreatureDataset.label_map}
    acc = sum(acc_pred_label.values()) / len(test_dataset)
    print(f'Inference Loss: {avg_loss} Average Accuracy: {acc*100:.2f}%')
    for l in MarineCreatureDataset.label_map:
        print(f'Label: {l:18} Recall: {acc_label[l]*100:.2f}%')
    
    if opt.visualize:
        visualize(opt, dataset=test_dataset)


if __name__ == '__main__':

    inference()
