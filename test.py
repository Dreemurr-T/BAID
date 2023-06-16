import scipy.stats
from data import BBDataset
from torch.utils.data import DataLoader
from model import SAAN
import torch.nn as nn
import torch
import torch.optim as optim
from common import *
import argparse
import torch.nn.functional as F
import pandas as pd
import scipy
from tqdm import tqdm

test_dataset = BBDataset(file_dir='test', test=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/SAAN')
    parser.add_argument('--checkpoint_name', type=str,
                        default='model_best.pth')

    return parser.parse_args()


def test(args):
    device = args.device
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    df = pd.read_csv('test/data.csv')
    predictions = []

    model = SAAN(num_classes=1)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    with torch.no_grad():
        for step, test_data in tqdm(enumerate(test_loader)):
            image = test_data[0].to(device)

            predicted_label = model(image)
            prediction = predicted_label.squeeze().cpu().numpy()
            predictions.append(prediction * 10)

    scores = df['score'].values.tolist()

    print(scipy.stats.spearmanr(scores, predictions))
    print(scipy.stats.pearsonr(scores, predictions))

    acc = 0
    for i in range(len(scores)):
        cls1 = 1 if scores[i] > 5 else 0
        cls2 = 1 if predictions[i] > 5 else 0
        if cls1 == cls2:
            acc += 1
    print(acc/len(scores))

    df.insert(loc=2, column='prediction', value=predictions)
    df.to_csv('test/result.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    test(args)
