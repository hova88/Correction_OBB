import torch
import pickle
import argparse
from tqdm import tqdm

from stats_utils import *

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument(
    "--result_file",
    type=str,
    default='data/result_lyft_bus_20.pkl',
    help="specify the pkl file for view",
)
args = parser.parse_args()
file_name = args.result_file.split("/")[-1]


with open(args.result_file , 'rb') as f:
    infos = pickle.load(f)

print("{} contained {} results for statistics".format(file_name.upper(),len(infos)))
print("-----------------Start Analysis------------------------")
pred_boxes = torch.zeros(len(infos),7)
fake_boxes = torch.zeros(len(infos),7)
gt_boxes   = torch.zeros(len(infos),7)
for idx in tqdm(range(len(infos))):
    gt_box = torch.from_numpy(infos[idx]['gt_boxes'])[:,:7]
    fake_box = torch.from_numpy(infos[idx]['fake_boxes'])[:,:7]
    pred_box = torch.from_numpy(infos[idx]['pred_boxes'])[:,:7]

    gt_boxes[idx] = gt_box
    fake_boxes[idx] = fake_box
    pred_boxes[idx] = pred_box
    pred_boxes = torch.where(torch.isnan(pred_boxes), torch.full_like(pred_boxes, 0), pred_boxes)
pred_iou,fake_iou,diff_iou = calculate_iou(pred_boxes , fake_boxes , gt_boxes)
interval , quantity = statistics_diff_iou(fake_iou,diff_iou , step = 0.001)
print("-----------------Finish Analysis------------------------")
print("Draw histogram of {}...".format(file_name.upper()))
plot_histogram(interval , quantity , width = 0.001)

