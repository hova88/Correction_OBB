import pickle
import argparse
from tqdm import tqdm

from src import *

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument(
    "--data_path",
    type=str,
    default="data/lyft_bus_20.pkl"
    help="specify the pkl file for view",
)
parser.add_argument(
    "--save_path",
    type=str,
    default='./data/result_',
    help="specify the pkl file for view",
)
args = parser.parse_args()
file_name = args.data_path.split("/")[-1]

with open(args.data_path , 'rb') as f:
    infos = pickle.load(f)

print("{} contained {} samples for correction".format(file_name.upper(),len(infos)))
print("-----------------------------------------")
for info in tqdm(infos):
    points = info['points'][:,:3]
    points_in_fb = points[info['fk_mask']]
    fk_boxes = np.copy(info['fake_boxes'])
    info['pred_boxes'] = get_pred_box(points , points_in_fb , fk_boxes).reshape(1,7)

if args.save_path == './data/result_':
    args.save_path += file_name  

with open(args.save_path , 'wb') as f:
    pickle.dump(infos,f)
print("-----------------------------------------")
print("The result has been dump as {} ".format(args.save_path))