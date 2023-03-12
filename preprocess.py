# %%
from plyfile import PlyData, PlyElement
import pandas as pd
import numpy as np
import glob
import os
import torch
import sys

def read_plyfile(filepath):
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values

def write_plyfile(nparray, output_path, instance_id=True):
    '''
    nparray = array([[x, y, z, red, green, blue, label, (instance_id)],
                    ...,
                    [x, y, z, red, green, blue, label, (instance_id)]])
    output_path = '.../*.ply'
    '''
    if instance_id:
        dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', '<u4'), ('instance_id', '<u4')]
    else:
        dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', '<u4')]
    
    tmp = [tuple(a) for a in nparray]
    tmp = np.array(tmp, dtype=dtype)
    el = PlyElement.describe(tmp, 'vertex')
    PlyData([el]).write(output_path)


VALID_CLASS_IDS_200 = (
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191)


ValidClassIDS_to_idx = {}
for idx, ID in enumerate(VALID_CLASS_IDS_200):
    ValidClassIDS_to_idx[ID] = idx

def Preprocess(data_pth, root, test=False):
    num_per_class = [0]*201
    filenames = glob.glob(os.path.join(data_pth, '*.ply'))
    if test:
        os.makedirs(os.path.join(root, 'test'), exist_ok=True)
    else:
        os.makedirs(os.path.join(root, 'train'), exist_ok=True)
        os.makedirs(os.path.join(root, 'val'), exist_ok=True)
        val_scence_list = [f'{i:04d}' for i in np.random.choice(a=499, size=50, replace=False)]

    for i, fn in enumerate(filenames):
        _, tail = os.path.split(fn)
        scene, num = tail.rsplit('e', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)
        points = read_plyfile(fn)
        
        if test:
            coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
            colors = np.ascontiguousarray(points[:, 3:]) / 127.5 - 1
            torch.save((coords, colors), os.path.join(root, 'test', tail.rsplit('.', 1)[0] + '.pth'))
        else:
            coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
            colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
            for point in points:
                if int(point[6]) in VALID_CLASS_IDS_200:
                    point[6] = ValidClassIDS_to_idx[int(point[6])]
                else:
                    point[6] = -100
                    
                num_per_class[int(point[6])] += 1
            labels = np.ascontiguousarray(points[:, 6])
            instance_ids = np.ascontiguousarray(points[:, 7])

            if scene in val_scence_list:
                torch.save((coords, colors, labels, instance_ids), os.path.join(root, 'val', tail.rsplit('.', 1)[0] + '.pth'))
            else:
                torch.save((coords, colors, labels, instance_ids), os.path.join(root, 'train', tail.rsplit('.', 1)[0] + '.pth'))
        print(f"Processing ... [{i+1}/{len(filenames)}]", end='\r')

data_root, out_root = sys.argv[1], 'dataset_preprocessed'
train_data_pth = os.path.join(data_root, 'train')
test_data_pth = os.path.join(data_root, 'test')
os.makedirs(out_root, exist_ok=True)

Preprocess(train_data_pth, out_root, test=False)
Preprocess(test_data_pth, out_root, test=True)