import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from ..data_util import crop_pc, voxelize
from ...transforms.point_transform_cpu import PointsToTensor
import glob
from tqdm import tqdm
import logging
import pickle


VALID_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191
]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (188., 189., 34.),
    3: (152., 223., 138.),
    4: (255., 152., 150.),
    5: (214., 39., 40.),
    6: (91., 135., 229.),
    7: (31., 119., 180.),
    8: (229., 91., 104.),
    9: (247., 182., 210.),
    10: (91., 229., 110.),
    11: (255., 187., 120.),
    13: (141., 91., 229.),
    14: (112., 128., 144.),
    15: (196., 156., 148.),
    16: (197., 176., 213.),
    17: (44., 160., 44.),
    18: (148., 103., 189.),
    19: (229., 91., 223.),
    21: (219., 219., 141.),
    22: (192., 229., 91.),
    23: (88., 218., 137.),
    24: (58., 98., 137.),
    26: (177., 82., 239.),
    27: (255., 127., 14.),
    28: (237., 204., 37.),
    29: (41., 206., 32.),
    31: (62., 143., 148.),
    32: (34., 14., 130.),
    33: (143., 45., 115.),
    34: (137., 63., 14.),
    35: (23., 190., 207.),
    36: (16., 212., 139.),
    38: (90., 119., 201.),
    39: (125., 30., 141.),
    40: (150., 53., 56.),
    41: (186., 197., 62.),
    42: (227., 119., 194.),
    44: (38., 100., 128.),
    45: (120., 31., 243.),
    46: (154., 59., 103.),
    47: (169., 137., 78.),
    48: (143., 245., 111.),
    49: (37., 230., 205.),
    50: (14., 16., 155.),
    51: (196., 51., 182.),
    52: (237., 80., 38.),
    54: (138., 175., 62.),
    55: (158., 218., 229.),
    56: (38., 96., 167.),
    57: (190., 77., 246.),
    58: (208., 49., 84.),
    59: (208., 193., 72.),
    62: (55., 220., 57.),
    63: (10., 125., 140.),
    64: (76., 38., 202.),
    65: (191., 28., 135.),
    66: (211., 120., 42.),
    67: (118., 174., 76.),
    68: (17., 242., 171.),
    69: (20., 65., 247.),
    70: (208., 61., 222.),
    71: (162., 62., 60.),
    72: (210., 235., 62.),
    73: (45., 152., 72.),
    74: (35., 107., 149.),
    75: (160., 89., 237.),
    76: (227., 56., 125.),
    77: (169., 143., 81.),
    78: (42., 143., 20.),
    79: (25., 160., 151.),
    80: (82., 75., 227.),
    82: (253., 59., 222.),
    84: (240., 130., 89.),
    86: (123., 172., 47.),
    87: (71., 194., 133.),
    88: (24., 94., 205.),
    89: (134., 16., 179.),
    90: (159., 32., 52.),
    93: (213., 208., 88.),
    95: (64., 158., 70.),
    96: (18., 163., 194.),
    97: (65., 29., 153.),
    98: (177., 10., 109.),
    99: (152., 83., 7.),
    100: (83., 175., 30.),
    101: (18., 199., 153.),
    102: (61., 81., 208.),
    103: (213., 85., 216.),
    104: (170., 53., 42.),
    105: (161., 192., 38.),
    106: (23., 241., 91.),
    107: (12., 103., 170.),
    110: (151., 41., 245.),
    112: (133., 51., 80.),
    115: (184., 162., 91.),
    116: (50., 138., 38.),
    118: (31., 237., 236.),
    120: (39., 19., 208.),
    121: (223., 27., 180.),
    122: (254., 141., 85.),
    125: (97., 144., 39.),
    128: (106., 231., 176.),
    130: (12., 61., 162.),
    131: (124., 66., 140.),
    132: (137., 66., 73.),
    134: (250., 253., 26.),
    136: (55., 191., 73.),
    138: (60., 126., 146.),
    139: (153., 108., 234.),
    140: (184., 58., 125.),
    141: (135., 84., 14.),
    145: (139., 248., 91.),
    148: (53., 200., 172.),
    154: (63., 69., 134.),
    155: (190., 75., 186.),
    156: (127., 63., 52.),
    157: (141., 182., 25.),
    159: (56., 144., 89.),
    161: (64., 160., 250.),
    163: (182., 86., 245.),
    165: (139., 18., 53.),
    166: (134., 120., 54.),
    168: (49., 165., 42.),
    169: (51., 128., 133.),
    170: (44., 21., 163.),
    177: (232., 93., 193.),
    180: (176., 102., 54.),
    185: (116., 217., 17.),
    188: (54., 209., 150.),
    191: (60., 99., 204.),
    193: (129., 43., 144.),
    195: (252., 100., 106.),
    202: (187., 196., 73.),
    208: (13., 158., 40.),
    213: (52., 122., 152.),
    214: (128., 76., 202.),
    221: (187., 50., 115.),
    229: (180., 141., 71.),
    230: (77., 208., 35.),
    232: (72., 183., 168.),
    233: (97., 99., 203.),
    242: (172., 22., 158.),
    250: (155., 64., 40.),
    261: (118., 159., 30.),
    264: (69., 252., 148.),
    276: (45., 103., 173.),
    283: (111., 38., 149.),
    286: (184., 9., 49.),
    300: (188., 174., 67.),
    304: (53., 206., 53.),
    312: (97., 235., 252.),
    323: (66., 32., 182.),
    325: (236., 114., 195.),
    331: (241., 154., 83.),
    342: (133., 240., 52.),
    356: (16., 205., 144.),
    370: (75., 101., 198.),
    392: (237., 95., 251.),
    395: (191., 52., 49.),
    399: (227., 254., 54.),
    408: (49., 206., 87.),
    417: (48., 113., 150.),
    488: (125., 73., 182.),
    540: (229., 32., 114.),
    562: (158., 119., 28.),
    570: (60., 205., 27.),
    572: (18., 215., 201.),
    581: (79., 76., 153.),
    609: (134., 13., 116.),
    748: (192., 97., 63.),
    776: (108., 163., 18.),
    1156: (95., 220., 156.),
    1163: (98., 141., 208.),
    1164: (144., 19., 193.),
    1165: (166., 36., 57.),
    1166: (212., 202., 34.),
    1167: (23., 206., 34.),
    1168: (91., 211., 236.),
    1169: (79., 55., 137.),
    1170: (182., 19., 117.),
    1171: (134., 76., 14.),
    1172: (87., 185., 28.),
    1173: (82., 224., 187.),
    1174: (92., 110., 214.),
    1175: (168., 80., 171.),
    1176: (197., 63., 51.),
    1178: (175., 199., 77.),
    1179: (62., 180., 98.),
    1180: (8., 91., 150.),
    1181: (77., 15., 130.),
    1182: (154., 65., 96.),
    1183: (197., 152., 11.),
    1184: (59., 155., 45.),
    1185: (12., 147., 145.),
    1186: (54., 35., 219.),
    1187: (210., 73., 181.),
    1188: (221., 124., 77.),
    1189: (149., 214., 66.),
    1190: (72., 185., 134.),
    1191: (42., 94., 198.),
}

_cmap = SCANNET_COLOR_MAP
_cmap = [(_cmap[i] if i in _cmap else (0,0,0)) for i in range(1192)]
_cmap = [np.array(i) for i in _cmap]

@DATASETS.register_module()
class ScanNet(Dataset):
    cmap = _cmap
    num_classes = 200
    classes = ['wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']
    gravity_dim = 2
    
    color_mean = [0.46259782, 0.46253258, 0.46253258]
    color_std =  [0.693565  , 0.6852543 , 0.68061745]
    """ScanNet dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (145841.0, 158783.87179487178, 84200.84445829492)
    """  
    def __init__(self,
                 data_root='data/ScanNet',
                 split='train',
                 voxel_size=0.04,
                 voxel_max=None,
                 transform=None,
                 loop=1, presample=False, variable=False,
                 n_shifted=1
                 ):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.presample = presample
        self.variable = variable
        self.loop = loop
        self.n_shifted = n_shifted
        self.pipe_transform = PointsToTensor() 

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(
                data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        elif split == 'test':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))

        logging.info("Totally {} samples in {} set.".format(
            len(self.data_list), split))

        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f'scannet_{split}_{voxel_size:.3f}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading ScanNet {split} split'):
                data = torch.load(item)
                coord, feat, label = data[0:3]
                coord, feat, label = crop_pc(
                    coord, feat, label, self.split, self.voxel_size, self.voxel_max, variable=self.variable)
                cdata = np.hstack(
                    (coord, feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
            # median, average, std of number of points after voxel sampling for val set.
            # (100338.5, 109686.1282051282, 57024.51083415437)
            # before voxel sampling
            # (145841.0, 158783.87179487178, 84200.84445829492)
    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = self.data_list[data_idx]
            data = torch.load(data_path)
            coord, feat, label = data[0:3]

        feat = (feat + 1) * 127.5
        label = label.astype(np.long).squeeze()
        data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'y': label}
        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        data['pos'], data['x'], data['y'] = crop_pc(
            data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
            downsample=not self.presample, variable=self.variable)
            
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3]], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3]])
        """
        if self.transform is not None:
            data = self.transform(data)
        
        if not self.presample: 
            data['pos'], data['x'], data['y'] = crop_pc(
                data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable)
        
        data = self.pipe_transform(data)
         
        if 'heights' not in data.keys():
            data['heights'] =  data['pos'][:, self.gravity_dim:self.gravity_dim+1] - data['pos'][:, self.gravity_dim:self.gravity_dim+1].min()
        return data

    def __len__(self):
        return len(self.data_list) * self.loop