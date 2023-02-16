import glob
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage import io
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

test_list = glob.glob('Data_test/rawData_test/*')
size = 512

def func(p1, p2):
    p1 = np.array(Image.fromarray(np.array(Image.open(p1))[:, :, 0]).resize((size, size))).reshape(size * size)
    p2 = np.array(Image.fromarray(np.array(Image.open(p2))[:, :]).resize((size, size))).reshape(size * size)
    p1 = np.where(p1 >= 200, 0, 255)
    p2 = np.where(p2 <= 0, 0, 255)

    # p1 = p1.reshape(size, size)
    # io.imsave('test.png', p1)
    # exit()
    return p1 // 255, p2 // 255


def cal_single_image(name):
    dr, p, f = [], [], []
    name = name.split('/')[-1]
    output_root = 'Results/test/' + name + '_CLSTM'
    output_list = glob.glob(output_root + '/output*.png')
    for i in tqdm(range(len(output_list))):
        num = output_list[i].split('output')[1].split('.')[0]
        p1, p2 = func(output_list[i], 'Data_test/rawData_test/' + name + '/vessel_' + str(int(num) + 1) + '.png')
        mask = p1
        output = p2
        dr.append(recall_score(mask, output, average='macro'))
        p.append(precision_score(mask, output, average='macro'))
        f.append(f1_score(mask, output, average='macro'))
    return dr, p, f

DR = []
P = []
F = []
for name in test_list:
    dr, p, f = cal_single_image(name)
    DR += dr
    P += p
    F += f
print(sum(DR) / len(DR), sum(P) / len(P), sum(F) / len(F))