from test import generate_single_sequence
from tqdm import tqdm
import glob

test_list = glob.glob('Data_test/rawData_test/*')
for name in tqdm(test_list):
    print(name)
    generate_single_sequence(name)
