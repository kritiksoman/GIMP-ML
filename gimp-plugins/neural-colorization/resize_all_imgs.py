from multiprocessing import Pool
from PIL import Image
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Resize all colorful imgs to 256*256 for training")
    parser.add_argument("-d",
                        "--dir",
                        required=True,
                        type=str,
                        help="The directory includes all jpg images")
    parser.add_argument("-n",
                        "--nprocesses",
                        default=10,
                        type=int,
                        help="Using how many processes")
    args = parser.parse_args()
    return args

def doit(x):
    a=Image.open(x)
    if a.getbands()!=('R','G','B'):
        os.remove(x)
        return
    a.resize((256,256),Image.BICUBIC).save(x)
    return

args=parse_args()
pool = Pool(processes=args.nprocesses)
jpgs = []
flist = os.listdir(args.dir)
full_flist = [os.path.join(args.dir,x) for x in flist]
pool.map(doit, full_flist)
print('done')