import argparse

paser = argparse.ArgumentParser()
"parameter Setting"

paser.add_argument('--dataroot', type = str, default = 'data')
paser.add_argument('--mode', type = str, default = 'train')
paser.add_argument('--batchsize', type = int, default = 8)
paser.add_argument('--img_size', type = int, default = 120, help = 'image size')
paser.add_argument('--lr', type = float, default = 0.0005, help = 'learning rate')
paser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'wegiht decay')
paser.add_argument('--epoch', type = float, default = 500, help = 'the number of iteration')
paser.add_argument('--show_interview', type = int, default = 5, help = 'show frequency')

