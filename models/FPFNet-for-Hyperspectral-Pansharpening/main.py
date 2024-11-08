import argparse
import pprint
## PyTorch 版本: 1.10.1
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from huston_dataloader import My_dataset_huston
from lib.config import config
from lib.config import update_config
from lib.utils.utils import create_logger
from model_noback_huston import get_noback_model_huston
from scipy.io import savemat

device = torch.device("cuda:0")
learning_rate = 0.001

"""
数据集的根目录
"""
root = '/media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/yyh/FPF_revise/dataset/Huston'

batch_size = 8
epochs = 500


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))

    """
    加载模型
    """
    # model = get_noback_model(config).to(device)  # pavia
    model = get_noback_model_huston(config).to(device)  # huston
    # model = get_noback_model_Bost(config).to(device)  # Boston
    model = nn.DataParallel(model)
    optim_model = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optim_model, step_size=200, gamma=0.65)

    """
    加载训练集和验证集
    """
    train_data = My_dataset_huston(root, 'train')
    test_data = My_dataset_huston(root, 'test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=0)
    best_loss, running_loss, test_loss = 10, 0, 10
    criteon = nn.L1Loss()  # 损失函数

    """
    网络训练
    """
    global_step = 0
    for epoch in range(epochs):
        for step, (Pan, lrHS, ref) in enumerate(train_loader):
            Pan = Pan.type(torch.float).to(device)
            lrHS = lrHS.type(torch.float).to(device)
            ref = ref.type(torch.float).to(device)
            model.train()
            output = model(Pan, lrHS)
            running_loss = criteon(output, ref)
            optim_model.zero_grad()
            running_loss.backward()
            # optim_model.step()
            scheduler.step()
        global_step += 1
        if epoch % 1 == 0:
            if running_loss <= best_loss:
                best_loss = running_loss
                A = model.state_dict()
                torch.save(model.state_dict(), 'best.mdl')
                torch.save(model.state_dict(), 'net_best.pth')
            print(global_step, best_loss)
        if epoch >= 300 and epoch % 10 == 0:
            torch.save(model.state_dict(), 'best{}.mdl'.format(epoch))
            torch.save(model.state_dict(), 'net_model{}.pth'.format(epoch))

    """
    网络测试
    """
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load('best.mdl'))  # best 410 zuoyou
    model.eval()
    for test_step, (test_pan, test_lrhs, test_ref) in enumerate(test_loader):
        test_pan = test_pan.type(torch.float).to(device)
        test_lrhs = test_lrhs.type(torch.float).to(device)
        test_ref = test_ref.type(torch.float).to(device)
        with torch.no_grad():
            test_output = model(test_pan, test_lrhs)
            test_loss = criteon(test_output, test_ref)
            print(test_step+1, test_loss)
            savemat("%d.mat" % (test_step + 1), {'x': test_output.detach().cpu().numpy()})


if __name__ == '__main__':
    main()


