import torch
import time
import numpy as np
import scipy.io as sio
from Spa_downs import *
from LoadDataset_Batch import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
import matplotlib.pyplot as plt
from torch.nn.functional import upsample
from torch.nn.functional import interpolate
from function import *
import torchvision.transforms as transforms
from Network import *
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

def test(val_loader, model, P, WS, transform, print_each, params, mat_path):

    names = get_img_name(Path='Data/%s/' % (params.dataset), datasets=params.dataset + '_val')
    total = 0
    for iteration, val_data in enumerate(val_loader, 1):
        with torch.no_grad():
            # Load the data into the GPU if required
            val = val_data.type(torch.cuda.FloatTensor)
            val_ws = np.random.randint(0,5,1)[0]
            val_ws = WS[val_ws]
            val_down_spa = Spa_Downs(
                val.shape[1], factor, kernel_type='gauss12', kernel_width=val_ws[0],
                sigma=val_ws[1],preserve_size=True
            ).type(torch.cuda.FloatTensor)

            VAL_LR_HSI = val_down_spa(val)
            VAL_HR_MSI = torch.matmul(P,val.reshape(-1,val.shape[1],val.shape[2]*val.shape[3])).reshape(-1,P.size()[1],val.shape[2],val.shape[3])
            Input = transform([VAL_HR_MSI, VAL_LR_HSI])
            val_out = model(Input)
            val_out = torch.squeeze(val_out)

            sio.savemat(os.path.join(mat_path, names[iteration-1])+'.mat', {'hsi': np.array(val_out.detach().cpu()).transpose((1,2,0))})
            print('{0}th test image is saved.'.format(total))
            total += 1
    print('Test results generated and saved.')


#learning rate decay
def LR_Decay(optimizer, n, params, rate):
    lr_d = params.lr * (rate**n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_d
    print(lr_d)

def train(train_loader, model, start_epoch, stop_epoch, save_path, params, P, WS, transform):
    start_time = time.time()
    logfile = os.path.join(save_path, 'loss.txt')
    f = open(logfile, 'w+')
    f.write('The total loss, loss1, loss2 is : \n\n\n')

    L1 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #Training

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        print('*'*10, 'The {}th epoch for training.'.format(epoch+1), '*'*10)
        print_freq  = int((len(train_loader.dataset)/batch_size)/5)
        running_loss = 0  #the total loss

        for iteration, Data in enumerate(train_loader, 1):
            GT = Data.type(torch.cuda.FloatTensor)

            ws = np.random.randint(0,5,1)[0]
            ws = WS[ws]
            # ws = [8, 2]
            down_spa = Spa_Downs(
                GT.shape[1], factor, kernel_type='gauss12', kernel_width=ws[0],
                sigma=ws[1],preserve_size=True
            ).type(torch.cuda.FloatTensor)

            #Generate the LR_HSI
            LR_HSI = down_spa(GT)
            assert LR_HSI.size()[2] * factor == GT.size()[2], 'The LR_HSI size is incorrect.'
            #Generate the HR_MSI
            HR_MSI = torch.matmul(P,GT.reshape(-1,GT.shape[1],GT.shape[2]*GT.shape[3])).reshape(-1,P.size()[1], GT.shape[2],GT.shape[3])
            #Generate the UP_HSI
            UP_HSI = upsample(LR_HSI, (GT.shape[2],GT.shape[3]), mode='bicubic')
            #Generate the input data
            Input = transform([HR_MSI, LR_HSI])

            out = model(Input)
            loss = L1(out, GT)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss  += loss.data.cpu()
            if iteration% print_freq ==0:
                print('Epoch {:d} | Batch {}/{} | loss {:f}'.format(epoch, iteration, len(train_loader), running_loss/float(iteration)))

        print('*'*10, 'The average loss/batch is {:.4f}.'.format(running_loss/len(train_loader)), '*'*10)

        if epoch%10 == 0:
            LR_Decay(optimizer, epoch/10, params, 0.9)
            print('Adjusting the learning rate by timing 0.9.')

        if epoch%10 == 9:
            # torch.save(model, save_path+'model_'+str(int(epoch/10))+'.pth')
            torch.save(model.state_dict(), os.path.join(save_path, 'model_'+str(int(epoch/10))+'.pth'))

    T = time.time()-start_time

    print('Total training time is {}'.format(T))
    f.write('Total traing time is {}.\n'.format(T))

    f.close()

    return model




if __name__=='__main__':
    np.random.seed(10)
    params = parse_arg()

    lr = params.lr
    batch_size=params.batch_size
    factor = params.factor
    dataset = params.dataset
    patch_size = params.patch_size
    stride = params.stride
    net = params.model
    fusion = params.fusion
    pan = params.pan
    mem_load = params.mem_load
    noise = params.noise
    Train_image_num=dataset_dict[dataset][0]
    Valid_image_num = dataset_dict[dataset][1]
    snr = 30

    if pan:
        # Dim=[1, 32, 31]
        save_path = './Models/%s_X%s/PAN/%s_%s_%s_%s' %(dataset, str(factor), net, fusion, str(patch_size), str(stride))
        save_dir = './Models/%s_X%s/PAN/%s' %(dataset, str(factor), net)
        Dim = [1, 1 + dataset_dict[dataset][4], dataset_dict[dataset][4]]
        P = torch.ones(1,Dim[2])
        P = Variable(torch.unsqueeze(P/P.sum(), 0).type(torch.cuda.FloatTensor))
    else:
        # Dim=[3, 34, 31]
        save_path = './Models/%s_X%s/MSI/%s_%s_%s_%s' %(dataset, str(factor), net, fusion, str(patch_size), str(stride))
        save_dir = './Models/%s_X%s/MSI/%s' %(dataset, str(factor), net)

        Dim = [3, 3 + dataset_dict[dataset][4], dataset_dict[dataset][4]]
        P = Variable(torch.unsqueeze(torch.from_numpy(P['P']),0)).type(torch.cuda.FloatTensor)

    save_path += '_sd'
    WS = [[8,2],[8,2],[8,2],[8,2],[8,2]]


    if noise:
        add_noise = lambda x: add_wgn(x, snr=snr)
        save_path += '_noise%s' % (str(snr))
    else:
        add_noise = lambda x: tensor_copy(x)

    save_path = save_path + '_' + str(lr)

    if not os.path.isdir(save_path):
            os.makedirs(save_path)

    checkpoint_dir = os.path.join(save_path, 'model_best.pth')


    if net == 'MDANet':
        model = MDANet(Dim=Dim, nDenselayer=4, nFeat=64, growthRate=32).cuda()
        transform = transforms.Compose([transforms.Lambda(lambda data: [add_noise(i) for i in data]),
            transforms.Lambda(lambda data: [data[0], upsample(data[1], (data[0].shape[2],data[0].shape[3]), mode='bicubic')]),
            transforms.Lambda(lambda data: ins(data, torch.cat((data[1],data[0]),1), 1)),
            ])

    else:
        raise ValueError('Unknown model')



    if os.path.exists(checkpoint_dir):
        tmp = torch.load(checkpoint_dir)
        model.load_state_dict(tmp)
        print('Model loaded successfully!')

    Path = 'Data'
    Path = os.path.join(Path, params.dataset)

    if params.phase == 'train':
        #Load Train Dataset
        if mem_load:
            allData = all_data_in(Path=Path, datasets=params.dataset, Train_image_num=Train_image_num)
            dataset = LoadDataset_Mem(allData = allData, patch_size=patch_size, stride=stride)
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory = True)
        else:
            dataset = LoadDataset(Path=Path, datasets=params.dataset, patch_size=patch_size, stride=stride, up_mode='bicubic', Train_image_num=Train_image_num)
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory = True)

        model = train(data_loader, model, 0, dataset_dict[params.dataset][2], save_path, params, P, WS, transform)

    elif params.phase == 'test':
        #Load Train Dataset
        if mem_load:
            allValData = all_data_in(Path=Path, datasets=params.dataset+'_val', Train_image_num=Valid_image_num)
            print('data loaded to memory successfully!')
            val_dataset = LoadDataset_Mem_Val(allValData= allValData)
            val_data_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory = True)

        else:
            val_dataset = Val_LoadDataset(Path=Path, datasets=params.dataset+'_val', Train_image_num=Valid_image_num)
            val_data_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory = True)

        # mat_path = save_path.replace('Models', 'Results')
        mat_path = save_dir.replace('Models', 'Results')
        if not os.path.isdir(mat_path):
            os.makedirs(mat_path)
        test(val_data_loader, model, P, WS, transform, True, params, mat_path)

    else:
        raise ValueError ('Unknown phase of the network. Plz select train or test.')







