import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *
import math
from model import Net
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=2, help="angular resolution")
    parser.add_argument("--angout", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=2, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='SASRNet')
    parser.add_argument('--trainset_dir', type=str, default='./LFData/TrainingData_2x2_sx2SR_5x5')
    parser.add_argument('--testset_dir', type=str, default='./LFData/TestData_all_2x2_sx2SR_5x5/')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument("--stride", type=int, default=32, help="stride for patch cropping")

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/SASRNet_2x2xSR_5x5_epoch_1.pth.tar')

    return parser.parse_args()

if not os.path.exists('./log'):
    os.mkdir('./log')

def train(cfg, train_loader, test_Names, test_loaders):

    net = Net(cfg.angin, cfg.angout, cfg.upscale_factor)
    net.to(cfg.device)
    cudnn.benchmark = False
    epoch_state = 0
    ##### get input index ######         
    ind_source = torch.tensor([cfg.angout*cfg.angout//2])

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimazer'])
            epoch_state = model["epoch"]
            print("load pre-train at epoch {}".format(epoch_state))
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    #net = torch.nn.DataParallel(net, device_ids=[0, 1])
    
    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out  = net(data)
            loss = criterion_Loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open('../checkpoint/' + cfg.model_name + '_training.txt', 'a')
            txtfile.write(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())) + '\n')
            txtfile.close()
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                #'state_dict': net.module.state_dict(),  # for torch.nn.DataParallel
                'optimizer': optimizer.state_dict(),
                'loss': loss_list,},
                save_path='../checkpoint/', filename=cfg.model_name + '_' + str(cfg.angin) + 'x' + str(cfg.angin)+ 'xSR_' + str(cfg.angout) +
                            'x' + str(cfg.angout) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net, ind_source)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile = open('../checkpoint/' + cfg.model_name + '_training.txt', 'a')
                txtfile.write('Dataset----%10s,\t PSNR---%f,\t SSIM---%f\n' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile.close()
                pass
            pass

        scheduler.step()
        pass


def valid(test_loader, net, ind_source):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape

        minibatch = 4
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, subLFin.shape[2], subLFin.shape[3])
        
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
        #torch.cuda.empty_cache()
        out_lf = torch.cat(out_lf, 0)
        subLFout = out_lf.view(numU, numV, cfg.angout * cfg.patchsize*cfg.upscale_factor, cfg.angout * cfg.patchsize*cfg.upscale_factor)
        outLF = LFintegrate(subLFout, cfg.angout, cfg.patchsize*cfg.upscale_factor, cfg.stride*cfg.upscale_factor, h0*cfg.upscale_factor, w0*cfg.upscale_factor)
        outLF = outLF.clip(0, 1)

        psnr, ssim = cal_metrics(label, outLF, cfg.angout)

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def save_ckpt(state, save_path='../checkpoint', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename), _use_new_zipfile_serialization=False)


def main(cfg):
    setup_seed(10)
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=32, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
if __name__ == '__main__':
    cfg = parse_args()    
    main(cfg)
