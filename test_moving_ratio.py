import os, sys
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from data import CreateFtTLoader
from util import util
import torchvision.utils as tvutil
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import functools
from util.util import ssim_metric
from models.fft_utils import create_mask
from util import draw_figure 
import pickle
from torch.nn import functional as F
import pickle


''' Evaluate 
MSE,
SSIM,
Correlation between Uncertainty and MSE
'''
if __name__ == '__main__':
    opt = TestOptions().parse()
    assert (opt.no_dropout)

    # compute E[x] and then |E[x] - y|
    compute_sample_expt_first = True
    # otherwise, compute x - y and then E[x - y]
    # x is a 2D array in [N, n_sample]
    mean_expt = lambda x: np.mean(np.mean(x,1))

    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)

    model = create_model(opt)
    model.setup(opt)
    import pdb; pdb.set_trace()
    reconst_ssim, reconst_loss, error_uncertainty_list, reconst_loss_sample = [], [], [], []
    visuals = {}

    conjudge_symmetric = True
    opt.how_many = 512
    # # create website
    web_dir = os.path.join(opt.results_dir, opt.name, 'moving_ratio_experiment_%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    filename = 'MSE_SSIM_moving_ratio.pickle'

    def imagenet_loss(src, tar, average_sample=True):
        if not model.mri_data:
            src.clamp_(-1,1)
            src = src.add(1).div_(2).add_(-0.43).div_(0.23)
            tar = tar.add(1).div_(2).add_(-0.43).div_(0.23)
        src = src[:,:1,:,:]
        tar = tar[:,:1,:,:]

        if average_sample:
            err = float(F.mse_loss(src, tar, size_average=True))
        else:
            err = F.mse_loss(src, tar, reduce=False)
            b,c,h,w = err.shape
            err = err.sum(1).sum(1).sum(1) / (c*h*w)
            err = err.numpy()
        return err 
    
    # ratios = np.arange(0.02, 0.25, 0.02) # [0.02,0.05,0.1,0.15,0.2,0.25]
    ratios = np.arange(0.02, 0.8, 0.01) # only for draw box plot MSEVAR
    masks = []
    kMA = []
    for ratio in ratios:
        cur_mask = create_mask((opt.batchSize, opt.fineSize), random_frac=False, mask_fraction=ratio).to(model.device)
        if conjudge_symmetric:
            for jj in range(len(cur_mask)):
                for j in range(1,127):
                    if cur_mask[jj, 0, j, 0] == 1:
                        cur_mask[jj, 0, 128-j, 0] = 1
        masks.append(cur_mask)
        kMA.append(masks[-1][0,0,:,0].sum().item())

    kMA = [a/opt.fineSize * 100 for a in kMA]
    sum_over = lambda x: x.sum(1).sum(1).sum(1)
    print('Generate mask kmA, ', kMA)
    val_count = 0
    use_uncertainty = 'energypasnetplus' in opt.name

    for i, data in enumerate(test_data_loader):
        r_loss, r_ssim, err_unc_list, r_loss_sample = [], [], [], []

        for j in range(len(masks)):
            ''' show mse vs different mask'''
            model.set_input_exp(data[1:], masks[j])
            model.test()
            model.fake_B = model.fake_B.cpu()
            model.real_B = model.real_B.cpu()
            
            # r_loss_sample.append(imagenet_loss(model.fake_B.cpu(), model.real_B.cpu(), average_sample=False) )
            r_loss_sample.append(torch.stack([model.fake_B.cpu(), model.real_B.cpu()], 4).numpy())
            r_loss.append(imagenet_loss(model.fake_B.cpu(), model.real_B.cpu()))
            r_ssim.append(ssim_metric(model.fake_B.cpu(), model.real_B.cpu()))
            
            if use_uncertainty:
                mse = F.mse_loss(model.fake_B[:,:1,:,:], model.real_B[:,:1,:,:], reduce=False)
                mse = sum_over(mse)
                uncertainties = [sum_over(logvar.cpu().exp()) for logvar in model.logvars[2:]]
                err_unc_list.append(torch.stack([mse] + uncertainties, 0))

            sys.stdout.write(f'\r iter {i} ratio {kMA[j]}')
            sys.stdout.flush()
        error_uncertainty_list.append(torch.stack(err_unc_list, 0))
        reconst_loss.append(r_loss)
        reconst_ssim.append(r_ssim)
        # reconst_loss_sample.append(np.stack(r_loss_sample, 1))
        reconst_loss_sample.append(np.stack(r_loss_sample, 0))
        val_count += model.fake_B.shape[0]
        if val_count >= opt.how_many and opt.how_many > 0:
            break


    saveroot = webpage.get_image_dir()
    reconst_loss = np.array(reconst_loss).mean(0)
    reconst_ssim = np.array(reconst_ssim).mean(0)
    reconst_loss_sample = np.concatenate(reconst_loss_sample, 1)

    if use_uncertainty:
        error_uncertainty_list = torch.stack(error_uncertainty_list, 0)
        B,_,H,W = model.fake_B.shape
        Iter, Nmask, N, Batch = error_uncertainty_list.shape
        # import pdb; pdb.set_trace()
        error_uncertainty_list = error_uncertainty_list.permute(0,3,1,2).contiguous().view(Iter*Batch,Nmask,N)
        # error_uncertainty_list = error_uncertainty_list.sum(0) #[ len(masks), 4]
        error_uncertainty_list.div_(H*W) # see sum_over, it needs normalize here
        visuals['mse_uncertainty_cor'] = draw_figure.compute_mse_uncertainty_cor(error_uncertainty_list,
                                     pdfsavepath=os.path.join(saveroot,'mse_uncertainty_cor.pdf'))

    
    figmse = draw_figure.draw_curve(kMA, reconst_loss, ylabel='MSE', pdfsavepath=os.path.join(saveroot,'mse.pdf'))
    figssim = draw_figure.draw_curve(kMA, reconst_ssim, ylabel='SSIM', pdfsavepath=os.path.join(saveroot,'ssim.pdf'))
    
    # draw mse variance
    mse_sample = []
    for i in range(reconst_loss_sample.shape[0]):
        # looping over batch
        data = reconst_loss_sample[i]
        src, tar = data[...,0], data[...,1]
        mse_sample.append(imagenet_loss(torch.from_numpy(src), torch.from_numpy(tar), average_sample=False))

    mse_sample = np.stack(mse_sample, 1)
    reconst_loss_sample = reconst_loss_sample[:,:,0,:,:,0] 
    metadat = {'mse': mse_sample, 'samples': reconst_loss_sample, 'kMA': kMA}
    pickle.dump(metadat, open('experiments/kspace_line_var/metadata512.pickle','wb'), protocol=4)
    figmsevar = draw_figure.draw_curve_msevar(kMA, mse_sample, ylabel='MSE', pdfsavepath=os.path.join(saveroot,'msevar.pdf'))

    visuals['SSIM'] = figssim
    visuals['MSE'] = figmse
    visuals['MSEVAR'] = figmsevar

    metadata = {}
    metadata['kMA'] = kMA
    metadata['MSE'] = reconst_loss
    metadata['SSIM'] = reconst_ssim
    metadata['mse_uncertainty_cor'] = error_uncertainty_list

    savepath = os.path.join(webpage.get_image_dir(), filename)
    pickle.dump(metadata, open(savepath,'wb'))

    visuals['metadata'] = filename
    save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
    