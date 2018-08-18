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
    sample_loss, reconst_loss = [], []
    expt_r_loss, expt_s_loss = [], []
    vis_losses, invis_losses = [], []
    sample_ssim, reconst_ssim = [], []
    stage1_losses = []
    val_count = 0
    saved = False
    ngrid = 9 # how many image to show in each grid

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    
    # test
    total_test_n = len(test_data_loader)*opt.batchSize if opt.how_many < 0 else (opt.how_many)
    if not hasattr(model, 'sampling'): opt.n_samples = 1 # determnistic model needs no sampling
    
    if model.mri_data:
        tensor2im = functools.partial(util.tensor2im, renormalize=False)
    else:
        tensor2im = util.tensor2im

    def imagenet_loss(src, tar):
        if not model.mri_data:
            src.clamp_(-1,1)
            src = src.add(1).div_(2).add_(-0.43).div_(0.23)
            tar = tar.add(1).div_(2).add_(-0.43).div_(0.23)
        src = src[:,:1,:,:]
        tar = tar[:,:1,:,:]

        err = float(F.mse_loss(src, tar, size_average=True)) 
        return err 
    
    def compute_percentile(loss_array, none=None):
        # loss_array is in [N, n_samples]
        pct_10 = np.percentile(loss_array, 10, axis=1).mean()
        pct_90 = np.percentile(loss_array, 90, axis=1).mean()
        pct_50 = np.percentile(loss_array, 90, axis=1).mean()

        return pct_10, pct_90, pct_50
    
    ## mask is used to inspect the error from bg
    # background_mask = torch.zeros(1,1,128,128).cuda()
    # background_mask[:,:,32:96, 32:96] = 1

    for i, data in enumerate(test_data_loader):
        s_loss, r_loss = [], []
        s_sample, r_sample = [], []
        s_ssim, r_ssim = [], []
        for j in range(opt.n_samples):
            model.set_input(data)
            model.test()
            
            # tensity_mask = model.real_B > -0.1
            # mask = tensity_mask.float() * background_mask
            # # import pdb ; pdb.set_trace()
            # model.fake_B *= mask
            # model.real_B *= mask

            r_sample.append(model.fake_B.cpu())
            r_loss.append(imagenet_loss(model.fake_B.cpu(), model.real_B.cpu()))
            r_ssim.append(ssim_metric(model.fake_B.cpu(), model.real_B.cpu()))

            if hasattr(model, 'sampling'):
                model.test(sampling=True)
                s_sample.append(model.fake_B.cpu())
                s_loss.append(imagenet_loss(model.fake_B.cpu(), model.real_B.cpu()))
                s_ssim.append(ssim_metric(model.fake_B.cpu(), model.real_B.cpu()))
            else:
                s_loss = r_loss
                s_sample = r_sample
                s_ssim = r_ssim
            visuals = model.get_current_visuals()

            # compute fft vis and invis losses
            vis_loss, invis_loss = model.compute_special_losses()
            vis_losses.append(vis_loss)
            invis_losses.append(invis_loss)

        if hasattr(model, 'fake_B_G'):
            stage1_losses.append(imagenet_loss(model.fake_B_G.cpu(), model.real_B.cpu()))
        else:
            stage1_losses.append(imagenet_loss(model.fake_B.cpu(), model.real_B.cpu()))
                
        # compute expectation of samples then compute MSE
        c_expt_r_loss = [imagenet_loss(torch.stack(r_sample, 0).mean(dim=0), model.real_B.cpu())]
        c_expt_s_loss = [imagenet_loss(torch.stack(s_sample, 0).mean(dim=0), model.real_B.cpu())]

        
        reconst_loss.append(r_loss)
        sample_loss.append(s_loss)
        expt_r_loss.append(c_expt_r_loss)
        expt_s_loss.append(c_expt_s_loss)
        sample_ssim.append(s_ssim)
        reconst_ssim.append(r_ssim)

        
        sample_pct10, sample_pct90, _ = compute_percentile(sample_loss, s_sample)
        rec_pct10, rec_pct90, _ = compute_percentile(reconst_loss, r_sample)
        
        sys.stdout.write('\r processing %d / %d image MSE[q] = %.3f (%.3f, %.3f, %3f) MSE[p](%d) = %.3f (%.3f, %.3f, %.3f) MSE[vis/inv] = %.3f/%.3f S1 = %.3f SSIM[q]/[p] = %.3f/%.3f' % 
                            (val_count, total_test_n, 
                            mean_expt(expt_r_loss), rec_pct10, rec_pct90, mean_expt(reconst_loss),
                            opt.n_samples, 
                            mean_expt(expt_s_loss), sample_pct10, sample_pct90, mean_expt(sample_loss),
                            np.mean(vis_losses), np.mean(invis_losses),
                            np.mean(stage1_losses),
                            mean_expt(reconst_ssim), mean_expt(sample_ssim)))
        sys.stdout.flush()
        
        val_count += ngrid
        # Only save some. Don't explode the disk
        if val_count < 256:
            for k, v in visuals.items():
                v = v[:ngrid,:1,:,:]
                if model.mri_data:
                    v = util.mri_denormalize(v)
                visuals[k] = tensor2im(tvutil.make_grid(v, nrow=int(np.sqrt(ngrid))))
            
            save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        elif not saved:
            webpage.save()
            saved = True
            
        if val_count >= opt.how_many and opt.how_many > 0:
            break

    print('\n', '-'*100)
    print('Summary:')
    print('Percentile MSE: ')
    ret = OrderedDict()
    ret['p10_ctx'], ret['p90_ctx'], ret['median_ctx'] = compute_percentile(sample_loss)
    ret['p10'], ret['p90'] , ret['median'] = compute_percentile(reconst_loss)
    for k, v in ret.items(): print('  ' + k, '%.5f'% v)
