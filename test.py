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

if __name__ == '__main__':
    opt = TestOptions().parse()
    # opt.nThreads = 1   # test code only supports nThreads = 1
    # opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    # opt.no_flip = True  # no flip
    # opt.display_id = -1  # no visdom display
    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
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
    val_count = 0
    
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    
    # test
    total_test_n = len(test_data_loader)*opt.batchSize if opt.how_many < 0 else (opt.how_many)
    if not hasattr(model, 'sampling'): opt.n_samples = 1 # determnistic model needs no sampling

    def imagenet_loss(src, tar):

        src.clamp_(-1,1)
        src = src.add(1).div_(2).add_(-0.43).div_(0.23)
        tar = tar.add(1).div_(2).add_(-0.43).div_(0.23)
        b,c,h,w = src.shape
        err = float(F.mse_loss(src[:,:1,:,:], tar[:,:1,:,:], size_average=True)) 

        return err 
    def compute_percentile(loss_array, none=None):
        # loss_array is in [N, n_samples]
        pct_10 = np.percentile(loss_array, 10, axis=1).mean()
        pct_90 = np.percentile(loss_array, 90, axis=1).mean()
        pct_50 = np.percentile(loss_array, 90, axis=1).mean()

        return pct_10, pct_90, pct_5

    def compute_percentile2(loss_array, samples):
            # loss_array is in [N, n_samples]
        def get_expt_loss(ratio):
            pct = np.percentile(loss_array, ratio, axis=1)
            pct = np.tile(np.expand_dims(pct,1), (1,loss_array.shape[1]))
            ids = np.where(loss_array > pct)


        pct_10 = np.percentile(loss_array, 10, axis=1).mean()
        pct_90 = np.percentile(loss_array, 90, axis=1).mean()
        pct_50 = np.percentile(loss_array, 90, axis=1).mean()

        return pct_10, pct_90, pct_50    
        
    for i, data in enumerate(test_data_loader):
        s_loss, r_loss = [], []
        s_sample, r_sample = [], []

        for j in range(opt.n_samples):
            model.set_input(data)
            model.test()
            
            r_sample.append(model.fake_B.cpu())
            r_loss.append(imagenet_loss(model.fake_B, model.real_B))

            if hasattr(model, 'sampling'):
                model.test(sampling=True)
                s_sample.append(model.fake_B.cpu())
                s_loss.append(imagenet_loss(model.fake_B, model.real_B))
            else:
                s_loss = r_loss
                s_sample = r_sample
            visuals = model.get_current_visuals()

            # compute fft vis and invis losses
            vis_loss, invis_loss = model.compute_special_losses()
            vis_losses.append(vis_loss)
            invis_losses.append(invis_loss)

        # compute expectation of samples than compute MSE
        c_expt_r_loss = [imagenet_loss(torch.stack(r_sample, 0).mean(dim=0), model.real_B.cpu())]
        c_expt_s_loss = [imagenet_loss(torch.stack(s_sample, 0).mean(dim=0), model.real_B.cpu())]

        val_count += model.fake_B.shape[0]
        reconst_loss.append(r_loss)
        sample_loss.append(s_loss)
        expt_r_loss.append(c_expt_r_loss)
        expt_s_loss.append(c_expt_s_loss)
        
        sample_pct10, sample_pct90, _ = compute_percentile(sample_loss, s_sample)
        rec_pct10, rec_pct90, _ = compute_percentile(reconst_loss, r_sample)
        
        sys.stdout.write('\r processing %d / %d image E[posterior] = %.5f (%.5f, %.5f, %5f) E[prior](%d) = %.5f (%.5f, %.5f, %5f) vis/inv mse = %.5f/%.5f ..' % 
                            (val_count, total_test_n, 
                            mean_expt(expt_r_loss), rec_pct10, rec_pct90, mean_expt(reconst_loss),
                            opt.n_samples, 
                            mean_expt(expt_s_loss), sample_pct10, sample_pct90, mean_expt(sample_loss),
                            np.mean(vis_losses), np.mean(invis_losses)))
        sys.stdout.flush()
        
        # Only save some. Don't explode the disk
        if val_count < 256:
            for k, v in visuals.items():
                if v.shape[1] == 2:
                    v = v[:,:1,:,:]
                visuals[k] = util.tensor2im(tvutil.make_grid(v))
            
            save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        else:
            webpage.save()
        if val_count >= opt.how_many and opt.how_many > 0:
            break

    print('\n', '-'*100)
    print('Summary:')
    print('Percentile MSE: ')
    ret = OrderedDict()
    ret['p10_ctx'], ret['p90_ctx'], ret['median_ctx'] = compute_percentile(sample_loss)
    ret['p10'], ret['p90'] , ret['median'] = compute_percentile(reconst_loss)
    for k, v in ret.items(): print('  ' + k, '%.5f'% v)
