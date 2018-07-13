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

    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)

    model = create_model(opt)
    model.setup(opt)
    reconst_loss = []
    vis_losses, invis_losses = [], []
    val_count = 0

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    total_test_n = len(test_data_loader)*opt.batchSize if opt.how_many < 0 else (opt.how_many)
    if not hasattr(model, 'sampling'): opt.n_samples = 1 # determnistic model needs no sampling
    topk = {}
    for i in range(0, 11):
        topk[i] = []
    for i, data in enumerate(test_data_loader):
        
        sample_reconst_loss = []
        for j in range(opt.n_samples):
            
            model.set_input(data)
            if hasattr(model, 'sampling'):
                model.test(sampling=True)
            else:
                model.test()
            visuals = model.get_current_visuals()
        
            ## scale the value to the imagenet normalization for comparison
            fake_B = model.fake_B.add(1).div_(2).add_(-0.43).div_(0.23)
            real_B = model.real_B.add(1).div_(2).add_(-0.43).div_(0.23)

            # MSE only evaluate on the real part if output_nc==2
            sample_reconst_loss.append(float(F.mse_loss(fake_B[:,:1,:,:], real_B[:,:1,:,:], size_average=True)))
            val_count += model.fake_B.shape[0]

            # compute fft vis and invis losses
            vis_loss, invis_loss = model.compute_special_losses()
            vis_losses.append(vis_loss)
            invis_losses.append(invis_loss)

        sample_reconst_loss = np.sort(sample_reconst_loss)
        l = len(sample_reconst_loss)
        
        stdout_str = 'topk: '
        for i in range(0, 11):
            k = max(1, int(i/10*opt.n_samples)) if i > 0 else 1
            topk[i].append(np.mean(sample_reconst_loss[:k]))
            stdout_str += '{:.5f} '.format(np.mean(topk[i]))
        reconst_loss.append(topk[0])
        
        sys.stdout.write('\r '+stdout_str)
        # sys.stdout.write('\r processing %d / %d image [mse loss = %.5f] [vis loss = %.5f] [inv loss = %.5f] ...' % 
        #                     (val_count, total_test_n, np.mean(reconst_loss), np.mean(vis_losses), np.mean(invis_losses)))
        sys.stdout.flush()

        # only save some. Don't explode the disk
        if val_count < 256:
            for k, v in visuals.items():
                if v.shape[1] == 2:
                    v = v[:,:1,:,:]
                visuals[k] = util.tensor2im(tvutil.make_grid(v))
            
            save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
        if val_count >= opt.how_many and opt.how_many > 0:
            break

    b,c,h,w = model.real_B.shape
    print('Reconstruction loss: ', np.mean(reconst_loss))

    webpage.save()
                
# ret['median'] = np.percentile(ret['mse'].clone().cpu().data.numpy(), 50, axis=1).mean()
#     ret['median_ctx'] = np.percentile(ret['mse_ctx'].clone().cpu().data.numpy(), 50, axis=1).mean()
#     ret['p10'] = np.percentile(ret['mse'].clone().cpu().data.numpy(), 10, axis=1).mean()
#     ret['p10_ctx'] = np.percentile(ret['mse_ctx'].clone().cpu().data.numpy(), 10, axis=1).mean()
#     ret['p90'] = np.percentile(ret['mse'].clone().cpu().data.numpy(), 90, axis=1).mean()
#     ret['p90_ctx'] = np.percentile(ret['mse_ctx'].clone().cpu().data.numpy(), 90, axis=1).mean()