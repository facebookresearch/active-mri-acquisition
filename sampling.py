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
from PIL import Image
import functools
from util.draw_figure import draw_histogram
from models.fft_utils import create_mask

if __name__ == '__main__':
    opt = TestOptions().parse()
    assert (opt.no_dropout)
    
    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)
    test_data_loader = iter(test_data_loader) # to prevent each loop-restart every time
    model = create_model(opt)
    model.setup(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, 'sampling_%s' % (opt.which_epoch))
    # web_dir = os.path.join(opt.results_dir, opt.name, 'kspace_noises_sampling_%s' % (opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    if model.mri_data:
        tensor2im = functools.partial(util.tensor2im, renormalize=False)
    else:
        tensor2im = util.tensor2im

    for it in range(10):
        visuals = {}
        model.display_data = None
        # do validation and compute metrics
        metasavepath = os.path.join(webpage.get_image_dir(), f'validation_metadata{it}.pickle')
        visuals, losses = model.validation(test_data_loader, how_many_to_display=36, how_many_to_valid=10, 
                                            n_samples=opt.n_samples, metasavepath=metasavepath)    
        if hasattr(model, 'sampling'):
            # sampling from prior multiple times
            sample_x = model.sampling(model.display_data[0], n_samples=opt.n_samples, max_display=16, return_all=True)
            sample_x = sample_x[:16] # display 4x4 grid gif
            if model.mri_data:
                for i, x in enumerate(sample_x):
                    sample_x[i] = util.mri_denormalize(x, zscore=5)
            visuals_gif_seq = []
            for i in range(sample_x.shape[1]):
                visuals_gif = tensor2im(tvutil.make_grid(sample_x[:,i,:,:,:], nrow=4))[:,:,0]
                visuals_gif_seq.append(Image.fromarray(visuals_gif))
            visuals['sample_gif'] = visuals_gif_seq
            
            # # sample from posterior multiple times
            # sample_x = model.sampling(model.display_data[0], n_samples=opt.n_samples, max_display=16, return_all=True, sampling=False)
            # sample_x = sample_x[:16] # display 4x4 grid gif
            # if model.mri_data:
            #     for i, x in enumerate(sample_x):
            #         sample_x[i] = util.mri_denormalize(x, zscore=5)
            # visuals_gif_seq = []
            # for i in range(sample_x.shape[1]):
            #     visuals_gif = tensor2im(tvutil.make_grid(sample_x[:,i,:,:,:], nrow=4))[:,:,0]
            #     visuals_gif_seq.append(Image.fromarray(visuals_gif))
            # visuals['rec_gif'] = visuals_gif_seq

        ''' show histogram of uncertainty '''
        uncertainty = []
        for j, data in enumerate(test_data_loader):
            if j > 1: break
            model.set_input(data)
            model.test()
            uncertainty.append(np.stack([logvar.exp().cpu().numpy() for logvar in model.logvars],0))

        uncertainty = np.concatenate(uncertainty, 1) #[3,B*J,1,H,W]
        visuals['uncertainty'] = draw_histogram(uncertainty.reshape(uncertainty.shape[0],-1), os.path.join(webpage.get_image_dir(), f'uncertainty_{it}.pdf'))
        visuals['validation_metadata'] = os.path.basename(metasavepath)


        save_images(webpage, visuals, f'sampling ({opt.n_samples} samples) iter {it}', aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        sys.stdout.write(f'\r --> iter {it}')
        sys.stdout.flush()
        webpage.save()


