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

if __name__ == '__main__':
    opt = TestOptions().parse()
    assert (opt.no_dropout)

    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)

    model = create_model(opt)
    model.setup(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, 'sampling_%s' % (opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # sampling multiple times
    visuals, losses = model.validation(test_data_loader, how_many_to_valid=100, n_samples=opt.n_samples)

    sample_x = model.sampling(model.display_data[0], n_samples=opt.n_samples, max_display=16, return_all=True)
    sample_x = sample_x[:16] # display 4x4 grid gif
    visuals_gif_seq = []
    for i in range(sample_x.shape[1]):
        visuals_gif = util.tensor2im(tvutil.make_grid(sample_x[:,i,:,:,:], nrow=4))[:,:,0]
        visuals_gif_seq.append(Image.fromarray(visuals_gif))
    visuals['sample_gif'] = visuals_gif_seq

    sample_x = model.sampling(model.display_data[0], n_samples=opt.n_samples, max_display=16, return_all=True, sampling=False)
    sample_x = sample_x[:16] # display 4x4 grid gif
    visuals_gif_seq = []
    for i in range(sample_x.shape[1]):
        visuals_gif = util.tensor2im(tvutil.make_grid(sample_x[:,i,:,:,:], nrow=4))[:,:,0]
        visuals_gif_seq.append(Image.fromarray(visuals_gif))
    visuals['rec_gif'] = visuals_gif_seq

    import pdb ; pdb.set_trace()
    save_images(webpage, visuals, f'sampling ({opt.n_samples} samples)', aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
