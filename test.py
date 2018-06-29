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

    opt.results_dir = opt.checkpoints_dir
    # _, val_data_loader = CreateFtTLoader(opt, valid_size=0.9)
    test_data_loader = CreateFtTLoader(opt, is_test=True)

    model = create_model(opt)
    model.setup(opt)
    model.eval() # eval mode can be used or not for instancenorm. The difference is small
    # model.netG.train()
    reconst_loss = []
    val_count = 0

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    total_test_n = len(test_data_loader)*opt.batchSize if opt.how_many < 0 else (opt.how_many)
    for i, data in enumerate(test_data_loader):
        
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        ## scale the value to the imagenet normalization
        # model.fake_B.add_(1).div_(2).add_(-0.43).div_(0.23)
        # model.real_B.add_(1).div_(2).add_(-0.43).div_(0.23)

        reconst_loss.append(float(F.mse_loss(model.fake_B, model.real_B, size_average=True)))
        val_count += model.fake_B.shape[0]
    
        # print('processing (%04d)-th / %d image... [mse loss = %.5f]' % (val_count, total_test_n, np.mean(reconst_loss)), flush=True)
        sys.stdout.write('\r processing %d / %d image [mse loss = %.5f] ...' % (val_count, total_test_n, np.mean(reconst_loss)))
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
