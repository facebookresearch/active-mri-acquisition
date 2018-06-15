import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from data import CreateFtTLoader
from util import util
import torchvision.utils as tvutil
import torch

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
    _, val_data_loader = CreateFtTLoader(opt, valid_size=0.9)

    model = create_model(opt)
    model.setup(opt)
    model.eval()
    reconst_loss = 0
    val_count = 0

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(val_data_loader):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        reconst_loss += float(torch.pow((model.fake_B - model.real_B), 2).sum())
        val_count += model.fake_B.shape[0]
        for k, v in visuals.items():
            visuals[k] = util.tensor2im(tvutil.make_grid(v))
        if i % 5 == 0:
            print('processing (%04d)-th image...' % (i))
        save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    b,c,h,w = model.real_B.shape
    reconst_loss /= (val_count*h*w*c)
    print('Reconstruction loss: ', reconst_loss)

    webpage.save()
