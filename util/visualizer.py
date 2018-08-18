import numpy as np
import os
import ntpath
import time
import glob
from . import util
from . import html
from scipy.misc import imresize
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

def gray2heatmap(grayimg, cmap='jet'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(grayimg)
    rgb_img = np.delete(rgba_img, 3, 2) * 255.0
    rgb_img = rgb_img.astype(np.uint8)
    return rgb_img

def save_images(webpage, visuals, name, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    # short_path = ntpath.basename(image_path[0])
    # name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        
        if type(im_data) is not list:
            image_name = '%s_%s.png' % (name, label)
            im = util.tensor2im(im_data)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)
        else:
            image_name = '%s_%s.gif' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            im_data[0].save(fp=save_path, format='gif', save_all=True, append_images=im_data[1:])

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

class Visualizer():
    def __init__(self, opt, use_html=False):
        self.use_html = use_html
        self.win_size = 256
        self.no_tb = opt.debug
        # self.name = name
        # self.opt = opt
        self.saved = False
        self.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.checkpoints_dir, 'logfile.txt')
        self.name = opt.name
        if self.use_html:
            self.web_dir = os.path.join(self.checkpoints_dir, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdir(self.web_dir)
            util.mkdir(self.img_dir)

        self.ncols = 4
        if not self.no_tb:
            # remove existing 
            if not opt.continue_train:
                for filename in glob.glob(self.checkpoints_dir+"/events*"):
                    os.remove(filename)
            self.writer = SummaryWriter(self.checkpoints_dir)
        else:
            print('[Visualizer] -> do not create visualizer in debug mode')
    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, mode='train'):
        if self.no_tb: return
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image) 
            self.writer.add_image('{}/{}'.format(mode, label), image_numpy, epoch)

        if self.use_html and (not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()
        

    # losses: dictionary of error labels and values
    def plot_current_losses(self, mode, n_iter, **args):
        if self.no_tb: return
        for k, v in args.items():
            self.writer.add_scalar('{}/{}'.format(mode, k), v, n_iter)

        self.writer.export_scalars_to_json("{}/tensorboard_all_scalars.json".format(self.checkpoints_dir))

    def display_current_histograms(self, mode, n_iter, **args):
        if self.no_tb: return
        for k, v in args.items():
            self.writer.add_histogram('{}/{}'.format(mode, k), v, n_iter)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, iter_epoch, t, losses, t_data):
        message = '(epoch: %d, iters: %d/%d, time: %.3f, data: %.3f) ' % (epoch, i, iter_epoch, t, t_data)
        for k, v in losses.items():
            # message += '%s: %.6f ' % (k, v)
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

