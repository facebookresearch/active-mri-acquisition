import os
import torch
from collections import OrderedDict
from . import networks
from util import util
import torchvision.utils as tvutil
import torch.nn.functional as F
import numpy as np

class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str) and hasattr(self, 'loss_' + name):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))                    
                net.load_state_dict(state_dict, strict=not self.opt.non_strict_state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def validation(self, val_data_loader, how_many_to_display=64, how_many_to_valid=4096*4):
        val_data = []
        val_count = 0
        need_sampling = hasattr(self, 'sampling') 
        losses = {
            'reconst_loss': [],
            'FFTVisiable_loss': [],
            'FFTInvisiable_loss': []
        }
        if need_sampling:
            losses['sampling_loss'] = []

        netG = getattr(self, 'netG')

        # visualization
        if not hasattr(self, 'display_data'):
            self.display_data = []
            for data in val_data_loader:
                self.display_data.append(data)
                val_count += data[0].shape[0]
                if val_count >= how_many_to_display: break
        
        for it, data in enumerate(self.display_data):
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            c = min(self.fake_B.shape[0], how_many_to_display)
            real_A, fake_B, real_B, = self.real_A[:c,...].cpu(), self.fake_B[:c,...].cpu(), self.real_B[:c,...].cpu() # save mem
            val_data.append([real_A[:c,:1,...], fake_B[:c,:1,...], real_B[:c,:1,...]])            
            
        visuals = {}
        visuals['inputs'] = util.tensor2im(tvutil.make_grid(torch.cat([a[0] for a in val_data], dim=0)))
        visuals['reconstructions'] = util.tensor2im(tvutil.make_grid(torch.cat([a[1] for a in val_data], dim=0)))
        visuals['groundtruths'] = util.tensor2im(tvutil.make_grid(torch.cat([a[2] for a in val_data], dim=0)))  

        if need_sampling:
            # we need to do sampling
            sample_x, pixel_diff_mean, pixel_diff_std = self.sampling(self.display_data[0])
            
            visuals['sample'] = util.tensor2im(tvutil.make_grid(sample_x, nrow=int(np.sqrt(sample_x.shape[0]))))
            nrow = int(np.ceil(np.sqrt(pixel_diff_mean.shape[0])))
            visuals['pixel_diff_mean'] = util.tensor2im(tvutil.make_grid(pixel_diff_mean, nrow=nrow, normalize=True, scale_each=True ))
            visuals['pixel_diff_std'] = util.tensor2im(tvutil.make_grid(pixel_diff_std, nrow=nrow, normalize=True, scale_each=True))

            losses['pixel_diff_std'] = np.mean(visuals['pixel_diff_std'])

        # evaluation
        val_count = 0
        for it, data in enumerate(val_data_loader):
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            # only evaluate the real part if has two channels
            losses['reconst_loss'].append(float(F.mse_loss(self.fake_B[:,:1,...], self.real_B[:,:1,...], size_average=True)))

            if need_sampling:
                self.test(sampling=True)
                losses['sampling_loss'].append(float(F.mse_loss(self.fake_B[:,:1,...], self.real_B[:,:1,...], size_average=True)))

            fft_vis, fft_inv = self.compute_special_losses()
            losses['FFTVisiable_loss'].append(fft_vis)
            losses['FFTInvisiable_loss'].append(fft_inv)

            val_count += self.fake_B.shape[0]
            if val_count >= how_many_to_valid: break
        
        for k, v in losses.items():
            losses[k] = np.mean(v)
            print('\t {} (avg): {} '.format(k, losses[k]))

        return visuals, losses