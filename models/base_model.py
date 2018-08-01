import os, sys
import torch
from collections import OrderedDict
from . import networks
import torchvision.utils as tvutil
import torch.nn.functional as F
import numpy as np
from util import util
import functools
from .fft_utils import create_mask

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

        self.validation_phase = False
        self.mri_data = self.opt.dataroot in ('KNEE')

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

        # if self.opt.KL_w_decay:

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name).cpu()
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
        # save optimizer         
        optimizers = getattr(self, 'optimizers')
        optim_dict = {}
        save_filename = '%s_optim.pth' % (which_epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        for i, optim in enumerate(optimizers):
            optim_dict[i] = optim.state_dict()
        torch.save(optim_dict, save_path)

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

        if self.isTrain and hasattr(self, 'optimizers'):
            # load optimizer        
            optimizers = getattr(self, 'optimizers')
            load_filename = '%s_optim.pth' % (which_epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            if os.path.isfile(load_path):
                state_dict = torch.load(load_path, map_location=str(self.device))
                for i, optim in enumerate(optimizers):
                    optim.load_state_dict(state_dict[i])
                print('loading the optimizer from %s' % load_path)
            else:
                print('can not found optimizer checkpoint at', load_path)

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


    def validation(self, val_data_loader, how_many_to_display=64, how_many_to_valid=4096*4, n_samples=8):
        
        if self.mri_data:
            tensor2im = functools.partial(util.tensor2im, renormalize=False)
        else:
            tensor2im = util.tensor2im

        val_data = []
        val_count = 0
        need_sampling = hasattr(self, 'sampling') 
        losses = {
            'reconst_loss': [],
            'reconst_ssim': [],
            'FFTVisiable_loss': [],
            'FFTInvisiable_loss': []
        }
        if need_sampling:
            losses['sampling_loss'] = []
            losses['sampling_ssim'] = []

        netG = getattr(self, 'netG')
        
        # turn on the validation sign
        self.validation_phase = True

        # visualization for a fixed number of data
        if not hasattr(self, 'display_data'):
            self.display_data = []
            for data in val_data_loader:
                self.display_data.append(data)
                val_count += data[0].shape[0]
                if val_count >= how_many_to_display: break
                   
        # prepare display data
        for it, data in enumerate(self.display_data):
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            c = min(self.fake_B.shape[0], how_many_to_display)
            real_A, fake_B, real_B, = self.real_A[:c,...].cpu(), self.fake_B[:c,...].cpu(), self.real_B[:c,...].cpu() # save mem
            val_data.append([real_A[:c,:1,...], fake_B[:c,:1,...], real_B[:c,:1,...]])            
        
        if self.mri_data:
            for i in range(3):
                for a in val_data:
                    util.mri_denormalize(a[i])
        visuals = {}
        visuals['inputs'] = tensor2im(tvutil.make_grid(torch.cat([a[0] for a in val_data], dim=0)[:how_many_to_display] ))
        visuals['reconstructions'] = tensor2im(tvutil.make_grid(torch.cat([a[1] for a in val_data], dim=0)[:how_many_to_display]))
        visuals['groundtruths'] = tensor2im(tvutil.make_grid(torch.cat([a[2] for a in val_data], dim=0)[:how_many_to_display]))  

        if need_sampling:
            # we need to do sampling
            sample_x, pixel_diff_mean, pixel_diff_std = self.sampling(self.display_data[0], n_samples=n_samples)
            
            if self.mri_data: util.mri_denormalize(sample_x)

            visuals['sample'] = tensor2im(tvutil.make_grid(sample_x, nrow=int(np.sqrt(sample_x.shape[0]))))
            nrow = int(np.ceil(np.sqrt(pixel_diff_mean.shape[0])))
            visuals['pixel_diff_mean'] = tensor2im(tvutil.make_grid(pixel_diff_mean, nrow=nrow, normalize=True, scale_each=True), renormalize=False)
            visuals['pixel_diff_std'] = tensor2im(tvutil.make_grid(pixel_diff_std, nrow=nrow, normalize=True, scale_each=True), renormalize=False)
            
            losses['pixel_diff_std'] = np.array(torch.mean(pixel_diff_mean).item())
            losses['pixel_diff_mean'] = np.array(torch.mean(pixel_diff_std).item())

        # evaluation
        val_count = 0
        for it, data in enumerate(val_data_loader):
            ## posterior
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            # only evaluate the real part if has two channels
            losses['reconst_loss'].append(float(F.mse_loss(self.fake_B[:,:1,...], self.real_B[:,:1,...], size_average=True)))
            losses['reconst_ssim'].append(float(util.ssim_metric(self.fake_B[:,:1,...], self.real_B[:,:1,...])))

            if need_sampling:
                # compute at posterior
                bits_per_dim = self.compute_logistic()
                losses['bits_per_dim'] = bits_per_dim
                losses['div'] = self.compute_KL()
            # prior
            if need_sampling:
                self.set_input(data)
                self.test(sampling=True)
                losses['sampling_loss'].append(float(F.mse_loss(self.fake_B[:,:1,...], self.real_B[:,:1,...], size_average=True)))
                losses['sampling_ssim'].append(float(util.ssim_metric(self.fake_B[:,:1,...], self.real_B[:,:1,...])))
            else:
                losses['sampling_loss'] = losses['reconst_loss']

            # compute at prior
            fft_vis, fft_inv = self.compute_special_losses()
            losses['FFTVisiable_loss'].append(fft_vis)
            losses['FFTInvisiable_loss'].append(fft_inv)
            
            sys.stdout.write('\r validation [rec loss: %.5f smp loss: %.5f]' % (np.mean(losses['reconst_loss']), np.mean(losses['sampling_loss'])))
            sys.stdout.flush()

            val_count += self.fake_B.shape[0]
            if val_count >= how_many_to_valid: break

        print(' ')
        for k, v in losses.items():
            if k in ('sampling_loss', 'reconst_loss'):
                nan_n = len([a for a in v if str(a) == 'nan'])
                if nan_n < (0.1 * len(v)):
                    v = [a for a in v if str(a) != 'nan']
            losses[k] = np.mean(v)
            print('\t {} (E[error]): {} '.format(k, losses[k]))

        self.validation_phase = False

        return visuals, losses
    
    def gen_random_mask(self, batchSize=1):
        if self.isTrain and self.opt.dynamic_mask_type != 'None' and not self.validation_phase:
            if self.opt.dynamic_mask_type == 'random':
                mask = create_mask((batchSize, self.opt.fineSize), random_frac=True, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
            elif self.opt.dynamic_mask_type == 'random_lines':
                seed = np.random.randint(10000)
                mask = create_mask((batchSize, self.opt.fineSize), random_frac=False, mask_fraction=self.opt.kspace_keep_ratio, seed=seed).to(self.device)
        else:
            mask = create_mask(self.opt.fineSize, random_frac=False, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
            
        return mask
    
    def compute_special_losses(self):
        # compute losses between fourier spaces of fake_B and real_B
        # if output one dimension
        # import pdb ; pdb.set_trace()
        if self.fake_B.shape[1] == 1:
            _k_fakeB = self.RFFT(self.fake_B)
            _k_realB = self.RFFT(self.real_B)
        else:
            # if output are two dimensional
            _k_fakeB = self.FFT(self.fake_B)
            _k_realB = self.FFT(self.real_B)

        b = self.fake_B.shape[0] if self.mask.shape[0] == 1 else 1
    
        mask_deno = self.mask.sum() * b * self.fake_B.shape[1] * self.fake_B.shape[3]
        invmask_deno = (1-self.mask).sum() * b * self.fake_B.shape[1] * self.fake_B.shape[3]

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB * self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB * (1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)
