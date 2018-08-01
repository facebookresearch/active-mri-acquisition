import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
import torchvision.utils as tvutil
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .fft_utils import *
from torch.autograd import Variable
import sys

class FTGAUSCNNModel(BaseModel):
    def name(self):
        return 'FTGAUSCNNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, 
                                    self.gpu_ids, no_last_tanh=True)
        self.RFFT = RFFT().to(self.device)
        self.mask = create_mask(opt.fineSize).to(self.device)
        self.IFFT = IFFT().to(self.device)
        self.IRFFT = IRFFT().to(self.device)
        # for evaluation
        self.FFT = FFT().to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.MSELoss(reduce=False)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

        if int(np.ceil(self.opt.output_nc/2)) == 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

    def set_input(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)

        if self.isTrain and self.opt.dynamic_mask_type != 'None' and not self.validation_phase:
            if self.opt.dynamic_mask_type == 'random':
                    self.mask = create_mask(self.opt.fineSize, random_frac=True, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
            elif self.opt.dynamic_mask_type == 'random_lines':
                seed = np.random.randint(100)
                self.mask = create_mask(self.opt.fineSize, random_frac=False, mask_fraction=self.opt.kspace_keep_ratio, seed=seed).to(self.device)
        else:
            self.mask = create_mask(self.opt.fineSize, random_frac=False, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
        # doing FFT
        # if has two dimension output, 
        # we actually want toe imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img)
        if int(np.ceil(self.opt.output_nc/2)) == 2:
            if self.imag_gt.shape[0] != img.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(img)
            img = torch.cat([img, self.imag_gt], dim=1)

        if AtoB:
            self.real_A = self.IFFT(fft_kspace * self.mask)
            self.real_B = img
        else:
            self.real_A = self.IFFT(fft_kspace)
            self.real_B = img
    
    def compute_special_losses(self):
        # compute losses between fourier spaces of fake_B and real_B
        # if output one dimension
        if self.fake_B.shape[1] == 1:
            _k_fakeB = self.RFFT(self.fake_B)
            _k_realB = self.RFFT(self.real_B)
        else:
            # if output are two dimensional
            _k_fakeB = self.FFT(self.fake_B)
            _k_realB = self.FFT(self.real_B)

        mask_deno = self.mask.sum() * self.fake_B.shape[0] * self.fake_B.shape[1] * self.fake_B.shape[3]
        invmask_deno = (1-self.mask).sum() * self.fake_B.shape[0] * self.fake_B.shape[1] * self.fake_B.shape[3]

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB * self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB * (1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)

    def test(self):
        with torch.no_grad():
            self.forward()

    def forward(self, sampling=False):
        h, b = self.mask.shape[2], self.real_A.shape[0]
        output  = self.netG(self.real_A)
        o = int(np.ceil(self.opt.output_nc/2))
        img, self.logvar = output[:,:o,:,:], output[:,o:,:,:]
        mask = Variable(self.mask.view(self.mask.shape[0],1,h,1))#.expand(b,h,1,1))
        if img.shape[1] == 1:
            ft_x = self.RFFT(img)
            ft_in = self.FFT(self.real_A)
            self.fake_B = self.IRFFT((1-mask) * ft_x + ft_in) 
        else:
            ft_x = self.FFT(img)
            self.fake_B = self.IFFT((1-mask) * ft_x) + self.real_A
        
        # if sampling:
            
        #     self.fake_B = self.reparam(self.fake_B[:,:1,:,:], self.logvar)

    def backward_G(self):
        o = int(np.floor(self.opt.output_nc/2))
        l2 = self.criterion(self.fake_B[:,:o,:,:], self.real_B[:,:o,:,:]) 
        one_over_var = torch.exp(-self.logvar)
        # gaussian loss
        self.loss_G = (0.5 * (one_over_var * l2 + self.logvar)).mean()

        self.loss_G.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def reparam(self, mu, logvar):

        eps = torch.zeros_like(mu).normal_()
        std = torch.exp(logvar*0.5)
        spl = eps.mul(std).add_(mu)

        return spl

    def sampling(self, data, n_samples=8, max_display=8, return_all=False, sampling=True):
        def replicate_tensor(data, times, expand=False):
            ks = list(data.shape)
            data = data.view(ks[0], 1, ks[1], ks[2], ks[3])
            data = data.repeat(1, times, 1, 1, 1) # repeat will copy memories which we do not need here
            if not expand:
                data = data.view(ks[0]*times, ks[1], ks[2], ks[3])
            return data
        
        data[0] = data[0][:max_display]
        self.set_input(data)
        self.test() 

        data = data[0]
        b,c,h,w = data.shape
        sample_x = []
        for i in range(n_samples):
            spl = self.reparam(self.fake_B[:,:1,:,:], self.logvar) 
            sample_x.append(spl.cpu())
        
        sample_x = torch.stack(sample_x, 1)
        input_imgs = replicate_tensor(data, n_samples).cpu()

        # compute sample mean std
        all_pixel_diff = (input_imgs.view(b,n_samples,c,h,w) - sample_x.view(b,n_samples,c,h,w)).abs()
        pixel_diff_mean = torch.mean(all_pixel_diff, dim=1) # n_samples
        pixel_diff_std = torch.std(all_pixel_diff, dim=1) # n_samples

        if return_all:# return all samples
            return sample_x.view(b,n_samples,c,h,w)
        else:    
            # return 
            mean_samples_x = sample_x.view(b,n_samples,c,h,w).mean(dim=1)
            sample_x = sample_x.view(b,n_samples,c,h,w)[:,:max_display,:,:,:].contiguous()
            sample_x[:,0,:,:] = data[:,:1,:,:] # GT
            # sample_x[:,-1,:,:] = pixel_diff_mean.div(pixel_diff_mean.max()).mul_(2).add_(-1) # MEAN
            sample_x[:,-1,:,:] = pixel_diff_std.div(pixel_diff_std.max()).mul_(2).add_(-1) # STD
            sample_x[:,1,:,:] = mean_samples_x # E[x]

            sample_x = sample_x.view(b*max_display,c,h,w)
        
        return sample_x, pixel_diff_mean, pixel_diff_std

        
    def validation(self, val_data_loader, how_many_to_display=64, how_many_to_valid=4096*4, n_samples=8):
        
        if self.mri_data:
            tensor2im = functools.partial(util.tensor2im, renormalize=False)
        else:
            tensor2im = util.tensor2im

        val_data = []
        val_count = 0
        need_sampling = True
        losses = {
            'reconst_loss': [],
            'FFTVisiable_loss': [],
            'FFTInvisiable_loss': []
        }
        if need_sampling:
            losses['sampling_loss'] = []
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
            var = torch.exp(self.logvar[:c,...]).cpu()
            val_data.append([real_A[:c,:1,...], fake_B[:c,:1,...], real_B[:c,:1,...], var[:c,:1,...]])            
        
        if self.mri_data:
            for i in range(3):
                for a in val_data:
                    util.mri_denormalize(a[i])

        visuals = {}
        visuals['inputs'] = tensor2im(tvutil.make_grid(torch.cat([a[0] for a in val_data], dim=0)))
        visuals['reconstructions'] = tensor2im(tvutil.make_grid(torch.cat([a[1] for a in val_data], dim=0)))
        visuals['groundtruths'] = tensor2im(tvutil.make_grid(torch.cat([a[2] for a in val_data], dim=0)))  
        # show variance images
        _tmp = tvutil.make_grid(torch.cat([a[3] for a in val_data], dim=0), normalize=True, scale_each=True)
        _tmp.mul_(2).add_(-1)
        visuals['var'] = tensor2im(_tmp)  

        if need_sampling:
            # we need to do sampling
            sample_x, pixel_diff_mean, pixel_diff_std = self.sampling(self.display_data[0], n_samples=n_samples)
            import pdb; pdb.set_trace()
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

            # prior
            if need_sampling:
                # sampling
                fake_B = self.reparam(self.fake_B[:,:1,:,:], self.logvar)
                losses['sampling_loss'].append(float(F.mse_loss(fake_B[:,:1,...], self.real_B[:,:1,...], size_average=True)))
            else:
                losses['sampling_loss'] = losses['reconst_loss']

            # compute at prior
            fft_vis, fft_inv = self.compute_special_losses()
            losses['FFTVisiable_loss'].append(fft_vis)
            losses['FFTInvisiable_loss'].append(fft_inv)
            sys.stdout.write('\r validation [rec loss: %.5f]' % (np.mean(losses['reconst_loss'])))
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
