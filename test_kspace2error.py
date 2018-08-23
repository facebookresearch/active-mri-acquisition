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
import functools
from util.util import ssim_metric
from models.fft_utils import RFFT, IFFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

rfft = RFFT()
ifft = IFFT()


def draw_curve2(data):
    color = ['b']
    fig = plt.figure(figsize=(20,10))
    fig.add_subplot(111)
    x = np.arange(data.shape[0])
    plt.plot(x, data[:,0], 'r', label='algorithm', linewidth=10)
    plt.fill_between(x, data[:,0]-data[:,1], data[:,0]+data[:,1],
                    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.ylabel('Discriminator score', fontsize=30)
    plt.xlabel('# of recommended kspace line', fontsize=30)
    plt.title('avgerage reconstruced k-space D-score', fontsize=30)
    plt.legend(fontsize=25)

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def draw_curve(data):
    color = ['r','g','b']
    fig = plt.figure(figsize=(20,10))
    n = len(data)
    fig.add_subplot(121)
    for i, (key, values) in enumerate(data.items()):
        plt.plot(values[:,0], color[i], label=key, linewidth=5)
    plt.ylabel('MSE', fontsize=30)
    plt.xlabel('# of recommended kspace line', fontsize=30)
    plt.legend(fontsize=25)

    fig.add_subplot(122)
    for i, (key, values) in enumerate(data.items()):
        plt.plot(values[:,1], color[i], label=key, linewidth=5)
    plt.ylabel('SSIM', fontsize=30)
    plt.xlabel('# of recommended kspace line', fontsize=30)
    plt.legend(fontsize=25)
    
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def merge_kpsace(pred, gt, mask):
    p_fft = rfft(pred[:,:1,:,:])
    g_fft = rfft(gt[:,:1,:,:])
    assert len(mask.shape) == 4
    assert mask.shape[1] == 1 and mask.shape[3] == 1
    res = g_fft * mask + p_fft * (1-mask)
    res = ifft(res)

    return res

def replace_kspace_lines(pred, gt, mask, n_line, mask_score, random=None):
    # data [B,1,128,128]
    # mask [B,1,128,1]
    # mask_score [B,128]
    # random [B,n_line]
    mask_new = mask.mul(1).squeeze() # copy mem
    if random is None:
        mask_score.masked_fill_(mask_new.byte(), 100) # do not select from real one set it to a large value
        min_ranked_score, indices = torch.sort(mask_score, 1, descending=False)
        indices = indices[:,:n_line]
    else:
        indices = random

    for i, ind in enumerate(indices):
        mask_new[i, ind] = 1
    sys.stdout.write(f'\r mask ratio: {mask_new[0].sum()}')
    sys.stdout.flush()
    mask_new = mask_new.unsqueeze(1).unsqueeze(3)
    results = merge_kpsace(pred, gt, mask_new)

    return results, mask_new

def compute_scores(fake, real):
    fake = fake[:,:1,:,:]#.clamp(-3,3)
    real = real[:,:1,:,:]
    mse = F.mse_loss(fake, real)
    ssim = util.ssim_metric(fake, real)

    return [mse.item(), ssim.item()]

def mse_loss(src, tar):
    
    src = src[:,:1,:,:]
    tar = tar[:,:1,:,:]

    err = float(F.mse_loss(src, tar, size_average=True)) 
    return err 

def compute_D_score(score, mask):
    masked_pred_kspace_score = (score * (1-mask.squeeze())).reshape(-1)
    masked_pred_kspace_score = masked_pred_kspace_score[mask.squeeze().reshape(-1) == 0]
    if len(masked_pred_kspace_score) == 0:
        return [0, 0]
    else:
        return [masked_pred_kspace_score.mean(), masked_pred_kspace_score.std()]


if __name__ == '__main__':
    opt = TestOptions().parse()
    assert (opt.no_dropout)

    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)

    model = create_model(opt)
    model.setup(opt)

    val_count = 0
    # saved = False
    ngrid = 9 # how many image to show in each grid
    
    use_forward = True

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, 'test_recommend_%s' % (opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    
    # test
    total_test_n = len(test_data_loader)*opt.batchSize if opt.how_many < 0 else (opt.how_many)
    if not hasattr(model, 'sampling'): opt.n_samples = 1 # determnistic model needs no sampling
    
    if model.mri_data:
        tensor2im = functools.partial(util.tensor2im, renormalize=False)
    else:
        tensor2im = util.tensor2im

    observed_line_n = int(0.25 * opt.fineSize)
    unobsered_line_n = opt.fineSize - observed_line_n
    n_recom_each = 1
    n_forward = 40 # unobsered_line_n // n_recom_each
    imsize = opt.fineSize
    np.random.seed(1234)

    for i, data in enumerate(test_data_loader):
        score_list = []
        score_list_random = []
        score_list_random2 = []
        score_D = []
        visuals = {}

        model.set_input(data)
        
        model.test()
        mask = model.mask
        (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
        realB = model.real_B
        fakeB = model.fake_B
        
        score_list.append(compute_scores(fakeB, realB))
        score_list_random.append(compute_scores(fakeB, realB)) # randome selection + repalce
        score_list_random2.append(compute_scores(fakeB, realB)) # random selection + network forward

        score_D.append(compute_D_score(pred_kspace_score, mask))

        bz = fakeB.shape[0]
        '''Start to recommend'''
        print('step 1 algorithm: ')
        old_fakeB = fakeB
        old_mask = mask
        old_mask_count = old_mask.squeeze().sum(1).mean()
        for _ in range(n_forward):
            # print(f'real_A range {model.real_A.min()}-{model.real_A.max()}')
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, pred_kspace_score, None)
            new_mask_count = new_mask.squeeze().sum(1).mean()
            assert new_mask_count == old_mask_count + n_recom_each, f'{new_mask_count} != {old_mask_count} + {n_recom_each}'

            if use_forward:
                #do another forward
                model.set_input_exp(data[1:], new_mask)
                model.test()
                (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
                old_fakeB = model.fake_B
                score_D.append(compute_D_score(pred_kspace_score, new_mask))
            else:
                old_fakeB = new_fakeB
            old_mask = new_mask
            old_mask_count = old_mask.squeeze().sum(1).mean()
            score_list.append(compute_scores(old_fakeB, realB))
        score_D = score_D[:-1] # the last one has no meaning 

        
        save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        webpage.save()

        val_count += bz


        if val_count >= opt.how_many and opt.how_many > 0:
            break
        if i >= 1: break



    # save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()   
    #  
    # import pdb; pdb.set_trace()
    # if i >= 10: break

        # import pdb; pdb.set_trace()

        # visuals = model.get_current_visuals()
        # val_count += ngrid
        # # Only save some. Don't explode the disk
        # if val_count < 256:
        #     for k, v in visuals.items():
        #         v = v[:ngrid,:1,:,:]
        #         if model.mri_data:
        #             v = util.mri_denormalize(v)
        #         visuals[k] = tensor2im(tvutil.make_grid(v, nrow=int(np.sqrt(ngrid))))
            
        #     save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        # elif not saved:
        #     webpage.save()
        #     saved = True
            
        # if val_count >= opt.how_many and opt.how_many > 0:
        #     break
