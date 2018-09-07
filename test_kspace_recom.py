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
from models.fft_utils import RFFT, IFFT, FFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
from util import visualizer
from models.fft_utils import create_mask
from pylab import *
rc('axes', linewidth=2)
import copy
import pickle
from util import util

rfft = RFFT()
ifft = IFFT()
fft = FFT()


def draw_curve5(value, std, pdfsavepath=None, ylabel=''):
    # plot mse of algorithm and its std
    global Percentage
    # uncertanity is a dict each value is a list a a[1] is var
    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
    fig = plt.figure(figsize=(10,10))
    plt.tick_params(labelsize=25)
    color = 'black' 
    ls = '-' 
    # data = [u[-1] for u in values] # -1 is var mean value
    # key_name = key.replace('_', '+')
    xval = [float(a)/len(value) * Percentage[-1] for a in range(1, len(value)+1)]
    plt.plot(xval, value, color=color, linewidth=5, linestyle=ls)
    plt.fill_between(xval, value-std, value+std, label='var.', 
                    alpha=0.2, edgecolor='#FFFFFF', facecolor='#c2d5db')

    # plt.legend(fontsize=25)
    plt.ylabel(ylabel, fontsize=40)
    plt.xlabel('kMA (%)', fontsize=40)

    if pdfsavepath is not None:
        plt.savefig(pdfsavepath)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def __draw_curve4(Pscore):
    # for percentile
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 3))
    # both has N * 3 matric
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10*2,10))
    plt.tight_layout()
    # assert mse.shape[1] == 3 and uncertainty.shape[1] == 3

    names = ['top10', 'median', 'bottom10']
    linestyles = ['-',':','--']
    for j, k in enumerate(Pscore.keys()):
        # import pdb; pdb.set_trace()
        # two method comparison
        mse = np.array(Pscore[k][0])
        uncertainty = np.array(Pscore[k][1])
        for i in range(3):
            key_name = f'{k.replace("_","")} {names[i]}'
            ax1.plot(mse[:,i], color=colors[j], label=key_name, linewidth=5, linestyle=linestyles[i])

        for i in range(3):
            key_name = f'{k.replace("_","")} {names[i]}'
            ax2.plot(uncertainty[:,i], color=colors[j], label=key_name, linewidth=5, linestyle=linestyles[i])

    ax1.set_ylabel('MSE', fontsize=40)
    ax1.set_xlabel('kMA', fontsize=40)
    ax2.set_ylabel('Uncertainty', fontsize=40)
    ax2.set_xlabel('kMA', fontsize=40)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.tick_params(labelsize=25)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve4(Pscore):
    # for percentile
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 3))
    # both has N * 3 matric
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10*2,10))
    plt.tight_layout()
    # assert mse.shape[1] == 3 and uncertainty.shape[1] == 3

    names = ['top10', 'median', 'bottom10']
    linestyles = ['-',':','--']
    for j, k in enumerate(Pscore.keys()):
        # two method comparison
        mse = np.array(Pscore[k][0])
        uncertainty = np.array(Pscore[k][1])
        x = range(len(mse[:,0]))
        ax1.errorbar(x, mse[:,0], color=colors[j], label=k.replace("_"," "), 
                        linewidth=5, linestyle=linestyles[i], yerr=mse[:,1])
        ax2.errorbar(x, uncertainty[:,0], color=colors[j], label=k.replace("_"," "), 
                        linewidth=5, linestyle=linestyles[i], yerr=uncertainty[:,1])

    ax1.set_ylabel('MSE', fontsize=40)
    ax1.set_xlabel('kMA', fontsize=40)
    ax2.set_ylabel('Uncertainty', fontsize=40)
    ax2.set_xlabel('kMA', fontsize=40)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.tick_params(labelsize=25)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve3(uncertainty, pdfsavepath=None, std=None):
    global Percentage
    # uncertanity is a dict each value is a list a a[1] is var
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
    fig = plt.figure(figsize=(10,10))
    plt.tick_params(labelsize=25)
    for i, (key, values) in enumerate(uncertainty.items()):
        color = 'black' if 'ours' in key else colors[i]
        ls = '-' if 'ours' in key else "--"
        data = [u[-1] for u in values] # -1 is var mean value
        key_name = key.replace('_', '+')
        xval = [float(a)/len(data) * Percentage[-1] for a in range(1, len(data)+1)]
        plt.plot(xval, data, color=color, label=key_name, linewidth=5, linestyle=ls)

    plt.legend(fontsize=25)
    plt.ylabel('Uncertainty score', fontsize=40)
    plt.xlabel('kMA (%)', fontsize=40)

    
    if pdfsavepath is not None:
        plt.savefig(pdfsavepath)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve2(data, pdfsavepath=None):
    global Percentage
    # for discriminator score
    fig = plt.figure(figsize=(15,10))
    plt.tick_params(labelsize=25)
    fig.add_subplot(111)
    # x = np.arange(data.shape[0])
    xval = [float(a)/data.shape[0] * Percentage[-1] for a in range(1, data.shape[0]+1)]
    plt.plot(xval, data[:,0], 'k', label='ours', linewidth=10)
    plt.fill_between(xval, data[:,0]-data[:,1], data[:,0]+data[:,1],
                    alpha=0.2, edgecolor='#FFFFFF', facecolor='#c2d5db')
    plt.ylabel('Discriminator score', fontsize=40)
    plt.xlabel('kMA (%)', fontsize=40)
    # plt.title('avgerage reconstruced k-space D-score', fontsize=30)
    plt.legend(fontsize=25)

    
    if pdfsavepath is not None:
        plt.savefig(pdfsavepath)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve(data, logscale=False, pdfsavepath=None):
    # for MSE and SSIM
    global Percentage
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))

    fig = plt.figure(figsize=(20,10))
    n = len(data)
    fig.add_subplot(121)

    for i, (key, values) in enumerate(data.items()):
        color = 'black' if 'ours' in key else colors[i]
        ls = '-' if 'ours' in key else "--"
        key_name = key.replace('_', '+')
        xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
        if logscale:
            yval = np.exp(values[:,0])
        else:
            yval = values[:,0]
        plt.plot(xval, yval, color=color, label=key_name, linewidth=5, linestyle=ls)
    if logscale:
        plt.ylabel('exp MSE', fontsize=40)
    else:
        plt.ylabel('MSE', fontsize=40)
    plt.xlabel('kMA (%)', fontsize=40)
    plt.legend(fontsize=25)
    plt.tick_params(labelsize=25)

    fig.add_subplot(122)
    for i, (key, values) in enumerate(data.items()):
        color = 'black' if 'ours' in key else colors[i]
        ls = '-' if 'ours' in key else "--"
        key_name = key.replace('_', '+')
        xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
        plt.plot(xval, values[:,1], color=color, label=key_name, linewidth=5, linestyle=ls)
    plt.ylabel('SSIM', fontsize=40)
    plt.xlabel('kMA (%)', fontsize=40)
    plt.legend(fontsize=25)
    
    plt.tick_params(labelsize=25)
    
    if pdfsavepath is not None:
        plt.savefig(pdfsavepath)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def merge_kpsace(pred, gt, mask):
    if is_rawdata:
        p_fft = fft(pred[:,:2,:,:])
        g_fft = fft(gt[:,:2,:,:])
    else:
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
        if is_rawdata and conjudge_symmetric:
            # Experimental: Add conjudge 
            mask_score.masked_fill_(mask_new.byte(), 1000000)
            inver_idx = torch.Tensor(list(np.arange(cs_h//2,cs_h,1)[::-1])).long()
            mask_score[:,1:cs_h//2+1] = mask_score[:,1:cs_h//2+1] + mask_score[:,inver_idx]
            min_ranked_score, indices = torch.sort(mask_score[:,:cs_h//2+1], 1, descending=False)
            indices = indices[:,:n_line]
        else:
            mask_score.masked_fill_(mask_new.byte(), 1000000) # do not select from real one set it to a large value
            min_ranked_score, indices = torch.sort(mask_score, 1, descending=False)
            indices = indices[:,:n_line]
    else:
        indices = random

    for i, ind in enumerate(indices):
        try:
            if mask_new[i, ind] == 1:
                import pdb; pdb.set_trace()
            mask_new[i, ind] = 1
            if conjudge_symmetric:
                mask_new[i, cs_h-ind] = 1
        except:
            print(i, ind)

    sys.stdout.write(f'\r mask ratio: {mask_new[0].sum()}')
    sys.stdout.flush()
    mask_new = mask_new.unsqueeze(1).unsqueeze(3)
    results = merge_kpsace(pred, gt, mask_new)

    return results, mask_new

global InfoPerc
InfoPerc = []
def compute_scores(fake, real, mask):
    if is_rawdata:
        assert fake.shape[1] == 2 and real.shape[1] == 2
        fake = fake.norm(dim=1, keepdim=True)
        real = real.norm(dim=1, keepdim=True)
    else:
        fake = fake[:,:1,:,:]#.clamp(-3,3)
        real = real[:,:1,:,:]
    mse = F.mse_loss(fake, real)
    ssim = util.ssim_metric(fake, real)
    
    std = F.mse_loss(fake, real, reduce=False).view(fake.shape[0],-1).mean(1).std()

    # get_mse_percentile(fake, real)
    # # compute information percentage:
    # assert mask.shape[2] == 128 
    # indices = mask.squeeze().mul(1)
    # h = indices.shape[1]
    # pers = []
    # for index in indices:
    #     # 2nd line to 64 are conjudge symmetric to the other part
    #     tmp = index.cpu().numpy()[1:64] + np.array(index.tolist()[:64:-1])
    #     per = np.nonzero(tmp)[0].size / 65
    #     pers.append(per*100//1 /100)
    # InfoPerc.append(array(pers))   
    return [mse.item(), ssim.item(), std.item()]

# to track the MSE per kspace recom and corresponding uncertainty 
# for top10, bottom10 and median
global MSE, Uncertainty
IndicesP = []
MSE = []
Uncertainty = []
## --------------
# deprecated
def __get_mse_percentile(fake, real):
    mse = F.mse_loss(fake, real, reduce=False)
    mse = mse.view(mse.shape[0],-1).mean(1)
    _, indices = mse.sort(dim=0, descending=False)
    l = len(indices)
    top10 = indices[:int(l*0.1)]
    bottom10 = indices[-int(l*0.1):]
    median = indices[l//2]
    global IndicesP
    IndicesP = [top10, median, bottom10]

    _tmp = []
    for v in IndicesP:
        _tmp.append(mse[v].mean().item())
    MSE.append(_tmp)
def __get_mse_percentile_for_uncertainty(logvar):
    uncertainties = logvar.exp().view(logvar.shape[0],-1).mean(1)
    global IndicesP
    assert len(IndicesP) ==3
    
    _tmp = []
    for v in IndicesP:
        _tmp.append(uncertainties[v].mean().item())
    Uncertainty.append(_tmp)
    IndicesP = []
def get_mse_percentile(fake, real):
    mse = F.mse_loss(fake, real, reduce=False)
    _tmp = [mse.mean(), mse.std()]
    MSE.append(_tmp)
def get_mse_percentile_for_uncertainty(logvar):
    uncertainties = logvar.exp()
    Uncertainty.append([uncertainties.mean(), uncertainties.std()])
    IndicesP = []
## --------------

def compute_D_score(score, mask):
    masked_pred_kspace_score = (score * (1-mask.squeeze())).reshape(-1)
    masked_pred_kspace_score = masked_pred_kspace_score[mask.squeeze().reshape(-1) == 0]
    if len(masked_pred_kspace_score) == 0:
        return [0, 0]
    else:
        return [masked_pred_kspace_score.mean(), masked_pred_kspace_score.std()]

def compute_uncertainty(logvars, ngrid=4):
    var = [logvar.exp().mean().item() for logvar in logvars]
    var_map = logvars[-1][:ngrid].exp()
    # var_map = var_map.div(var_map.max())
    uncertainty_map = util.tensor2im(tvutil.make_grid(var_map, nrow=int(np.sqrt(ngrid)), padding=10), renormalize=False)
    uncertainty_map = visualizer.gray2heatmap(uncertainty_map[:,:,0])

    get_mse_percentile_for_uncertainty(logvars[-1])

    return [uncertainty_map, var[-1]]

def compute_recontruct_uncertainty(reconstruction, uncertainty, ngrid=4):
    if is_rawdata:
        assert reconstruction.shape[1] == 2
        reconstruction = reconstruction.norm(dim=1, keepdim=True)
        zscore = model.zscore
    else:
        zscore = 3
    reconstruction = reconstruction[:ngrid,:1].cpu()
    uncertainty = uncertainty[:ngrid].cpu()

    uncertainty_map = util.tensor2im(tvutil.make_grid(uncertainty, nrow=int(np.sqrt(ngrid)), padding=10), renormalize=False)
    uncertainty_map = visualizer.gray2heatmap(uncertainty_map[:,:,0])

    reconstruction = util.mri_denormalize(reconstruction, zscore=zscore)
    reconstruction = util.tensor2im(tvutil.make_grid(reconstruction, nrow=int(np.sqrt(ngrid)), padding=10), renormalize=False)

    return [reconstruction, uncertainty_map]

def compute_errormap(real, fake, ngrid=4):
    if is_rawdata:
        fake = fake.norm(dim=1, keepdim=True)
        real = real.norm(dim=1, keepdim=True)
    error = (real - fake)[:ngrid,:1]**2
    error = error.sqrt()
    errormap = util.tensor2im(tvutil.make_grid(error, nrow=int(np.sqrt(ngrid)), padding=10, pad_value=0.6), renormalize=False)
    return errormap

def tensor2img(data, ngrid=4):
    if is_rawdata:
        data = data.norm(dim=1, keepdim=True)
        zscore = model.zscore
    else:
        zscore = 3
    data = util.mri_denormalize(data[:ngrid,:1].mul(1), zscore=zscore)
    imgs = util.tensor2im(tvutil.make_grid(data, nrow=int(np.sqrt(ngrid)), padding=10, pad_value=0.6), renormalize=False)
    return imgs
def compute_mask(masks, ngrid=4):
    masks = masks[:ngrid].repeat(1,1,1,opt.fineSize)
    masks = util.tensor2im(tvutil.make_grid(masks, nrow=int(np.sqrt(ngrid)), padding=10, pad_value=0.6), renormalize=False)
    return masks

def animate2(images, error_map, iter, start_line=33):
    ''' animate the uncertainty maps as # of lines increase '''
    # images[0] includes [reconstruction, uncertainty]
    def normalize(data):
        maxv = 0
        for i in range(len(data)):
            # clamp use a small value (<1) so the init visualization is more shaper
            maxv = max(data[i].clamp_(0,0.6).max().item(), maxv)
        for i in range(len(data)):
            data[i].div_(maxv)

    ims = []
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(5*3+1,5), gridspec_kw={'width_ratios':[5, 5, 5, 1]})
    plt.tight_layout()

    # plt.title(f'# of measurement acquired: {i}')
    ax1.set_title('Reconstruction', fontsize=16)
    ax2.set_title('Error Map', fontsize=16)
    ax3.set_title('Uncertainty Map', fontsize=16)
    ax4.set_title('kMA (%)', fontsize=16)
    tot = len(images)
    
    global Percentage
    xval = [int(float(a)/tot * Percentage[-1]) for a in range(1, tot+1)]

    normalize([a[1] for a in images])

    # import pdb; pdb.set_trace()
    for i, n in enumerate(range(start_line, tot+start_line)):
        
        rec, unc = compute_recontruct_uncertainty(images[i][0], images[i][1])
        kMA = xval[i]
        ax1.axis('off')
        im = ax1.imshow(rec, animated=True)

        ## Mask 
        ax2.axis('off')
        im2 = ax2.imshow(error_map[i], animated=True)
        
        ax3.axis('off')
        im3 = ax3.imshow(unc, animated=True)

        im4, = ax4.plot(np.ones(i+1), np.array(xval[:i+1]), animated=True, lw=20, color='darkred')
        ax4.xaxis.set_visible(False) # same for y axis.

        ims.append([im, im2, im3, im4])

    plt.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000)
    global opt
    global save_dir
    name = f"demo_uncertainty{iter}.mp4"
    save_path = os.path.join(save_dir, name)
    ani.save(save_path)
    print('\n -> video saved at', save_path)

    gif_name = f'gif_demo_uncertainty{iter}.gif'
    if save_gif:
        ani.save(os.path.join(save_dir, gif_name), writer='imagemagick', fps=10)
    return name, gif_name

def animate(images, masks, eval_scores, uncertainty, D_score, iter,
            comp_score=None, start_line=33):
    ''' animate the acqusition planning process '''
    # scores[0][0] is mse and [0][1] is ssim
    # images[0] is a image grid for error map
    # animate images with scires
    # uncertainty[0] image uncertainty overall value
    # comp_score has multiple comparative results in tuple
    def normalize(data):
        import pdb; pdb.set_trace()
        for i in range(len(data)):
            maxv = 0
            maxv = max(data[i].max().item(), maxv)
        for i in range(len(uncertainty)):
            data[i] = data[i] / (maxv)

    ims = []
    k = 5
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,k, figsize=(5*k,5))
    plt.tight_layout()

    # plt.title(f'# of measurement acquired: {i}')
    ax1.set_title('Error Map', fontsize=16)
    ax2.set_title('Cartesian Subsampling Mask', fontsize=16)
    ax3.set_title('MSE', fontsize=16)
    ax4.set_title('Uncertainty', fontsize=16)
    ax5.set_title('k-space Score', fontsize=16)

    # normalize([u[0] for u in uncertainty])
    # normalize(images)

    x, y1, y2, y3, y5 = [], [], [], [], []
    tot = len(images)

    plot_property = {
       'color':'#131925', 
       'marker':'o', 
       'markeredgecolor':'#EE0000', 
       'markerfacecolor':'#EE0000', 
       'lw':5,
       'markeredgewidth': 0.5   
    }
    plot_property2 = {
       'color':'gray', 
       'lw':5,
       'linestyle': ":",
    }
    plot_property3 = {
       'color':'cyan', 
       'lw':5,
       'linestyle': "--",
    }
    global Percentage
    xval = [int(float(a)/tot * Percentage[-1]) for a in range(1, tot+1)]

    for i, n in enumerate(range(start_line, tot+start_line)):
        
        kMA = xval[i]
        ax1.axis('off')
        im = ax1.imshow(images[i], animated=True)

        ## Mask 
        ax2.axis('off')
        im2 = ax2.imshow(masks[i], animated=True)

        x.append(kMA)
        y1.append(eval_scores[i][0])
        
        im3, = ax3.plot(x, y1, **plot_property, label='algorithm' if i == 0 else "")
        # text3 = ax3.text(5, 5, f'measurement {i}', color='b', fontsize=16, transform=ax3.transAxes,)
        title3 = ax3.text(0.8,0.9," kMA {}%".format(kMA), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax3.transAxes, fontsize=16)
        ax3.set_xlabel('kMA (%)', fontsize=16)
        if comp_score is not None:
            im3_2, = ax3.plot(x, [u[0] for u in comp_score[1][:i+1]], **plot_property3, label='random_reconstruction' if i == 0 else "")
            im3_3, = ax3.plot(x, [u[0] for u in comp_score[0][:i+1]], **plot_property2, label='random' if i == 0 else "")
            legend = ax3.legend(loc='lower left')
        ax3.set_xlim([0, x[-1]+1])
        
        ## SSIM score
        # y2.append(eval_scores[i][1])
        # im3, = ax3.plot(x, y2, color='gray', marker='o', markeredgecolor='r')
        # text1 = ax3.text(2,2, f'measurement {i}',animated=True, color='red', fontsize=10)
        # ax3.set_xlim([0, x[-1]+1])

        im4, = ax4.plot(x, [u[-1] for u in uncertainty[:i+1]], **plot_property)
        title4 = ax4.text(0.8,0.9," kMA {}%".format(kMA), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax4.transAxes, fontsize=16)
        ax4.set_xlabel('kMA (%)', fontsize=16)
        ax4.set_xlim([0, x[-1]+1])

        avg_D = np.array([u[0] for u in D_score[:i+1]])
        std_D = np.array([u[1] for u in D_score[:i+1]])

        im5, = ax5.plot(x, avg_D, **plot_property, label='score')
        fill5 = ax5.fill_between(x, avg_D-std_D, avg_D+std_D, label='var.', 
                    alpha=0.2, edgecolor='#FFFFFF', facecolor='#c2d5db')
        title5 = ax5.text(0.2,0.9," kMA {}%".format(kMA), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax5.transAxes, fontsize=16)
        ax5.set_xlabel('kMA (%)', fontsize=16)
        ax5.set_xlim([0, x[-1]+1])

        if comp_score is not None:
            ims.append([im, im2, im3, im3_2, im3_3, im4, im5, fill5, title3, title4, title5])
        else:
            ims.append([im, im2, im3, im4, im5, fill5, title3, title4, title5])
    plt.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000)
    global opt
    global save_dir
    name = f"demo{iter}.mp4"
    save_path = os.path.join(save_dir, name)
    ani.save(save_path)
    print('\n -> video saved at', save_path)

    gif_name = f'gif_demo{iter}.gif'
    if save_gif:
        ani.save(os.path.join(save_dir, gif_name), writer='imagemagick', fps=10)
    return name, gif_name

opt = None
save_dir = None
"---------Parameter settings--------"
## If debug, we just recommend a few lines and show results
debug = False
## If conjudge_symmetric is True. We select a line and this conjudge symmetric one is automatically selected
## The code will adapt automatically for that 
conjudge_symmetric=True
## How many lines we observed at init. If it is euqal to 0, we use the LF 10 lines
## If = 25, we use the default 25% subsampling pattern initially
observed_line_n = 0
## If n_forward is None, it will calculate automatically. For debug, set it to a small value
n_forward = None if not debug else 10
tot_iter = 2
# for displaying the xval of plot, the percentage of oberved lines
Percentage = [] 
# If save metadata
save_metadata = True
# If save gif together with mp4 demo version
save_gif = True
# If it is raw k-space data
is_rawdata = False
"------------------------------------" 

if __name__ == '__main__':
    
    opt = TestOptions().parse()
    assert (opt.no_dropout)

    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)
    model = create_model(opt)
    model.setup(opt)
    cs_h = 128
    
    if 'raw' in opt.name.lower():
        # conjudge_symmetric = False
        is_rawdata = True
        print(f'>> raw data: conjudge_symmetric: {conjudge_symmetric}, is_rawdata: {is_rawdata}')
        cs_h = 320

    val_count = 0
    # saved = False
    ngrid = 9 # how many image to show in each grid
    use_forward = True

    if debug:
        print('-'*100)
        print('>> Warning in debug mode')

    # create website
    if conjudge_symmetric:
        web_dir = os.path.join(opt.results_dir, opt.name, 'test_recommend_%s' % (opt.which_epoch) if not debug else 'test_recommend_%s_debug' % (opt.which_epoch))
    else:
        web_dir = os.path.join(opt.results_dir, opt.name, 'test_recommend_%s_nocs' % (opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    save_dir = webpage.get_image_dir()
    # test
    if not hasattr(model, 'sampling'): opt.n_samples = 1 # determnistic model needs no sampling
    
    if model.mri_data:
        tensor2im = functools.partial(util.tensor2im, renormalize=False)
    else:
        tensor2im = util.tensor2im

    '''parameters need to set'''
    if conjudge_symmetric:
        observed_line_n = 0 #int(init_ratio * opt.fineSize)

    if observed_line_n != 32:
        init_mask = create_mask(opt.fineSize, random_frac=False, mask_fraction=observed_line_n / opt.fineSize).to(model.device)
        observed_line_n = int(init_mask[0].sum().item())
        print('>> start observed line: ', observed_line_n)

    if conjudge_symmetric:
        print('>> keep conjudge symmetric')
        for jj in range(len(init_mask)):
            for j in range(1,cs_h-1):
                if init_mask[jj, 0, j, 0] == 1:
                    init_mask[jj, 0, cs_h-j, 0] = 1

    unobsered_line_n = opt.fineSize - observed_line_n
    n_recom_each = 1
    if n_forward is None:
        if conjudge_symmetric:
            n_forward = unobsered_line_n // 2 
        else:
            n_forward = unobsered_line_n // n_recom_each
    imsize = opt.fineSize
    np.random.seed(1234)

    D_bias = OrderedDict()
    print(f'>> start from {observed_line_n} lines and add {n_recom_each} line per forward, need {n_forward} forward')
    for i, data in enumerate(test_data_loader):
        
        print('\n Conduct iteration', i)
        score_list = OrderedDict()
        visual_list = OrderedDict()
        uncertainty_list = OrderedDict()
        score_D = []
        visuals = {}

        if observed_line_n != 32:
            if init_mask.shape[0] == 1:
                init_mask = init_mask.repeat(data[0].shape[0],1,1,1)
            model.set_input_exp(data[1:], init_mask)
        else:
            model.set_input(data)

        '''Start point and init all scores'''
        model.test()
        mask = model.mask
        (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
        realB = model.real_B
        fakeB = model.fake_B
        fakeB = model.fake_B
        init_logvars = model.logvars
        
        visual_list['groundtruth'] = tensor2img(realB)
        visual_list['input'] = tensor2img(model.real_A)
        score_list['algorithm'] = [compute_scores(fakeB, realB, mask)]
        visual_list['algorithm_errormap'] = [compute_errormap(fakeB, realB)]
        score_list['algorithm_mask'] = [ compute_mask(mask) ]
        uncertainty_list['algorithm']= [compute_uncertainty(init_logvars)]
        
        score_D.append(compute_D_score(pred_kspace_score, mask))
        reconstruction_list = [ [fakeB.cpu(), model.logvars[-1].cpu().exp()] ] # for visualize video of reconstruction and uncertainty
        
        PScore = {} # Pertentile score
        InfoPScore = {} # Information Percentile score
        
        bz = fakeB.shape[0]

        '''Start to recommend'''
        print('\n step 1 algorithm: ')
        old_fakeB = fakeB
        old_mask = mask
        old_mask_count = old_mask.squeeze().sum(1).mean()
        D_bias[f'iter{i}'] = []
        Percentage = [old_mask[0].sum().item() / opt.fineSize]
        for nf in range(n_forward):
            # print(f'real_A range {model.real_A.min()}-{model.real_A.max()}')
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, pred_kspace_score, None)
            new_mask_count = new_mask.squeeze().sum(1).mean()
            mask_choice = new_mask - old_mask
            # assert new_mask_count == old_mask_count + n_recom_each, f'{new_mask_count} != {old_mask_count} + {n_recom_each}'
            D_bias[f'iter{i}'].append(torch.nonzero(mask_choice.squeeze()).cpu().numpy())
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

            reconstruction_list.append([old_fakeB.cpu(), model.logvars[-1].cpu().exp()])
            score_list['algorithm'].append(compute_scores(old_fakeB, realB, old_mask))
            uncertainty_list['algorithm'].append(compute_uncertainty(model.logvars))
            visual_list['algorithm_errormap'].append(compute_errormap(old_fakeB, realB))
            score_list['algorithm_mask'].append(compute_mask(old_mask))
            Percentage.append(old_mask[0].sum().item() / opt.fineSize)

            if nf < (n_forward-1) and old_mask[1].sum() == opt.fineSize:
                print ('>> reach the all lines earlier. break')
                break
        TOT_SEEN_LINE = old_mask[1].sum()

        score_D = score_D[:-1] # the last one has no meaning 
        score_D.append(score_D[-1])
        Percentage = [int(np.around(xval * 100)) for xval in Percentage]

        PScore['algorithm'] = [copy.deepcopy(MSE), copy.deepcopy(Uncertainty)]
        # InfoPScore['algorithm'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], []

        ## generate video demo
        video_path2, gif_path2 = animate2(reconstruction_list, visual_list['algorithm_errormap'], start_line=observed_line_n, iter=i)
        video_path, gif_path = animate(visual_list['algorithm_errormap'], score_list['algorithm_mask'], score_list['algorithm'], uncertainty_list['algorithm'], 
                        score_D, start_line=observed_line_n, iter=i)
        
        
        ''' Start to random select lines '''
        score_list['random'] = [compute_scores(fakeB, realB, mask)]
        
        # random some lines order
        bz = mask.shape[0]
        tmp_mask = mask.mul(1)
        if conjudge_symmetric:
            tmp_mask[:,:,cs_h//2+1:,:] = 1
        mask_zero_idx = np.nonzero(1-tmp_mask[0].squeeze())
        if not conjudge_symmetric:
            assert len(mask_zero_idx) == unobsered_line_n
        lines = [np.random.permutation(mask_zero_idx) for _ in range(bz)] # permute for each data
        cand_lines = np.stack(lines, 0).squeeze()

        old_fakeB = fakeB
        old_mask = mask
        print('\n step 2 random: ')
        
        for s in range(n_forward):
            cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

            old_fakeB = new_fakeB
            old_mask = new_mask
            score_list['random'].append(compute_scores(old_fakeB, realB, old_mask))
            if old_mask[1].sum() == opt.fineSize:
                print ('>> [random] reach the all lines earlier. break')
                break
        assert old_mask[1].sum() == TOT_SEEN_LINE
        # InfoPScore['random'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], []

        if use_forward:
            print('\n step 3 random + reconstruction: ')
            score_list['random_reconstruction'] = [compute_scores(fakeB, realB, mask)]
            uncertainty_list['random_reconstruction']= [compute_uncertainty(init_logvars)]
            old_fakeB = fakeB
            old_mask = mask
                        
            for s in range(n_forward):
                cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
                # calcuate which line to replace and replace it
                new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

                #do another forward
                model.set_input_exp(data[1:], new_mask)
                model.test()
                # (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
                old_fakeB = model.fake_B
                old_mask = new_mask
                score_list['random_reconstruction'].append(compute_scores(old_fakeB, realB, old_mask))
                uncertainty_list['random_reconstruction'].append(compute_uncertainty(model.logvars))
                if old_mask[1].sum() == opt.fineSize:
                    break
            assert old_mask[1].sum() == TOT_SEEN_LINE
        PScore['random_reconstruction'] = [copy.deepcopy(MSE), copy.deepcopy(Uncertainty)]

        # InfoPScore['random_reconstruction'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], {}
        ## animate compared results also      
        # animate(score_list['algorithm_errormap'], score_list['algorithm_mask'], score_list['algorithm'], score_list['algorithm_uncertainty'], 
        #             score_D, (score_list['random'], score_list['random_reconstruction']), start_line=observed_line_n)

        print('\n step 4 order: ')
        score_list['order'] = [compute_scores(fakeB, realB, mask)]

        mask_zero_idx = np.nonzero(1-mask[0].squeeze())
        if not conjudge_symmetric:
            assert len(mask_zero_idx) == unobsered_line_n
        lines = [np.sort(mask_zero_idx, axis=0) for _ in range(bz)] # permute for each data
        lines = [a if np.random.rand() < 0.5 else a[::-1] for a in lines]
        cand_lines = np.stack(lines, 0).squeeze()

        old_fakeB = fakeB
        old_mask = mask
        
        for s in range(n_forward):
            cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

            old_fakeB = new_fakeB
            old_mask = new_mask
            score_list['order'].append(compute_scores(old_fakeB, realB, old_mask))
            if old_mask[1].sum() == opt.fineSize:
                break

        print('\n step 5 order + reconstruction: ')
        old_fakeB = fakeB
        old_mask = mask
        score_list['order_reconstruction'] = [compute_scores(fakeB, realB, mask)]
        uncertainty_list['order_reconstruction']= [compute_uncertainty(init_logvars)]

        for s in range(n_forward):
            cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

            # do another forward
            model.set_input_exp(data[1:], new_mask)
            model.test()
            (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
            old_fakeB = model.fake_B
            old_mask = new_mask
            score_list['order_reconstruction'].append(compute_scores(old_fakeB, realB, old_mask))
            uncertainty_list['order_reconstruction'].append(compute_uncertainty(model.logvars))
            if old_mask[1].sum() == opt.fineSize:
                break
        assert old_mask[1].sum() == TOT_SEEN_LINE
        # InfoPScore['order_reconstruction'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], []

        "record everthing thing"
        score = OrderedDict()
        score_D = np.array(score_D)
        score_uncetainty = OrderedDict()

        score['random_C'] = np.array(score_list['random'])
        if use_forward:
            score['random_C_R'] = np.array(score_list['random_reconstruction'])
        score['order_C'] = np.array(score_list['order'])
        score['order_C_R'] = np.array(score_list['order_reconstruction'])
        score['ours_C_R'] = np.array(score_list['algorithm'])
        
        # curves of uncertainty
        score_uncetainty['random_C_R'] = uncertainty_list['random_reconstruction']
        score_uncetainty['order_C_R'] = uncertainty_list['order_reconstruction']     
        score_uncetainty['ours_C_R'] = uncertainty_list['algorithm']   

        ## draw curves 
        save_dir = webpage.get_image_dir()
        visuals['performance'] = draw_curve(score, pdfsavepath=os.path.join(save_dir, 'performance.pdf'))
        # visuals['performance_log'] = draw_curve(score,logscale=True)
        visuals['D-score'] = draw_curve2(score_D, pdfsavepath=os.path.join(save_dir, 'D-score.pdf'))
        visuals['uncertainty'] = draw_curve3(score_uncetainty, pdfsavepath=os.path.join(save_dir, 'uncertainty.pdf'))
        visuals['mse_std'] = draw_curve5(score['ours_C_R'][:,0], std=score['ours_C_R'][:,2], pdfsavepath=os.path.join(save_dir, 'mse_std.pdf'), ylabel='MSE') # the last column is std

        visuals['demo'] = video_path
        visuals['demo_uncertainty'] = video_path2
        visuals['groundtruth'] = visual_list['groundtruth'] 
        visuals['input'] = visual_list['input'] 
        
        if save_gif:
            visuals['gif'] = gif_path
            visuals['gif_uncertainty'] = gif_path2

        # save data
        if save_metadata:
            MetaData = {}
            MetaData['Percentage'] = Percentage
            MetaData['data'] = data
            MetaData['score'] = score
            MetaData['score_D'] = score_D
            MetaData['uncetainty'] = score_uncetainty
            
            meta_name = f'metadata{i}.pickle'
            with open(os.path.join(save_dir, meta_name),'wb') as f:
                pickle.dump(MetaData, f)
                print('save metadata at', os.path.join(save_dir, meta_name))

            visuals['metadata'] = meta_name

        ## (deprecated) draw percentile mse vs uncertainty
        ## visuals['model_percentile'] = draw_curve4(PScore)

        ## save figures and everthing
        save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        webpage.save()
        val_count += bz
        if i >= tot_iter: break

    # discriminator bias experiment
    savename = 'd_bias.pickle'
    with open(savename,'wb') as f:
        pickle.dump(D_bias, f)
        print('save metadata at', savename)