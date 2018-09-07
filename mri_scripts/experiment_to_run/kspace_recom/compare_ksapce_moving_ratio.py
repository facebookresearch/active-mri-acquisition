import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot  import imsave
plt.switch_backend('agg')

def set_plot(ax, xlabel, ylabel):
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.legend(fontsize=25)
    ax.tick_params(labelsize=25)
    ax.legend(fontsize=25)
    plt.tick_params(labelsize=25)

path = '/private/home/zizhao/work/checkpoint_fmri/mri_session_exp/'
methods = {
    'ResNet': 'knee_baseline_resnetzz',
    'PasNet (w/ u.)': 'knee_pasnetplus_uncertainty_w111',
    'PasNet (w/o u.)': 'knee_pasnetplus_nouncertainty_w111',
    
    'DenseNet': 'knee_baseline_denset103_residual',
    'UNet': 'knee_baseline_unet',
    'pix2pix': 'knee_baseline_pixl2pix_residual'
    # 'Algorithm': 'knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm',
}

# methods = {
#     'Algorithm': 'knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_2runfixpxlm',
#     'Algorithm_nomiddle_loss': 'knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_nouncatmiddle',
# }
file_path = 'moving_ratio_experiment_test_latest/images/MSE_SSIM_moving_ratio.pickle'
pdfsavepath = 'figures/MSE moving ratio comparision.pdf'

draw_uncertainty = False
col = 4 if draw_uncertainty else 2
fig, axs = plt.subplots(col,1,figsize=(10,7*col))

colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(methods)))
metric_keys = ['MSE','SSIM']
tot = -1
correlation = {}
for i, (method_name, method) in enumerate(methods.items()):
    cur_file_path = os.path.join(path, method, file_path)
    print('load', cur_file_path)
    MetaData = pickle.load(open(cur_file_path, 'rb'))
    Percentage = MetaData['kMA']

    ## Draw MSE and SSIM
    ls = '--' 
    color = colors[i]
    if i == len(methods) - 1: color = 'black'
    key_name = method_name.replace('_', '+')
    xval = Percentage
    for i, k in enumerate(metric_keys):
        axs[i].plot(xval[:tot], MetaData[k][:tot], color=color, label=key_name, linewidth=5, linestyle=ls)
    
    if draw_uncertainty:
        # Draw Uncertainty
        assert MetaData['mse_uncertainty_cor'].shape[2] == 2, f"{MetaData['mse_uncertainty_cor'].shape}"
        xval = MetaData['mse_uncertainty_cor'][:,:,1].mean(0).numpy()
        yval = MetaData['mse_uncertainty_cor'][:,:,0].mean(0).numpy()
        correlation[method_name] = [xval, yval]

for ax, label in zip(axs, metric_keys):
    set_plot(ax, ylabel=label, xlabel='kMA (%)')  

if draw_uncertainty:
    '''Uncertainty '''
    for i, k in enumerate(correlation.keys()):
        color = colors[i]
        ls = '--' 
        key_name = k.replace('_', '+')
        axs[2].plot(correlation[k][0], correlation[k][1], color=color, label=key_name, linewidth=5, linestyle=ls)

    rng = [min(correlation[k][0].min(), correlation[k][1].min()), max(correlation[k][0].max(), correlation[k][1].max())]
    axs[2].plot(np.arange(rng[0],rng[1],0.01), np.arange(rng[0],rng[1],0.01), '-r', label='Linear')
    set_plot(axs[2], ylabel='MSE', xlabel='Uncertainty')

    axs[3].plot(correlation['Algorithm'][1], correlation['Algorithm_nomiddle_loss'][1], color=color, label=key_name, linewidth=5, linestyle=ls)
    axs[3].plot(np.arange(0.02,0.1,0.01), np.arange(0.02,0.1,0.01), '-y')
    set_plot(axs[3], ylabel='Uncertainty (algorithm)', xlabel='Uncertainty (Algorithm_nomiddle_loss)')

fig.tight_layout()
fig.canvas.draw()
# Now we can save it to a numpy array.
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
imsave(pdfsavepath.replace('pdf','png'), data)
plt.savefig(pdfsavepath, bbox_inches='tight')

