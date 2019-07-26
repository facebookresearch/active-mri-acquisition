from options.train_options import TrainOptions
from data import create_data_loaders

import torch

def test_symmetry(options):
    train_loader, val_loader = create_data_loaders(options)
    diff = torch.zeros(184)
    num_images = 1000

    for idx, (mask, ground_truth, kspace) in enumerate(train_loader):
        print(idx)
        if idx == num_images:
            break
        high = int(kspace.shape[2]/2)
        low = high - 1
        while low >= 0 and high <= kspace.shape[1]:
            # print(low, high)
            diff[low] += torch.norm((kspace[:,:,low,:] - kspace[:,:,high,:]), p=1)/torch.norm(kspace[:,:,low,:], p=1)
            low -= 1
            high += 1

    diff = diff/num_images
    print(diff)

    torch.save(diff, '/private/home/sumanab/tests/normalized_symmetry.pt')
    # torch.save(mask, '/private/home/sumanab/tests/mask.pt')
    # torch.save(ground_truth, '/private/home/sumanab/tests/ground_truth.pt')
    # torch.save(kspace, '/private/home/sumanab/tests/kspace.pt')


if __name__ == '__main__':
    options = TrainOptions().parse()  # TODO: need to clean up options list
    options.device = torch.device('cuda:{}'.format(
        options.gpu_ids[0])) if options.gpu_ids else torch.device('cpu')

    test_symmetry(options)