import time
from options.train_options import TrainOptions
# from data import CreateDataLoader
from data import CreateFtTLoader
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    if opt.eval_full_valid:
        # use validation data
        val_data_loader = CreateFtTLoader(opt, is_test=True)
        train_data_loader, _ = CreateFtTLoader(opt, valid_size=0.001) # use all as training
    else:
        train_data_loader, val_data_loader = CreateFtTLoader(opt, valid_size=0.9 if not opt.debug else 0.99)

    dataset_size = len(train_data_loader)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt=opt)
    total_steps = 0
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += 1
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, i+1, dataset_size, t, losses, t_data)
                visualizer.plot_current_losses('train', total_steps, **losses)
                # visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        ''' perform epoch evaluation '''
        print('-> evaluating model {} ... '.format(opt.name))
        # TODO should we use evaluation mode? Not necessary
        if opt.eval_full_valid:
            visuals, losses = model.validation(val_data_loader, how_many_to_valid=float('inf'))
        elif opt.debug:
            visuals, losses = model.validation(val_data_loader, how_many_to_valid=4096)
        else:
            visuals, losses = model.validation(val_data_loader)

        visualizer.display_current_results(visuals, epoch, mode='eval')
        visualizer.plot_current_losses('eval', epoch, **losses)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
