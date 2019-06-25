from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4,
                            help='if positive, display all images in a single visdom web panel with certain number ' 
                                 'of images per row.')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=500,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, '
                                 '<epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true',
                            help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_vgg', type=float, default=0, help='perceptual loss weight')
        parser.add_argument('--no_cond_gan', action='store_true', help='do not use condition GAN')
        parser.add_argument('--consistency_loss', action='store_true', help='do not use condition GAN')
        parser.add_argument('--residual_loss', action='store_true', help='supervise the residual loss')
        parser.add_argument('--l2_weight', action='store_true', help='network l2 regularization')

        parser.add_argument('--dynamic_mask_type', type=str, choices=['None', 'random_zz', 'random_lowfreq'],
                            default='random_zz', help='Allows changing the mask during training')
        parser.add_argument('--debug', action='store_true', help='debug and use small training set')

        self.isTrain = True

        # ########################
        # New options
        # ########################
        parser.add_argument('--validation_train_split_ratio', type=float, default=0.9)
        parser.add_argument('--max_epochs', type=int, default=5,
                            help='number of epochs to train (default: 5)')
        parser.add_argument('--save_freq', type=int, default=200)

        # Options for Reconstruction Model
        parser.add_argument('--number_of_reconstructor_filters', type=int, default=128)
        parser.add_argument('--dropout_probability', type=float, default=0)
        parser.add_argument('--number_of_cascade_blocks', type=int, default=3)
        parser.add_argument('--number_of_layers_residual_bottleneck', type=int, default=6)
        parser.add_argument('--n_downsampling', type=int, default=3)
        parser.add_argument('--use_deconv', type=bool, default=True)

        # Options for Evaluator Model
        parser.add_argument('--number_of_evaluator_filters', type=int, default=256)
        parser.add_argument('--number_of_evaluator_convolution_layers', type=int, default=4)

        # Options for both Reconstructor and Evaluator Model
        parser.add_argument('--mask_embed_dim', type=int, default=6)
        # TODO : do we need it? We can set this based on dataroot
        parser.add_argument('--image_width', type=int, default=128)

        # Options moved from old model file
        parser.add_argument('--use_mse_as_disc_energy', action='store_true', help='use MSE as evaluator energy')
        parser.add_argument('--grad_ctx', action='store_true',
                            help='GAN criterion computes adversarial loss signal at provided k-space lines')
        parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for reconstruction loss')

        return parser
