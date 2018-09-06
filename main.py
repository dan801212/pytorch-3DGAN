import argparse
from train import train
from test import test


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else: 
        raise('Please choose a correct mode')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="3DGAN")
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'])
    
    #GAN params
    parser.add_argument("--gan-epochs", type=int, default=1500)
    parser.add_argument("--lr-G", type=float, default=0.0025)
    parser.add_argument("--lr-D", type=float, default=0.001)
    parser.add_argument("--optimizer-G", type=str, default='Adam', choices=['Adam', 'Sgd', 'RMSprop'],
                        help='choose between Sgd/ Adam')
    parser.add_argument("--optimizer-D", type=str, default='Adam', choices=['Adam', 'Sgd', 'RMSprop'],
                        help='choose between Sgd/ Adam')
    parser.add_argument("--Adam-beta-G", type=float, default=(0.5,0.99))
    parser.add_argument("--Adam-beta-D", type=float, default=(0.5,0.99))
    parser.add_argument("--z-dim", type=int, default=200)
    parser.add_argument("--z-start-vox", type=int, nargs='*', default=[1,1,1])
    parser.add_argument("--z-dis", type=str, default='norm', choices=['norm', 'uni'],
                        help='uniform: uni, normal: norm')
    parser.add_argument("--batch-size-gan", type=int, default=32)
    parser.add_argument('--d-thresh', type=float, default=0.8,
                        help='for balance discriminator and generator')
    parser.add_argument('--leak-value', type=float, default=0.2,
                        help='leaky relu')
    parser.add_argument('--soft-label', type=bool, default=True,
                        help='using soft_label')

    #step params
    parser.add_argument("--iter-G", type=int, default=1,
                        help='train G every n iteration')
    parser.add_argument("--save-freq", type=int, default=10,
                        help= 'Save model every n epoch')
    parser.add_argument("--vis-freq", type=int, default=100,
                        help= 'visualize model every n epoch')
    parser.add_argument("--update-lr-epoch", type=int, default=500,
                        help= 'lr decay by 10 every n epoch')

    # dir params
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--dataset-dir", type=str)

    # other params
    parser.add_argument('--use-tensorboard', type=bool, default=True,
                       help='using tensorboard to visualize')
    parser.add_argument('--manualSeed', type=int, default=0,
                        help='manual seed')

    args = parser.parse_args()
    main(args)
