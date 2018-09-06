import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.misc import imread


def generateLabel(tensor, label, args):
  if args.soft_label:
      if label == 1:
          out = torch.Tensor(tensor.size()).uniform_(0.7, 1.2)
      else:
          out = torch.Tensor(tensor.size()).uniform_(0, 0.3)
      out = Variable(out).cuda()

  else:
      if label == 1:
          out = torch.ones_like(tensor)
      else:
          out = torch.zeros_like(tensor)
      out = out.cuda()

  return out


def generateZ(args):

    if args.z_dis == "norm":
        Z = torch.randn(args.batch_size_gan, args.z_dim, args.z_start_vox[0], args.z_start_vox[1], args.z_start_vox[2]).normal_(0, 0.33)
    elif args.z_dis == "uni":
        Z = torch.rand(args.batch_size_gan, args.z_dim, args.z_start_vox[0], args.z_start_vox[1], args.z_start_vox[2])
    else:
        print("z_dist is not normal or uniform")
    Z = Z.type(torch.FloatTensor)

    return Z


def adjust_learning_rate(optimizer, epoch, init_lr, update_lr_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every udpate_lr epochs"""
    lr = init_lr * (0.1 ** (epoch // update_lr_epoch))
    print('Set new lr = ' + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_3D_scene(data_pred, data_label, log_img_name, src_img_name, free_voxels):
    x, y, z = np.meshgrid(np.arange(data_pred.shape[0]), np.arange(data_pred.shape[1]), np.arange(data_pred.shape[2]))
    [x, y, z] = (np.reshape(x, (-1)), np.reshape(y, (-1)), np.reshape(z, (-1)))

    fig = plt.figure(figsize=(25, 12))

    idx = (data_pred != 0) & (data_pred != 255) & (free_voxels != 1)  # TODO: Shouldn't this be data_label?
    idx = np.reshape(idx, (-1))
    ax = fig.add_subplot(131, projection='3d')
    scat = ax.scatter(x[idx], y[idx], z[idx], c=np.reshape(data_pred, (-1))[idx], cmap='jet', marker="s")
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title('pred')
    ax.axis('equal')
    az = 150
    el = 30
    ax.view_init(elev=el, azim=az)

    idx = (data_label != 0) & (data_label != 255)
    idx = np.reshape(idx, (-1))
    ax = fig.add_subplot(132, projection='3d')
    scat = ax.scatter(x[idx], y[idx], z[idx], c=np.reshape(data_label, (-1))[idx], cmap='jet', marker="s")
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title('gt')
    ax.axis('equal')
    az = 150
    el = 30
    ax.view_init(elev=el, azim=az)
    fig.colorbar(scat, shrink=0.5, aspect=5)

    if src_img_name is not None:
        ax = fig.add_subplot(133)
        im = imread(src_img_name)
        ax.imshow(im)
        plt.axis('off')

    plt.savefig(log_img_name, bbox_inches='tight')
    plt.close('all')

def plot_single_3D_scene(data_pred, log_img_name):
    x, y, z = np.meshgrid(np.arange(data_pred.shape[0]), np.arange(data_pred.shape[1]), np.arange(data_pred.shape[2]))
    [x, y, z] = (np.reshape(x, (-1)), np.reshape(y, (-1)), np.reshape(z, (-1)))

    fig = plt.figure(figsize=(25, 12))

    idx = (data_pred != 0) & (data_pred != 255)
    idx = np.reshape(idx, (-1))
    ax = fig.add_subplot(131, projection='3d')
    scat = ax.scatter(x[idx], y[idx], z[idx], c=np.reshape(data_pred, (-1))[idx], cmap='jet', marker="s")
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title('pred')
    ax.axis('equal')
    az = 150
    el = 30
    ax.view_init(elev=el, azim=az)

    plt.savefig(log_img_name, bbox_inches='tight')
    plt.close('all')


def print_all_params(num_epochs, update_lr_iter, iter_size, save_freq, vis_freq,
        save_dir, batch_size, learning_rate,
        resume, model_file, optimizer_mode,
        sampling_mode, fixed_weights, filelist_name,
        data_root, image_dir, data_info, num_train_files):

        print(' num_epochs = ' + str(num_epochs))
        print(' update_lr_iter = ' + str(update_lr_iter))
        print(' iter_size = ' + str(iter_size))
        print(' save_freq = ' + str(save_freq))
        print(' vis_freq = ' + str(vis_freq))
        print(' save_dir = ' + str(save_dir))
        print(' batch_size = ' + str(batch_size))
        print(' learning_rate = ' + str(learning_rate))
        print(' resume = ' + str(resume))
        print(' model_file = ' + str(model_file))
        print(' optimizer_mode = ' + str(optimizer_mode))
        print(' sampling_mode = ' + str(sampling_mode))
        print(' fixed_weights = ' + str(fixed_weights))
        print(' filelist_name = ' + str(filelist_name))
        print(' data_root = ' + str(data_root))
        print(' image_dir = ' + str(image_dir))
        # print(' data_info = ' + str(data_info))
        print(' num_train_files = ' + str(num_train_files))
