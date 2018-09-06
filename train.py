import os
import numpy as np
import torch
import random
from torch.autograd import Variable, gradcheck
import torch.nn.functional as F
import model
import ShapeNetDataset
from utils import plot_3D_scene, plot_single_3D_scene, print_all_params, adjust_learning_rate, generateZ, generateLabel
from logger import Logger

def train(args):
    print(str(args).replace(',', '\n'))
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # create folder we need
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.save_dir + '/logs'):
        os.makedirs(args.save_dir + '/logs')
        os.makedirs(args.save_dir + '/logs/gan')
    if not os.path.exists(args.save_dir + '/models'):
        os.makedirs(args.save_dir + '/models')
        os.makedirs(args.save_dir + '/models/G')
        os.makedirs(args.save_dir + '/models/D')

    #===================== for using tensorboard=================#
    if args.use_tensorboard:
        if not os.path.exists(args.save_dir + '/logs/tensorboard'):
            os.makedirs(args.save_dir + '/logs/tensorboard')
        logger = Logger(args.save_dir + '/logs/tensorboard')

    # ===================== Data Loader=================#

    dset_shape = ShapeNetDataset.CustomDataset(args.dataset_dir)
    shape_loader = torch.utils.data.DataLoader(dset_shape, batch_size=args.batch_size_gan, shuffle=True)

    #======================create gan=====================#
    D = model._D(args)
    G = model._G(args)
    # D.apply(model.initialize_weights)
    # G.apply(model.initialize_weights)
    print(D)
    print(G)
    D = D.cuda()
    G = G.cuda()


    if args.optimizer_G == 'Adam':
        print('G Using Adam optimizer')
        G_solver = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=args.Adam_beta_G)
    elif args.optimizer_G == 'RMSprop':
        print('G Using RMSprop optimizer')
        G_solver = torch.optim.RMSprop(G.parameters(), lr=args.lr_G)
    else:
        print('G Using Sgd optimizer')
        G_solver = torch.optim.SGD(G.parameters(), lr=args.lr_G, weight_decay=0.0005)

    if args.optimizer_D == 'Adam':
        print('D Using Adam optimizer')
        D_solver = torch.optim.Adam(D.parameters(), lr=args.lr_D, betas=args.Adam_beta_D)
    elif args.optimizer_D == 'RMSprop':
        print('D Using RMSprop optimizer')
        D_solver = torch.optim.RMSprop(D.parameters(), lr=args.lr_D)
    else:
        print('D Using Sgd optimizer')
        D_solver = torch.optim.SGD(D.parameters(), lr=args.lr_D, weight_decay=0.0005)

    criterion = torch.nn.BCEWithLogitsLoss()

    #======================training=====================#
    ite = 1
    for epoch in range(args.gan_epochs):
        for i, labels in enumerate(shape_loader):
            labels = labels.view(-1, 1, 64, 64, 64)
            # ============= Train the discriminator =============# maximize log(D(x)) + log(1 - D(G(z)))
            labels = Variable(labels.type(torch.FloatTensor)).cuda()
            d_real, d_real_no_sigmoid = D(labels)
            d_real_loss = criterion(d_real_no_sigmoid, generateLabel(d_real,1, args))

            Z = Variable(generateZ(args)).cuda()
            fake = G(Z)
            d_fake, d_fake_no_sigmoid = D(fake.detach())
            d_fake_loss = criterion(d_fake_no_sigmoid , generateLabel(d_fake,0, args))

            d_loss = d_real_loss + d_fake_loss

            d_real_acc = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acc = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acc = torch.mean(torch.cat((d_real_acc, d_fake_acc),0))

            if (d_total_acc <= args.d_thresh).data.cpu().numpy():
                D_solver.zero_grad()
                d_loss.backward()
                D_solver.step()

            # ============= Train the generator =============# maximize log(D(G(z)))
            if (i) % args.iter_G == 0 :
                d_fake, d_fake_no_sigmoid = D(fake)
                g_loss = criterion(d_fake_no_sigmoid, generateLabel(d_fake,1, args))

                G_solver.zero_grad()
                g_loss.backward()
                G_solver.step()

            print('Iter-{}; , D_loss : {:.4}, G_loss : {:.4}, D_acc : {:.4}'.format(ite, d_loss.data[0], g_loss.data[0], d_total_acc.data[0]))

            #======================tensorboard========================#
            if args.use_tensorboard:

                info = {
                    'loss/loss_D_R': d_real_loss.data[0],
                    'loss/loss_D_F': d_fake_loss.data[0],
                    'loss/loss_D': d_loss.data[0],
                    'loss/loss_G': g_loss.data[0],
                    'loss/acc_D': d_total_acc.data[0]
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, ite)

                for tag, value in G.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary('Generator/' + tag, value.data.cpu().numpy(), ite)
                    logger.histo_summary('Generator/' + tag + '/grad', value.grad.data.cpu().numpy(), ite)

                for tag, value in D.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary('Discriminator/' + tag, value.data.cpu().numpy(), ite)
                    logger.histo_summary('Discriminator/' + tag + '/grad', value.grad.data.cpu().numpy(), ite)


            if  (i) % 10 == 0:
                Z = Variable(generateZ(args)).cuda()
                G.eval()
                data_gen = G(Z)
                G.train(True)
                data_gen = data_gen.data.cpu().numpy()
                data_gen = data_gen[0,:,:,:,:]
                data_gen = data_gen.__ge__(0.5)
                print(np.count_nonzero(data_gen))
                data_gen = np.squeeze(data_gen)
                log_img_name = args.save_dir + '/logs/gan/' + str(epoch).zfill(5) + '_' + str(i).zfill(5) + '.png'
                plot_single_3D_scene(data_gen, log_img_name)

            ite += 1

        if epoch % args.save_freq == 0:
            save_model_name_D = os.path.join(args.save_dir + '/models/D/', '%05d.ckpt' % (ite - 1))
            save_model_name_G = os.path.join(args.save_dir + '/models/G/', '%05d.ckpt' % (ite - 1))
            torch.save(D, save_model_name_D)
            torch.save(G, save_model_name_G)
        adjust_learning_rate(D_solver, epoch+1, args.lr_D, args.update_lr_epoch)
        adjust_learning_rate(G_solver, epoch+1, args.lr_G, args.update_lr_epoch)