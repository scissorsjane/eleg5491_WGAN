from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

import models.dcgan as dcgan
import models.mlp as mlp

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

'''
print('plot loss figure')
plt.figure(figsize=(16,8),dpi=98)
x = np.arange(0,5)
y1=[1,2,3,4,5]
y2=[4,3,2,5,7]
y3=[5,1,3,1,5]
y4=[2,2,4,6,4]
plt.subplot(221)
plt.xlabel('iteration')
plt.ylabel('Loss_D')
plt.plot(x,y1,"g",label='errD',linewidth=2)
plt.subplot(222)
plt.plot(x,y2,"b",label='errG',linewidth=2)
plt.xlabel('iteration')
plt.ylabel('Loss_G')
p3 = plt.subplot(223)
plt.plot(x,y3,"r-",label='errD_real',linewidth=2)
p4 = plt.subplot(224)
plt.plot(x,y4,"g-.",label='errD_fake',linewidth=2)
#y = np.random.randn(60)

p1.set_xlabel('iteration')
p1.set_ylabel('Loss_D')
p2.set_xlabel('iteration')
p2.set_ylabel('Loss_G')
p3.set_xlabel('iteration')
p3.set_ylabel('Loss_D_real')
p4.set_xlabel('iteration')
p4.set_ylabel('Loss_D_fake')
#plt.scatter(x, y, s=20)
p1=plt.plot(x,y1,"g",label='errD',linewidth=2)
p2=plt.plot(x,y2,"b",label='errG',linewidth=2)
p3=plt.plot(x,y3,"r-",label='errD_real',linewidth=2)
p4=plt.plot(x,y4,"g-.",label='errD_fake',linewidth=2)

out_png =  'loss.png'
plt.savefig(out_png, dpi=150)
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--WGAN', action='store_true', help='use Wassertein GAN or not')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )

elif opt.dataset == 'celeba':
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.WGAN:
    if opt.noBN:
        netG = dcgan.WGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    else:
        netG = dcgan.WGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

#elif opt.mlp_G:
#    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

#if opt.mlp_D:
#    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
if opt.WGAN:
   
    netD = dcgan.WGAN_D(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)


else:
   
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)


    
netD.apply(weights_init) ##???????

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


#loss for DCGAN:

criterion=nn.BCELoss()


input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    if opt.WGAN:
        one, mone = one.cuda(), mone.cuda()
    else:
        criterion.cuda()
    input = input.cuda()
    label=label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0
errD_list=[]
errG_list=[]
errD_real_list=[]
errD_fake_list=[]
errD_w_list=[]
W_estimate=[]
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    #print('here1')
    i = 0
    while i < len(dataloader):

        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 25
            #Diters = opt.Diters
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1


            # clamp parameters to a cube for WGAN
            if opt.WGAN:
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = data_iter.next()
            #print ('here2')
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad()
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
              
            
                        
            if opt.WGAN:
                #print ('here3')
                input.resize_as_(real_cpu).copy_(real_cpu)  ###???
                inputv = Variable(input)
                errD_real = netD(inputv)
                errD_real.backward(one)
            else:
                inputv = Variable(input)
                inputv.data.resize_(real_cpu.size()).copy_(real_cpu)
                labelv = Variable(label)
                labelv.data.resize_(batch_size).fill_(real_label)
                output = netD(inputv)
                #print('output')
                #print (output.data[0])
                errD_real_w= output.mean(0)
                errD_real_w=errD_real_w.view(1)
                errD_real = criterion(output, labelv)
                #print (errD_real.data[0])
                errD_real.backward()
                D_x = output.data.mean()

            # train with fake
            if opt.WGAN:
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
            else:
                noisev = Variable(noise)
                fixed_noisev = Variable(fixed_noise)
                noisev.data.resize_(batch_size, nz, 1, 1)
                noisev.data.normal_(0, 1)
                fake = netG(noisev)
                labelv.data.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, labelv)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                errD_fake_w=output.mean(0)
                errD_fake_w=errD_fake_w.view(1)
                errD_w = errD_real_w - errD_fake_w
                
            optimizerD.step()
        #print('here2')
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise

        if opt.WGAN:
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()

        else:
            labelv.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()   #print  output????????
            optimizerG.step()

        gen_iterations += 1
        errD_list.append(errD.data[0])
        errG_list.append(errG.data[0])
        errD_real_list.append(errD_real.data[0])
        errD_fake_list.append(errD_fake.data[0])
        #if not opt.WGAN:
        #    errD_w_list.append(errD_w.data[0])
        if opt.WGAN:
            W_estimate.append(-errD.data[0])
        else:
            W_estimate.append(-errD_w.data[0])
    

        if opt.WGAN:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        else:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            '''print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))'''
        if gen_iterations % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            if opt.WGAN:
                fake = netG(Variable(fixed_noise, volatile=True))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
            else:
                fake = netG(fixed_noisev)
                fake.data = fake.data.mul(0.5).add(0.5)
                print (fake.data)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
            print('plot training curve')
            plt.figure(figsize=(16,12),dpi=98)
            x = np.arange(0,gen_iterations)
            plt.subplot(311)
            plt.xlabel('iteration')
            plt.ylabel('Loss_D')
            plt.plot(x,errD_list,"g",label='errD',linewidth=2)
            plt.subplot(312)
            plt.plot(x,errG_list,"b",label='errG',linewidth=2)
            plt.xlabel('iteration')
            plt.ylabel('Loss_G')
            plt.subplot(313)
            plt.plot(x,W_estimate,"r",label='Wassertein Estimate',linewidth=2)
            plt.xlabel('iteration')
            plt.ylabel('Wassertein Estimate')
            #y = np.random.randn(60)
    
            out_png =  '{0}/loss.png'.format(opt.experiment)
            plt.savefig(out_png, dpi=150)

    # do checkpointing
    #torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    #torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

"""
print('plot training curve')
plt.figure(figsize=(16,12),dpi=98)
x = np.arange(0,gen_iterations)
plt.subplot(311)
plt.xlabel('iteration')
plt.ylabel('Loss_D')
plt.plot(x,errD_list,"g",label='errD',linewidth=2)
plt.subplot(312)
plt.plot(x,errG_list,"b",label='errG',linewidth=2)
plt.xlabel('iteration')
plt.ylabel('Loss_G')
'''
plt.subplot(223)
plt.xlabel('iteration')
plt.ylabel('Loss_D_real')
plt.plot(x,errD_real_list,"r-",label='errD_real',linewidth=2)
plt.subplot(224)
plt.xlabel('iteration')
plt.ylabel('Loss_D_fake')
plt.plot(x,errD_fake_list,"g-.",label='errD_fake',linewidth=2)
'''
W_estimate=[]
j=0
while j < gen_iterations:
    if opt.WGAN:
        W_estimate.append(-errD_list[j])
    else:
        W_estimate.append(-errD_w_list[j])
    j=j+1

plt.subplot(313)
plt.plot(x,W_estimate,"r",label='Wassertein Estimate',linewidth=2)
plt.xlabel('iteration')
plt.ylabel('Wassertein Estimate')
#y = np.random.randn(60)

out_png =  '{0}/loss.png'.format(opt.experiment)
plt.savefig(out_png, dpi=150)

"""
