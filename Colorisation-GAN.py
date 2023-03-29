#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:12:58 2023

@author: ruby
"""

# Reproducing colour images from greyscale images taken from CEERS NIRcam data.
# Network is based on a conditional GAN with the pix2pix example from the 
# image-to-image translation with conditional adversarial networks paper. 

''' Generator model takes grayscale (L-channel) image and predicts the other 2 channels
    (a channel and b channel). 
    Discriminator takes the a- and b-channel and concatenates them with the
    input greyscale channel and decides whether this produced image is real/fake.
    Discriminator also sees 'real' images, i.e one that are not produced by the
    generator '''

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader
import time

home = "/Users/ruby/Documents/JWST RGB Images/"

# seed for reproducible results- probably not needed
np.random.seed(123)
total_images = len(os.listdir(home+'RGB'))  # 841 images
# take and random sample from the colour images and use 80% of images for training
random_indices = random.sample(list(range(total_images)), total_images) 
train_nums = round(total_images * 0.8)
# split into training and testing
train_indices = random_indices[:train_nums]
test_indices = random_indices[train_nums:]
# len(train_indices) = 673, len(test_indices)=168


# Making datasets and dataloaders
SIZE = 256

# create the dataset to feed to the model
class ColorisationDataset(Dataset):
    ''' Resizes the images to 256x256 pixel size.
        This reads an RGBA image, converts it to RGB and then to Lab colour
        space to separate the L channel and ab channels as inputs and labels,
        respectively. '''
    def __init__(self, indices, path, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                                                  transforms.RandomHorizontalFlip()])
        elif split == 'test':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)
        
        self.split = split
        self.img_indices = indices
        self.rgb_path = path+'RGB/'
        self.size = SIZE
        self.path = path
    
    def __len__(self):
        return len(self.img_indices)
    
    def __getitem__(self, idx):
        img_name = str(idx)+'.jpg'
        img = Image.open(self.rgb_path+img_name).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # convert RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...]/ 50.0 - 1.0 # normalise between -1 and 1
        ab = img_lab[[1, 2], ...]/ 110.0 # between -1 and 1
        return {'L': L, 'ab': ab}
    
def MakeDataloaders(batch_size=16, pin_memory=False, **kwargs):
    ''' Function to create dataloaders for the model.
        Takes a dataset and returns a dataloader with the given batch size. '''
    dataset = ColorisationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
    return dataloader

# initialising the dataloaders for the train and test set
trainloader = MakeDataloaders(indices=train_indices, path=home, split='train')
testloader = MakeDataloaders(indices=test_indices, path=home, split='test')

# check the size of tensors L (expect 1 channel) and ab (expect 2 colour channels)
data = next(iter(trainloader))
Ls, abs_ = data['L'], data['ab']
#print(Ls.shape, abs_.shape)
# torch.Size([16,1,256,256]), torch.Size([16,2,256,256])
#print(len(trainloader), len(testloader))
# len(trainloader)=43, len(testloader)=11

# Generator as proposed by the paper
class UnetBlock(nn.Module):
    ''' U-Net is used as the generator of the GAN.
        Creates the U-Net from the middle part down and adds down-sampling and
        up-sampling modules to the left and right of the middle module.
        8 layers down so start with a 256x256 image, down-sample to a 1x1 image,
        then up-sample to a 256x256 image with 2 channels. '''
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        ''' ni = number of filters in the inner convolution layer
            nf = number of filters in the outer convolution layer
            input_c = number of input channels
            submodule = previously defined submodules
            dropout = not using dropout layers '''
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(in_channels=input_c, out_channels=ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost: # if this module is the outermost module
            upconv = nn.ConvTranspose2d(in_channels=ni*2, out_channels=nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: # if this module is the innermost module
            upconv = nn.ConvTranspose2d(in_channels=ni, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(in_channels=ni*2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else: # add skip connections
            return torch.cat([x, self.model(x)], dim=1)

class Unet(nn.Module):
    ''' U-Net based generator.'''
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        ''' input_c = number of input channels (L)
            output_c = number of output channels (ab)
            n_down = number of downsamples: we start with 256x256 and after 
                                            8 layers, we have a 1x1 image at the bottleneck.
            num_filters = number of filters in the last convolution layer. '''
        super().__init__()
        unet_block = UnetBlock(num_filters*8, num_filters*8, innermost=True)
        for _ in range(n_down - 5):
            # adds intermediate layers with num_filters * 8 filters
            unet_block = UnetBlock(num_filters*8, num_filters*8, submodule=unet_block, dropout=True)
        out_filters = num_filters*8
        for _ in range(3):
            # gradually reduce the number of filters to num_filters
            unet_block = UnetBlock(out_filters//2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)
    
# Now the discriminator as proposed by the paper 
class PatchDiscriminator(nn.Module):
    ''' Patch discriminator stacks blocks of convolution-batchnorm-leakyrelu 
        to decide whether the image is real or fake. 
        Patch discriminator outputs one number for every NxN pixels of the input
        and decides whether each "patch" is real/fake. 
        Patches will be 70 by 70. '''
    def __init__(self, input_c, num_filters=64, n_down=3):
        ''' input_c = number of input channels
            num_filters = number of filters in last5 convolution layer
            n_down = number of layers '''
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        # use if statement to take care of not using a stride of 2 in the last block of the loop
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i+1), s=1 if i == (n_down-1) else 2) for i in range(n_down)]
        # do not use normalisation or activation for the last layer of the model
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # ouput 1 channel prediction
        self.model = nn.Sequential(*model)
    
    # make a separate method for the repetitive layers
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        ''' norm = batch norm layer
        act = apply activation '''
        layers = [nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=k, stride=s, padding=p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
 
# output shape of patch discriminator
# discriminator = PatchDiscriminator(3)
# dummy_var = torch.randn(16,3,256,256)
# out = discriminator(dummy_var)
# out.shape    # torch.Szie([16,1,30,30])

# Unique loss function for the GAN 
class GANLoss(nn.Module):
    ''' Calculates the GAN loss of the final model.
        Uses a "vanilla" loss and registers constant tensors for the real
        and fake labels. Returns tensors full of zeros or ones to compute the loss'''
        
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer(name='real_label', tensor=torch.tensor(real_label))
        self.register_buffer(name='fake_label', tensor=torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss() # don't use this
        
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds) # expand to the same size as predictions
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    
# initilise the model here
def init_weights(net, init='norm', gain=.02):
    ''' Image-to-image translation paper state that the model is initialised 
        with a mean of 0.0 and std 0.02'''
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                # fills tensor with values drawn from normal distribution N(mean,std^2)
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier': # taken from a paper
            # fills input tensor with avlues sampled from N(0,std^2)
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming': # taken from a paper
                # resulting tensor has values sampled from N(0,std^2)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(tensor=m.bias.data, val=0.0) # tensor filled with zeros
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(tensor=m.bias.data, val=0.0)
    
    net.apply(init_func)
    print(f"model initialised with {init} initialisation")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

# now to initialise the main GAN network
class GANModel(nn.Module):
    ''' Initialises the model defining the generator and discriminator in the
        __init__ function using the functions given and initialises the loss
        functions '''
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=.5, beta2=.999, lambda_L1=100.): 
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G.to(self.device)
        
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GAN_loss = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1_loss = nn.L1Loss()
        # initialise optimisers for generator and discriminator using Adam optimiser
        # and parameters stated in the paper 
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1,beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1,beta2))
        # keep track of losses
        self.generator_losses, self.discriminator_losses = [], []
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        # Get the input data and labels
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
    
    def forward(self):
        # For each batch in the training set, forward method is called and
        # outputs stored in fake_color variable
        self.fake_color = self.net_G(self.L)
        
    def backward_D(self):
        ''' Discriminator loss takes both target and input images.
            loss_D_real is sigmoid cross-entropy loss of the target images and an array
            of ones. 
            loss_D_fake is sigmoid cross-entropy loss of the input images and an
            array of zeros.
            Discriminator loss is loss_D = loss_D_real + loss_D_fake. '''
        # Train the discriminator by feeding the fake images produced by the 
        # generator 
        fake_img = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_img.detach()) # detach from generator's graph so they act like constants
        # label the fake images as fake 
        self.loss_D_fake = self.GAN_loss(preds=fake_preds, target_is_real=False)
        # Now feed a batch of real images from the training set and label them as real
        real_img = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_img)
        self.loss_D_real = self.GAN_loss(preds=real_preds, target_is_real=True)
        # Add the two losses for fake and real, take the average and cal backward()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * .5
        self.loss_D.backward()
        self.discriminator_losses += [self.loss_D.item()]
    
    def backward_G(self):
        ''' Generator loss is a sigmoid cross-entropy of input images and an 
            array of ones. Using the L1 loss, input images are structurally
            similar to the target images.
            Generator loss is defined as loss_G = loss_G_GAN + loss_G_L1*lambda_L1. '''
        # Train the generator by feeding the discriminator the fake image and 
        # fool it by assigning real labels and calculating adversarial loss.
        fake_img = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_img)
        self.loss_G_GAN = self.GAN_loss(preds=fake_preds, target_is_real=True)
        # Use L1 loss so images are not blurry or averaged over and compute the 
        # difference between the predicted channels and real channels and multiply 
        # by constant lambda 
        self.loss_G_L1 = self.L1_loss(self.fake_color, self.ab) * self.lambda_L1
        # Add L1 loss to the adversarial loss then call backward()
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.generator_losses += [self.loss_G_GAN.item()]
        
    def optimise(self):
        # Now optimise by the usual method of zeroing the gradients and calling
        # step() on the optimiser
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

# function to log the losses and visualise the outputs from the network
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count
        
def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)
    return loss_meter #might need to remove
     
def lab_to_rgb(L, ab):
    # takes a batch of images
    L = (L + 1.) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def loss_plot(model, save=True):
    gen_loss = model.generator_losses
    dis_loss = model.discriminator_losses
    fig = plt.figure(figsize=(12,6))
    plt.plot(gen_loss, label='Generator Loss', marker='o', color='red')
    plt.plot(dis_loss, label='Discriminator Loss', marker='x', color='blue', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    if save:
        fig.savefig(f"loss_{time.time()}.png")
    

def visualise(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15,8))
    for i in range(5):
        ax = plt.subplot(3, 5, i+1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.set_title("Input Image")
        ax.axis("off")
        ax = plt.subplot(3, 5, i+1+5)
        ax.imshow(fake_imgs[i])
        ax.set_title("Generated Image")
        ax.axis("off")
        ax = plt.subplot(3, 5, i+1+10)
        ax.imshow(real_imgs[i])
        ax.set_title("Real Image")
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorisation_{time.time()}.png")

        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

# now train the network, display epochs and losses
def train_model(model, trainloader, epochs, display_every=30):
    print("Starting training....")
    start = time.time()
    data = next(iter(trainloader)) # batch for visualising the model output after fixed intervals after training
    for e in range(epochs):
        # function returning a dictionary of objects to log the losses of the complete network
        loss_meter_dict = create_loss_meters() 
        i = 0
        for data in tqdm(trainloader):
            model.setup_input(data)
            model.optimise()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # updates the log objects
            i += 1
        print(f"\nEpoch {e+1}/{epochs}")
        if i % display_every == 0: 
            #print(f"\nEpoch {e+1}/{epochs}")
            print(f"Iteration {i}/{len(trainloader)}")
        total_loss = log_results(loss_meter_dict) # function prints out the losses
        #visualise(model, data, save=False) # displays the model's outputs
        print(total_loss)
    visualise(model, data, save=False)
    loss_plot(model, save=True)
    endtime = time.time()
    end = endtime - start
    print("Time to train network: {:.2f}s".format(end))

             
# initialise the network
model = GANModel()
train_model(model, trainloader, epochs=1) 
     
# =============================================================================
# Interpreting the losses:
#    > If loss_G_GAN or loss_D get very low, then either model is dominating the other
#    and the combined model is not successfully training.
#    > The value log(2)=0.69 is a good reference for loss_G_GAN and loss_D.
#    This indicates that the discriminator is equally uncertain about the input
#    image and target image.
#    > A value below 0.69 for loss_D suggests the discriminator is doing better than
#    random on the set of real and input images.
#    > A value below 0.69 for loss_G_GAN suggests that the generator is doing better
#    than random at fooling the discriminator.
#    > loss_G_L1 should go down during training.
# =============================================================================
       
                
        
        
            
            
   






























       
            
        



































        
        
                
