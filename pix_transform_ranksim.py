import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys
import h5py
import os
import matplotlib.pyplot as plt
import time
from baselines.baselines import bicubic
from utils.utils import downsample,align_images
from utils.plots import plot_result
import argparse
import random
import torch.nn.functional as F
import copy
import torch.nn as nn
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
# Follows the functions presented at https://github.com/BorealisAI/ranksim-imbalanced-regression/tree/main
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))

def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]

class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None

def batchwise_ranking_regularizer(features, targets, lambda_val):
    loss = 0
    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)
    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        x = features
        y = targets

    # Compute feature similarities
    xxt = torch.matmul(F.normalize(x.view(x.size(0),-1)), F.normalize(x.view(x.size(0),-1)).permute(1,0))

    # Compute ranking similarity loss
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0,1))
        feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), lambda_val)
        loss += F.mse_loss(feature_ranks, label_ranks)
    
    return loss
# Follows the functions presented at https://github.com/tatp22/multidim-positional-encoding
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

def extract_sub_patches(data, patch_size = 16):
    # Extract non-overlapping 16x16 images with a stride of 16
    
    num_samples, num_channels, original_height, original_width = data.shape

    # Calculate the number of patches along height and width
    num_patches_height = original_height // patch_size
    num_patches_width = original_width // patch_size

    # Initialize an empty list to store the extracted 16x16 images
    extracted_images = []

    # Iterate through each sample and extract non-overlapping 16x16 images
    for i in range(num_samples):
        for h in range(num_patches_height):
            for w in range(num_patches_width):
                extracted_image = data[i, :, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]
                extracted_images.append(extracted_image)

    # Convert the list of extracted images to a numpy array
    extracted_images = np.array(extracted_images)

    # The extracted_images array will have a shape of (num_samples * num_patches, num_channels, 16, 16)
    return extracted_images

def positional_encode(tensor):
    print(tensor.shape)
    assert tensor.shape[-1] == 3
    p_enc_2d_model = PositionalEncoding2D(3)
    x = tensor.clone()
    penc_no_sum = p_enc_2d_model(x)
    return torch.cat([x,penc_no_sum], dim = -1)

class PixTransformNetPositionalEncoding(nn.Module):

    def __init__(self, channels_in=5, kernel_size = 1,weights_regularizer = None):
        super(PixTransformNetPositionalEncoding, self).__init__()

        self.channels_in = channels_in
        self.spatial_net = nn.Sequential(nn.Conv2d(3,32,(1,1),padding=0),
                                         nn.ReLU(),nn.Conv2d(32,2048,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        self.color_net = nn.Sequential(nn.Conv2d(3,32,(1,1),padding=0),
                                       nn.ReLU(),nn.Conv2d(32,2048,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        self.head_net = nn.Sequential(nn.ReLU(),nn.Conv2d(2048, 32, (kernel_size,kernel_size),padding=(kernel_size-1)//2),
                                      nn.ReLU(),nn.Conv2d(32, 1, (1, 1),padding=0))

        if weights_regularizer is None:
            reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]
        
        self.params_with_regularizer = []
        self.params_with_regularizer += [{'params':self.spatial_net.parameters(),'weight_decay':reg_spatial}]
        self.params_with_regularizer += [{'params':self.color_net.parameters(),'weight_decay':reg_color}]
        self.params_with_regularizer += [{'params':self.head_net.parameters(),'weight_decay':reg_head}]


    def forward(self, input):

        input_spatial = input[:,self.channels_in-3:,:,:]
        input_color = input[:,0:self.channels_in-3,:,:]
        #input_spatial = input[:,self.channels_in-2:,:,:]
        #input_color = input[:,0:self.channels_in-2,:,:]

        merged_features = self.spatial_net(input_spatial) + self.color_net(input_color)
        
        return self.head_net(merged_features), merged_features
    
class PixTransformNet(nn.Module):

    def __init__(self, channels_in=5, kernel_size = 1,weights_regularizer = None):
        super(PixTransformNet, self).__init__()

        self.channels_in = channels_in
        self.spatial_net = nn.Sequential(nn.Conv2d(2,32,(1,1),padding=0),
                                         nn.ReLU(),nn.Conv2d(32,2048,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        self.color_net = nn.Sequential(nn.Conv2d(channels_in-2,32,(1,1),padding=0),
                                       nn.ReLU(),nn.Conv2d(32,2048,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        self.head_net = nn.Sequential(nn.ReLU(),nn.Conv2d(2048, 32, (kernel_size,kernel_size),padding=(kernel_size-1)//2),
                                      nn.ReLU(),nn.Conv2d(32, 1, (1, 1),padding=0))

        if weights_regularizer is None:
            reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]
        
        self.params_with_regularizer = []
        self.params_with_regularizer += [{'params':self.spatial_net.parameters(),'weight_decay':reg_spatial}]
        self.params_with_regularizer += [{'params':self.color_net.parameters(),'weight_decay':reg_color}]
        self.params_with_regularizer += [{'params':self.head_net.parameters(),'weight_decay':reg_head}]


    def forward(self, input):

        input_spatial = input[:,self.channels_in-2:,:,:]
        input_color = input[:,0:self.channels_in-2,:,:]

        merged_features = self.spatial_net(input_spatial) + self.color_net(input_color)
        
        return self.head_net(merged_features), merged_features

def bin_image_to_classes(image, num_classes,device):
    
    image = image.to(device)
    # Calculate bin edges
    bin_edges = torch.linspace(image.min().item(), image.max().item(), num_classes).to(device)
    
    # Apply bucketize to get class indices
    class_indices = torch.bucketize(image, bin_edges) #- 1
    
    return class_indices.to(device)



# Define the parser
parser = argparse.ArgumentParser(description="PIXTRANSFORM")
parser.add_argument("--data_dir", type=str, default="/home/tsoyda/data/super_VHM/4d_sentinel_dropna_gee.h5", help="Path to the data file (HDF5 format)")
parser.add_argument("--ranksim_weight", type=float, default=0.3, help="Ranksim Weight")
parser.add_argument("--positional_encoding", type=int, default=0, help="Positional Encoding")
parser.add_argument("--index_s", type=int, default=0, help="starting index of samples to predict")
parser.add_argument("--index_f", type=int, default=1, help="finishing index of samples to predict")
parser.add_argument("--filename", type=str, default="predictions", help="desired file name")
parser.add_argument("--ranksim_target", type = str, default = "z_mean", help = "Target of the RankSim") 
parser.add_argument("--lr", type = float, default = 0.001, help = "Learning Rate") 
parser.add_argument("--batchsize", type = int, default = 32, help = "Batchsize") 
parser.add_argument("--scaling", type = int, default = 4, help = "Scaling Factor") 



def pixtransform(data_dir = "/home/tsoyda/data/super_VHM/4d_sentinel_dropna_gee.h5", 
                 positional_encoding = 0, ranksim_weight = 0, indexes = [0,1],
                 lr = 0.001, batchsize = 32, scaling = 4, ranksim_target = "source"):

    dataset = h5py.File(data_dir) 
    target_imgs = np.array(dataset["eval"]).squeeze()
    guide_imgs =  np.array(dataset["guide"]).squeeze()
    source_imgs = np.array(dataset["source"]).squeeze()
    try:
        segmentation_imgs = np.array(dataset["gee"]).squeeze()
    except:
        segmentation_imgs = np.array(dataset["segmentation"]).squeeze()        
    dataset.close()

    params = {
            'img_idxs' : indexes,
            'scaling': scaling, #8,
            'greyscale': False, # Turn image into grey-scale
            'channels': -1,

            'spatial_features_input': True,
            "positional_encode": positional_encoding,
            'weights_regularizer': [0.0001, 0.001, 0.0001], # spatial color head
            'loss': 'l1',

            'optim': 'adam',
            'lr': lr,

            "early_stopping":True,
            "early_stopping_patience":200,
                    
            'batch_size':batchsize,
            'iteration': 10000, 
            "ranksim_interpolation_lambda" : 2,
            "ranksim_weight" : ranksim_weight,
            'logstep': 64,

            'final_TGV' : False, # Total Generalized Variation in post-processing
            'align': False, # Move image around for evaluation in case guide image and target image are not perfectly aligned
            'delta_PBP': 1, # Delta for percentage of bad pixels 
            }
    
    if params["positional_encode"]:
        params["spatial_features_input"] = False

    ##
    # Sorting GEE segmentation labels by mean depth
    converter = dict(zip(list(range(9)), [0, 4, 6, 7, 2, 3, 8, 5, 1]))
    def map_values(value):
        return converter.get(value, value)
    mapped_arr = np.vectorize(map_values)(segmentation_imgs)
    ##
    predictions = []
    ##

    for idx in indexes: #number of samples to train-predict
        guide_img = guide_imgs[idx] 
        target_img = target_imgs[idx]
        original_source_img = source_imgs[idx]
        source_img = downsample(original_source_img,params['scaling']) ## downsampled source image
        segmentation_img = mapped_arr[idx] #segmentation_imgs[idx] ## GEE segmentation image with sorted label
        bicubic_target_img = bicubic(source_img=source_img, scaling_factor=params['scaling'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(guide_img.shape) < 3:
            guide_img = np.expand_dims(guide_img, 0)

        if params["channels"] > 0:
            guide_img = guide_img[0:params["channels"], :, :]

        if params['greyscale']:
            guide_img = np.mean(guide_img, axis=0, keepdims=True)

        n_channels, hr_height, hr_width = guide_img.shape

        source_img = source_img.squeeze()
        lr_height, lr_width = source_img.shape

        assert (hr_height == hr_width)
        assert (lr_height == lr_width)
        assert (hr_height % lr_height == 0)

        D = hr_height // lr_height
        M = lr_height
        N = hr_height

        # normalize guide and source
        guide_img = (guide_img - np.mean(guide_img, axis=(1, 2), keepdims=True)) / np.std(guide_img, axis=(1, 2),
                                                                                            keepdims=True)
        source_img_mean = np.mean(source_img)
        source_img_std = np.std(source_img)
        source_img = (source_img - source_img_mean) / source_img_std
        if target_img is not None:
            target_img = (target_img - source_img_mean) / source_img_std

        if params['spatial_features_input']:
            x = np.linspace(-0.5, 0.5, hr_width)
            x_grid, y_grid = np.meshgrid(x, x, indexing='ij')

            x_grid = np.expand_dims(x_grid, axis=0)
            y_grid = np.expand_dims(y_grid, axis=0)

            guide_img = np.concatenate([guide_img, x_grid, y_grid], axis=0)
        
        #### prepare_patches #########################################################################
        # guide_patches is M^2 x C x D x D
        # source_pixels is M^2 x 1

        guide_img = torch.from_numpy(guide_img).float().to(device)
        source_img = torch.from_numpy(source_img).float().to(device)
        original_source_img = torch.from_numpy(original_source_img).float().to(device)

        if params["positional_encode"]:
            new_guide = guide_img.reshape(1,-1,hr_height, hr_height).permute(0,2,3,1)
            guide_img = positional_encode(new_guide).permute(0,3,1,2).squeeze()
            #print("Guide image positional encoded with shape = ",guide_img.shape)

        if target_img is not None:
            target_img = torch.from_numpy(target_img).float().to(device)
        segmentation_img = torch.from_numpy(segmentation_img).float().to(device)

        guide_patches = torch.zeros((M * M, guide_img.shape[0], D, D)).to(device)
        source_pixels = torch.zeros((M * M, 1)).to(device)
        segmentation_patches = torch.zeros((M * M, 1, D, D)).to(device)

        num_bins = 9 # number of bins for the target binning
        binned_target = bin_image_to_classes(target_img, num_bins, device) # binned target
        binned_target = binned_target.to(device)

        rich_segmentation_img = (segmentation_img + 1) * (original_source_img / original_source_img.max())

        for i in range(0, M):
            for j in range(0, M):
                guide_patches[j + i * M, :, :, :] = guide_img[:, i * D:(i + 1) * D, j * D:(j + 1) * D]
                source_pixels[j + i * M] = source_img[i:(i + 1), j:(j + 1)]
                if ranksim_target == "source":
                    segmentation_patches[j + i * M, :, :, :] = original_source_img.unsqueeze(0)[:, i * D:(i + 1) * D, j * D:(j + 1) * D]
                elif ranksim_target == "binned_target":
                    segmentation_patches[j + i * M, :, :, :] = binned_target.unsqueeze(0)[:, i * D:(i + 1) * D, j * D:(j + 1) * D]
                elif ranksim_target == "segmentation":
                    segmentation_patches[j + i * M, :, :, :] = segmentation_img.unsqueeze(0)[:, i * D:(i + 1) * D, j * D:(j + 1) * D]
                elif ranksim_target == "rich_segmentation":
                    segmentation_patches[j + i * M, :, :, :] = rich_segmentation_img.unsqueeze(0)[:, i * D:(i + 1) * D, j * D:(j + 1) * D]

        train_data = torch.utils.data.TensorDataset(guide_patches, source_pixels, segmentation_patches)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        ###############################################################################################

        #### setup network ############################################################################
        if params["positional_encode"]:
            mynet = PixTransformNetPositionalEncoding(channels_in=guide_img.shape[0],
                                    weights_regularizer=params['weights_regularizer']).train().to(device)
        else:
            mynet = PixTransformNet(channels_in=guide_img.shape[0],
                                    weights_regularizer=params['weights_regularizer']).train().to(device)

        optimizer = optim.Adam(mynet.params_with_regularizer, lr=params['lr'])
        if params['loss'] == 'mse':
            myloss = torch.nn.MSELoss()
        elif params['loss'] == 'l1':
            myloss = torch.nn.L1Loss()
        else:
            print("unknown loss!")
        ###############################################################################################


        avgpool = torch.nn.AvgPool2d(kernel_size = params["scaling"])

        ################################################################################################
        losses = []
        epochs = params["batch_size"] * params["iteration"] // (M * M)

        best_loss = 9999
        best_model_state_dict = None
        no_improvement_count = 0
        early_stopped = False

        
        for epoch in range(epochs):
            for (x, y, z) in train_loader:
                optimizer.zero_grad()

                y_pred, features = mynet(x)
                y_pred_flat = y_pred.view(y_pred.size(0), -1)
                y_mean_pred = torch.mean(y_pred, dim=[2, 3])
                
                z_flat = z.view(z.size(0), -1)

                z_mean = torch.mean(z, dim = [2,3])
                if ranksim_target == "y_pred":
                    z_mean = torch.mean(y_pred, dim=[2, 3])

                features_flat = features.view(features.size(0), -1) # flattened embeddings
                features_mean_flat = avgpool(features).reshape(features.size(0),-1) # avgpooled flattened embeddings

                source_patch_consistency = myloss(y_mean_pred, y)
                
                ##########################   Ranksim   ############################## 

                if params["ranksim_weight"] > 0:
                    ranking_loss = batchwise_ranking_regularizer(features_mean_flat, z_mean, params["ranksim_interpolation_lambda"])
                    total_loss = source_patch_consistency + params["ranksim_weight"] * ranking_loss
                else:
                    ranking_loss = torch.Tensor([0]).to(device)
                    total_loss = source_patch_consistency
                total_loss.backward()
                
                losses.append([source_patch_consistency.cpu().detach().item(),ranking_loss.cpu().detach().item()])
                optimizer.step()

            if params["early_stopping"]:
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model_state_dict = copy.deepcopy(mynet.state_dict())
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count > params["early_stopping_patience"]:
                    early_stopped = True
                    break



        # compute final prediction, un-normalize, and back to numpy, load the variables again
        if early_stopped and best_model_state_dict is not None:
            mynet.load_state_dict(best_model_state_dict)
        mynet.eval()
        predicted_target_img = mynet(guide_img.unsqueeze(0))[0].squeeze()
        predicted_target_img = source_img_mean + source_img_std * predicted_target_img
        predicted_target_img = predicted_target_img.cpu().detach().squeeze().numpy()
        predictions.append(predicted_target_img)

        if idx == 0:
            try:
                fig = plt.plot(losses)
                plt.savefig(f"RSW{ranksim_weight}_RST{ranksim_target}_PE{positional_encoding}_LR{lr}_batchsize{batchsize}_scaling{scaling}_samples_{idx}")
            except:
                pass
    return np.array(predictions)

if __name__ == "__main__":
    args = parser.parse_args()
    indexes = list(range(args.index_s, args.index_f))
    predictions = pixtransform(data_dir = args.data_dir, 
                               positional_encoding = args.positional_encoding, 
                               ranksim_weight = args.ranksim_weight, 
                               lr = args.lr,
                               batchsize = args.batchsize,
                               scaling = args.scaling,
                               ranksim_target = args.ranksim_target,
                               indexes = indexes)
    np.save(f"predictions{args.filename}_RSW{args.ranksim_weight}_RST{args.ranksim_target}_PE{args.positional_encoding}_LR{args.lr}_batchsize{args.batchsize}_scaling{args.scaling}_samples_{args.index_s}_{args.index_f}.npy",np.array(predictions))
    print(predictions.shape)