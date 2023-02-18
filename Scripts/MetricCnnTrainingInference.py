import torch, os, argparse, sys
sys.path.append('../Packages')
import data.convert as convert
import util.tensors as tensors
import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DatasetHCP
from pde import *
from model3D import *

def train(brain_id, input_dir, output_dir, gpu_device=0, epoch_num=10000, learning_rate=1e-4, terminating_loss=1e6, checkpoint_save_frequency=1000):
    torch.cuda.set_device(gpu_device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    output_dir = f'{output_dir}/{brain_id}'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    blocks = [40,30,40]
    model = DenseED(in_channels=3, out_channels=7, 
                    imsize=100,
                    blocks=blocks,
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
                    upsample='nearest')
                    
    model.train()
    model.cuda()
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    dataset_id = DatasetHCP(input_dir, sample_name_list=[str(brain_id)])
    dataloader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=0)

    with open(f'{output_dir}/loss.txt', 'w+') as f:
        f.write(f'Architecture {blocks}\n')
        f.write(f'Adadelta: lr={learning_rate};\n')

    for epoch in tqdm(range(epoch_num)):
        epoch_loss_id = 0

        for i, batched_id_sample in enumerate(dataloader_id):
            '''innerdomain backpropagate'''
            input_id = batched_id_sample['vector_field']
            input_id.requires_grad = True
            optimizer.zero_grad()

            u_pred_id = model(input_id)
            mask = batched_id_sample['mask'].squeeze()
            pde_loss = pde(u_pred_id.squeeze(), input_id.squeeze(), mask)
            f_pred_id = torch.einsum('...ij,...ij->...ij', pde_loss, mask.expand(3,-1,-1,-1))
            f_true_id = torch.zeros_like(f_pred_id)

            loss_id = criterion(f_pred_id, f_true_id)
            loss_id.backward()
            optimizer.step()
            epoch_loss_id += loss_id.item()
        scheduler.step(epoch_loss_id)

        with open(f'{output_dir}/loss.txt', 'a') as f:
            f.write(f'{epoch_loss_id}\n')
  
        if epoch%checkpoint_save_frequency==0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_id_state_dict': optimizer.state_dict(),
            'loss_id': epoch_loss_id,
            }, f'{output_dir}/epoch_{epoch}_checkpoint.pth.tar')

        if epoch_loss_id<terminating_loss:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_id_state_dict': optimizer.state_dict(),
            'loss_id': epoch_loss_id,
            }, f'{output_dir}/final_checkpoint.pth.tar')
            break
            
    
    checkpoint = torch.load(f'{output_dir}/epoch_{epoch}_checkpoint.pth.tar')
    model = DenseED(in_channels=3, out_channels=7, 
                    imsize=100,
                    blocks=blocks,
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
                    upsample='nearest')
    model.load_state_dict(checkpoint['model_state_dict'])

    vector_lin = convert.read_nhdr(f'{input_dir}/{brain_id}/{brain_id}_shrinktensor_principal_vector_field.nhdr').to('cuda').float()
    mask = convert.read_nhdr(f'{input_dir}/{brain_id}/{brain_id}_shrinktensor_filt_mask.nhdr').to('cuda').float()

    u_pred = model(vector_lin.unsqueeze(0))
    u_pred = u_pred.squeeze()

    metric_pred_mat = eigen_composite(u_pred)
    metric_pred_lin = tensors.mat2lin(metric_pred_mat)
    tensor_pred_mat = torch.inverse(metric_pred_mat)
    tensor_pred_lin = tensors.mat2lin(tensor_pred_mat)

    file_name = f'{output_dir}/{brain_id}_learned_metric_final.nhdr'
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(metric_pred_lin.cpu().detach().numpy(),(3,2,1,0))), file_name)

parser = argparse.ArgumentParser()
parser.add_argument('--brain_id', type=str, required=True, help='the HCP subject ID')
parser.add_argument('--input_dir', type=str, required=True, help='path to the brain data')
parser.add_argument('--output_dir', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--gpu_device', type=int, required=True, help='an integer for the accumulator')
parser.add_argument('--epoch_num', type=int, required=False, help='total epochs for training')
parser.add_argument('--learning_rate', type=float, required=False, help='initial learning rate of model')
parser.add_argument('--terminating_loss', type=float, required=False, help='loss threshold for termination')
parser.add_argument('--checkpoint_save_frequency', type=int, required=False, help='frequency of checkpoint save')
args = parser.parse_args()

train(brain_id=args.brain_id, input_dir=args.input_dir, output_dir=args.output_dir, gpu_device=args.gpu_device, epoch_num=args.epoch_num, learning_rate=args.learning_rate, terminating_loss=args.terminating_loss, checkpoint_save_frequency=args.checkpoint_save_frequency)
