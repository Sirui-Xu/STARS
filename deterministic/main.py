import os 
from torch.utils.data import DataLoader
# from model import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
from utils.parser import args
from tqdm import tqdm
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)
dim_used = np.array([0, 1, 2, 3, 4, 5, 18, 19, 20, 33, 34, 35, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
if not args.global_translation:
  dim_used = dim_used[12:]
  from utils import h36motion3d as datasets
else:
  dim_used.sort()
  from utils import h36motion3dab as datasets

model = Model(3,args.input_n,
                          args.output_n,args.st_gcnn_dropout,dim_used,args.n_pre,args.version).to(device)
print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model_name='h36_3d_'+str(args.input_n)+'_'+str(args.output_n)+'_'+str(args.skip_rate)+'_'+str(args.n_pre)+'_'+args.version+'_ckpt_'+['local','global'][args.global_translation]

def train():
    dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    vald_dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=1)
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-05)
    
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss = []
    val_loss = []

    for epoch in range(args.n_epochs):
      running_loss=0
      n=0
      model.train()
      for cnt,batch in tqdm(enumerate(data_loader)): 
          batch=batch.to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim
          
          sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
          sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//3,3)
          sequences_gt_all=batch[:, :args.input_n+args.output_n, dim_used].view(-1,args.input_n+args.output_n,len(dim_used)//3,3)

          optimizer.zero_grad() 

          sequences_predict, sequences_predict_all=model(sequences_train)
          sequences_predict = sequences_predict.permute(0,1,3,2)
          sequences_predict_all = sequences_predict_all.permute(0,1,3,2)
          
          loss=mpjpe_error(sequences_predict,sequences_gt) + 0.5 * mpjpe_error(sequences_predict_all,sequences_gt_all)


          if cnt % 200 == 0:
            print('[%d, %5d]  mpjpe loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
          loss.backward()  
          if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

          optimizer.step()
          running_loss += loss*batch_dim
      print('[%d]  mpjpe loss: %.3f' %(epoch + 1, running_loss.detach().cpu()/n)) 
      train_loss.append(running_loss.detach().cpu()/n)  
      model.eval()
      with torch.no_grad():
          running_loss=0 
          n=0
          for cnt,batch in enumerate(vald_loader):
              batch=batch.to(device)
              batch_dim=batch.shape[0]
              n+=batch_dim
              
              
              sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
              sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//3,3)


              sequences_predict, sequences_predict_all=model(sequences_train)
              sequences_predict = sequences_predict.permute(0,1,3,2)


              loss=mpjpe_error(sequences_predict,sequences_gt)
              # if cnt % 200 == 0:
              #           print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
              running_loss+=loss*batch_dim
          val_loss.append(running_loss.detach().cpu()/n)
          print('[%d, all]  validation loss: %.3f' %(epoch + 1, val_loss[-1]))
      if args.use_scheduler:
        scheduler.step()


      if val_loss[-1] == min(val_loss):
        print('----saving model-----')
        torch.save(model.state_dict(),os.path.join(args.model_path, model_name))
      elif (epoch+1) % 10 == 0:
        torch.save(model.state_dict(),os.path.join(args.model_path, model_name + "_" + str(epoch+1)))

        # plt.figure(1)
        # plt.plot(train_loss, 'r', label='Train loss')
        # plt.plot(val_loss, 'g', label='Val loss')
        # plt.legend()
        # plt.show()

def test():
  assert args.output_n >= args.test_output_n
  model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
  model.eval()
  accum_loss=0  
  n_batches=0 # number of batches for all the sequences
  actions=define_actions(args.actions_to_consider)
  # joints at same loc
  joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
  index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
  joint_equal = np.array([13, 19, 22, 13, 27, 30])
  index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

  for action in actions:
    running_loss=0
    n=0
    dataset_test = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=2,actions=[action])
    print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

    test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
    for cnt,batch in enumerate(test_loader):
      with torch.no_grad():

        batch=batch.to(device)
        batch_dim=batch.shape[0]
        n+=batch_dim
        
        
        all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

        sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
        sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]

        

        sequences_predict, sequences_predict_all=model(sequences_train)
        sequences_predict = sequences_predict.permute(0,1,3,2).contiguous().view(-1,args.output_n,len(dim_used))

        all_joints_seq[:,:,dim_used] = sequences_predict


        all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]

        loss=final_mpjpe_error(all_joints_seq.view(-1,args.output_n,32,3)[:,:args.test_output_n,:,:],sequences_gt.view(-1,args.output_n,32,3)[:,:args.test_output_n,:,:])
        running_loss+=loss*batch_dim
        accum_loss+=loss*batch_dim

    print('loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/n))
    n_batches+=n
  print('overall average loss in mm is: '+str(accum_loss/n_batches))


if __name__ == '__main__':

    if args.mode == 'train':
      train()
    elif args.mode == 'test':
      test()
    elif args.mode=='viz':
       model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
       model.eval()
       visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider,args.global_translation,model_name)



