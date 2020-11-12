import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from workspace_utils import active_session,keep_awake
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import loader,image_classifier
from workspace_utils import active_session
import argparse
import sys,os,json



def main():
    
    model_names = sorted(name for name in models.__dict__\
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]) and ( name.startswith("vgg") or
                                                                                             name.startswith("alexnet")))
    parser = argparse.ArgumentParser()  
    parser.add_argument("data_dir",help="name of the directory from where to load the data")
    parser.add_argument("--save_dir",help="directory to save checkpoints(default :none)",metavar='save')
    parser.add_argument("--arch",choices=model_names,default="vgg16",help="Choose Architecture (default:vgg16)")
    parser.add_argument("--gpu",action="store_true",help="Use GPU for training")                   
    parser.add_argument("--learning_rate",type=float,default=0.003,metavar='lr',dest='learning_rate',help="Learning Rate(default:0.003)")
    parser.add_argument("--epochs",type=int,default=1,dest='epochs',help="Number of Epochs for training(default:1)") 
    parser.add_argument("--print_every",type=int,default=5,metavar='P',dest='validate_every',help="Number of steps after which output should be printed(default:5)") 
    parser.add_argument("--skip_after",type=int,dest='skip_after',metavar='skip',help="Number of steps after which training module should be exited(default:None)") 
    parser.add_argument("--hidden_unit_1","-fc1",metavar='fc1',type=int,dest='n_fc1',default=4096,help="Number of hidden units for layer 1(default:4096)") 
    parser.add_argument("--hidden_unit_2","-fc2",metavar='fc2',type=int,default=2048,dest='n_fc2',help="Number of hidden units for layer 2(defaukt:2048)") 
    parser.add_argument("--hidden_unit_3","-fc3",metavar='fc3',type=int,default=1024,dest='n_fc3',help="Number of hidden units for layer 3(default:1024)")    
    args = parser.parse_args()  
   
      
    if args.save_dir and not os.path.exists(args.save_dir):
        print("save directory doesn't exist.please try again")
        sys.exit(-1)
        
    if os.path.exists(args.data_dir):        
      
        traindataloaders,validationdataloaders,class_to_idx= loader.get_loaders(args.data_dir)
        
        if traindataloaders and validationdataloaders:

            device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
            model = image_classifier.init_classifier(args.arch,device,args.n_fc1,args.n_fc2,args.n_fc3)
            model.classifier.class_to_idx = class_to_idx
            optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)            
            criterion = nn.NLLLoss()
            dataloaders = [traindataloaders,validationdataloaders]
            with active_session():
                training_loss = image_classifier.train(model,dataloaders,criterion,device,optimizer,epochs=args.epochs,validate_every=args.validate_every,skip_after=args.skip_after)
            if args.save_dir: 
                image_classifier.save_checkpoint(model,optimizer,args.epochs,training_loss,args.save_dir,args.arch)
            
        else:
            print("data couldn't be read or no valid train or valid directory.Pleae check if /train and /valid exists")                    
    
    else:
        print("data directory entered doesn't exists.Please try again")
     
    
    
if __name__== '__main__':
    main()

    

