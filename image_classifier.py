from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import helper
import os.path

# global checkpoint
checkpoint = None


class Network(nn.Module):
    def __init__(self,n_input,n_h1,n_h2,n_h3,n_output):
        
        super().__init__()      
        self.fc1 = nn.Linear(n_input,n_h1)       
        self.fc2 = nn.Linear(n_h1,n_h2)       
        self.fc3= nn.Linear(n_h2,n_h3)      
        self.output = nn.Linear(n_h3,n_output)
              
    
    def forward(self,x):
        x= F.dropout(F.relu(self.fc1(x)))
        x= F.dropout(F.relu(self.fc2(x)))        
        x= F.dropout(F.relu(self.fc3(x)))
        x= F.log_softmax(self.output(x),dim=1)          
        return x

    
def init_classifier(arch,device,fc1,fc2,fc3):
        
        model = models.__dict__[arch](pretrained=True)
         
        # freeze the weights
        for param in model.parameters():
            param.requires_grad_(False)
       
        n_inputs = 25088 if arch.startswith('vgg') else 9216
        classifier = Network(n_inputs,fc1,fc2,fc3,102)
        model.classifier = classifier
        model.to(device)
        
        return model
   
                  
def get_model(filepath):
    
    global checkpoint
    if not checkpoint:
        
        load_checkpoint(filepath)
    
    arch = checkpoint['arch']
    n_input,n_h1,n_h2,n_h3,n_output = checkpoint['n_input'],checkpoint['n_h1'],checkpoint['n_h2'],checkpoint['n_h3'],checkpoint['n_output']
  
    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad_(False)  
    classifier = Network(n_input,n_h1,n_h2,n_h3,n_output)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])   
    model.classifier = classifier
    model.classifier.class_to_idx = checkpoint['class_to_idx']
    
    return model

def get_optimizer(filepath):
    
    global checkpoint
    if not checkpoint:
        load_checkpoint(filepath)
    return checkpoint['optimizer_state_dict']

                  
def train(model,dataloader,criterion,device,optimizer,epochs=1,validate_every=None,skip_after=None):
    
    training_losses = {}
    model.train()
    for e in range(epochs):
        training_loss = 0
        normal = False
        for ii,(images,labels) in enumerate(dataloader[0]):
            if skip_after and ii==skip_after:
                break
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images),labels)            
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            if validate_every and (ii+1)%validate_every == 0:
                accuracy,valid_loss=validation(model,dataloader[1],criterion,device)
                
                helper.print_results(epoch=(str(e+1)+'/'+str(ii+1)),accuracy=format(accuracy,'.3f'),
                              training_loss = format(training_loss/(ii+1),'.3f'),valid_loss=format(valid_loss,'.3f'))
        else:        
            normal = True
            training_loss /= len(dataloader[0])
            #training_losses['epoch-'+e] = training_loss
        if not normal:            
            training_loss /= skip_after            
        training_losses['epoch-'+str(e+1)] = format(training_loss,'.3f')    
    return training_losses
            
                   
def validation(model,dataloader,criterion,device):
    model.eval()
    accuracy,valid_loss = 0,0
    with torch.no_grad():
        for images,labels in dataloader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss = criterion(output,labels)
            valid_loss += loss.item()
            ps = torch.exp(output)
            accuracy += calculate_accuracy(ps,labels)
    model.train()
    accuracy /= len(dataloader)
    return accuracy,valid_loss/len(dataloader)
        
def calculate_accuracy(ps,labels):
    top_k,topk_class = ps.topk(1,dim=1)
    equals= topk_class==labels.view(*topk_class.shape)
    return torch.mean(equals.type(torch.FloatTensor)).item()


def save_checkpoint(model,optimizer,epochs,training_loss,save_dir,arch):
    
    checkpoint={
            'arch':arch,
            'n_input':model.classifier.fc1.in_features,
            'n_h1':model.classifier.fc2.in_features,
            'n_h2':model.classifier.fc3.in_features,
            'n_h3':model.classifier.output.in_features,
            'n_output':model.classifier.output.out_features,
            'classifier_state_dict':model.classifier.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'epochs_trained':epochs,
            'train_loss':training_loss,
            'class_to_idx':model.classifier.class_to_idx
          }
    
    #print(checkpoint)
    torch.save(checkpoint,os.path.join(save_dir,'checkpoint.pth'))
    
def load_checkpoint(filepath):
    
    global checkpoint
    checkpoint = torch.load(filepath,map_location=lambda storage,loc:storage)
 
    
  




            
    

