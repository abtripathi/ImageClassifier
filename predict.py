import torch
import image_classifier
import os.path
from torchvision import transforms
import helper
import argparse,sys
from PIL import Image 




def main():
    
    names=None
    parser = argparse.ArgumentParser()  
    parser.add_argument("input",help="file path of image")
    parser.add_argument("checkpoint",help="directory to load checkpoint",metavar='L')
    parser.add_argument("--top_k","-k",dest='top_k',metavar='K',type=int,default="1",help="Top K class (default:1)")
    parser.add_argument("--gpu",action="store_true",help="Use GPU for inference")                   
    parser.add_argument("--category_names","-c", metavar='cat_names',dest="label_mappings", \
                        help="file name describing mapping of categories to real names(default:cat_to_name.json)")
   
    args = parser.parse_args()
        
    if not os.path.exists(args.checkpoint):
        print("checkpoint filepath doesn't exists.Please try again")
        sys.exit()
    if not os.path.exists(args.input):
        print("Image path doesn't exists.Please try again")  
        sys.exit()
    if args.label_mappings:
        if not os.path.exists(args.label_mappings):
             print("mapping file doesn't exist")
             sys.exit()
            
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu') 
    model = image_classifier.get_model(args.checkpoint)
    model.to(device)
    
    probabalities,class_names = predict(args.input,model,device,k=args.top_k)
    if args.label_mappings:
        category_dict = helper.label_mapping(args.label_mappings)
        names = [ category_dict[item] for item in class_names]
    category_names =  names if names else class_names
    zipped_list = zip(category_names,probabalities[0])
    for name,probability in zipped_list:
                #print(" name {} ".format(name) ,"Probability {}".format(probability))
        helper.print_results(name=name,probability=format(probability,'<45.3f'))
        
        
    
def predict(image_path, model,device,k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   

    inverted_dict = {value: key for key, value in model.classifier.class_to_idx.items()}
    image = process_image(image_path)
    
    logps = model(image.to(device))
    ps = torch.exp(logps)
    top_k,top_class = ps.topk(k,dim=1)  
    
    #print(top_class.squeeze().tolist())
    
    classes= [ inverted_dict[item] for item in top_class.tolist()[0]]
    #category_names = [ helper.cat_to_name[item] for item in classes ]
                    
    return top_k.tolist(),classes     

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
    image = Image.open(image)
    image = test_transforms(image)
    image = image.view(1,*image.shape)
   
    return image

    
if __name__=='__main__':
    main()
    
    