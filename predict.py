import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
import PIL
from PIL import Image
import json
import seaborn as sns
from collections import OrderedDict 
import torchvision
from PIL import Image
from math import ceil

def arg_parser():
    
    parser = argparse.ArgumentParser(description = 'Parser for predict.py')
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--image', dest="image", default='flowers/test/20/image_04910.jpg', action="store", type = str)
    parser.add_argument('checkpoint_path', default='./checkpoint.pth', nargs='?', action="store", type = str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    args = parser.parse_args()
    return args

def process_image(image):

    img_loader = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def main():
    args = arg_parser()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    use_gpu = torch.cuda.is_available() and args.gpu
    if use_gpu:
        print("Using GPU.")
        device = torch.device('cuda')
    else:
        print("Using CPU.")
        device = torch.device('cpu')
    
    checkpoint = torch.load(args.checkpoint_path)
    
    if args.arch == 'densenet121':
         model = models.densenet121(pretrained=True)
         input_size = model.classifier.in_features
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_feature
    
    else:
        raise TypeError("The arch specified is not supporte")
   
    output_size = 102 

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, args.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(args.hidden_units,output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    #to freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    #load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    torch_image = torch.from_numpy(np.expand_dims(process_image(args.image), axis=0)).type(torch.FloatTensor).to("cpu")

    # Get the log probabilities
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Get the top K probabilities and labels
    top_probs, top_labels = linear_probs.topk(args.top_k)

    # Detach the details from the tensors and convert to numpy arrays
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert labels to classes using class_to_idx dictionary
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]

    # Convert labels to flower names
    top_flowers = [cat_to_name[label] for label in top_labels]
    
    for i, j in enumerate(zip(top_flowers, top_probs)):
        print ("Rank {}:".format(i+1), "Flower: {}, likehood: {}%".format(j[0], j[1]))
    
if __name__ == '__main__': main()