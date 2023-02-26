import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms
    data_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    }
    
    #Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    #Using the image datasets and the trainforms, define the dataloaders
    train_loader = data_loader(image_datasets['train'])
    validation_loader = data_loader(image_datasets['valid'], train=False)
    test_loader = data_loader(image_datasets['test'], train=False)
    
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
        
    # Freezing parameters so we don't backpropagate through them 
    for param in model.parameters():
        param.requires_grad = False
     
    output_size = 102 

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, args.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(args.hidden_units,output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()
        
    print('Network architecture:', args.arch)
    print('Number of hidden units:', args.hidden_units)
    print('Number of epochs:', args.epochs)
    print('Learning rate:', args.learning_rate)
    
    if args.gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")
        
    steps=0
    print_every=1
    
    print("Training")
    for epoch in range(args.epochs):
        print("Epoch: {}/{}".format(epoch+1, args.epochs))

        # Activar el modo de entrenamiento    
        model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            steps += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Limpiar los gradientes existentes
            optimizer.zero_grad()

            # Forward pass - calcular los resultados a partir de los datos de entrada utilizando el modelo
            outputs = model(inputs)

            # Calcular pérdidas
            loss = criterion(outputs, labels)

            # Retropropagación de los gradientes
            loss.backward()

            # Actualizar los parámetros
            optimizer.step()

            # Calcular la precisión
            with torch.no_grad():
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
     
            #Validation
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0

                    for ii, (val_inputs, val_labels) in enumerate(validation_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        val_output = model.forward(val_inputs)
                        val_loss += criterion(val_output, val_labels)

                        val_ret, val_predictions = torch.max(val_output.data, 1)
                        val_correct_counts = val_predictions.eq(val_labels.data.view_as(val_predictions))
                        val_acc = torch.mean(val_correct_counts.type(torch.FloatTensor))
            
                print(f"Batch no: {i+1:03d}/{len(train_loader)} | Loss on training: {loss.item():.4f} | Accuracy on Training:                     {acc.item():.4f} | Loss on Validation: {val_loss.item():.4f} | Accuracy on Validation: {val_acc.item():.4f} ",                     end='\r')

                model.train()
            

    # Do validation on the test set
    
    correct_prediction, total = 0, 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data       
            images, labels = images.to(device), labels.to(device)
            # inferir utilizando el modelo
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_prediction += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct_prediction / total))
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'architecture': args.arch ,
        'n_hidden_units': args.hidden_units,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, 'checkpoint.pth')
    

    
if __name__ == '__main__': main()