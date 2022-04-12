import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchvision.models as models
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Argument configuration 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",
        help = "CNN model to evaluate. Options: AlexNet (default), VGG16 or ResNet18",
        default = "AlexNet")
parser.add_argument("-e", "--eval",type = int, choices=[0, 1],
        help = "Evaluate with PyTorch profiler. Options: 0 (default) no evaluation or 1 for evaluation",
        default = 0)
parser.add_argument("-d", "--dist_plot", type = int, choices=[0, 1],
        help = "Plots Convolutional layer kernel weight distributions. Options: 0 (default) none or 1 for plotting",
        default = 0)
parser.add_argument("-g", "--gen", type = int, choices=[0, 1],
        help = "Generates the ONNX for each layer model. Options: 0 (default) none or 1 for full generation",
        default = 0)        
args = parser.parse_args()

# Choosing model to evaluate
if args.model == "AlexNet":
    model = models.alexnet(pretrained=True).cuda()
elif args.model == "VGG16":
    model = models.vgg16(pretrained=True).cuda()
elif args.model == "ResNet18":
    model = models.resnet18(pretrained=True).cuda()
elif args.model == "SqueezeNet":
    model = models.squeezenet1_0(pretrained=True).cuda()
else:
    print("ERROR: Model is not included in the evaluation zoo")
    exit()
# Print distributions
if args.dist_plot:
    i = 0
    zero_count = []
    layer_names = []
    input_sizes = []
    K = []
    HW = []
    C = []
    N = []
    model_stats = summary(model,(1, 3, 224, 224),col_names=["input_size"])
    for layer_info in model_stats.summary_list:
        if not str(layer_info).find("Conv2d: 1") == -1 or not str(layer_info).find("Conv2d: 2") == -1 or not str(layer_info).find("Conv2d: 3") == -1:
            input_sizes.append(layer_info.input_size)
    for l in list(model.named_parameters()):
        if (not l[0].find("features") == -1 or not l[0].find("conv") == -1) and not l[0].find("weight") == -1:
            kernel = l[1].detach().cpu().numpy() 
            input_size = input_sizes[i]
            K.append(kernel.shape[2])
            HW.append(input_size[2])
            C.append(input_size[1])
            N.append(kernel.shape[0])
            i += 1
            plt.figure()
            #print(l[0], ':', kernel.shape)
            plt.title(args.model+": Conv "+str(i)+" Layer")
            layer_names.append("Conv "+str(i))
            plt.xlabel("Kernel values")
            plt.ylabel("Count")
            plt.hist(kernel.flatten(), bins = 2**8)
            bins = np.linspace(kernel.min(),kernel.max(),2**8)
            zero_count.append(np.count_nonzero(np.digitize(kernel.flatten(),bins)==np.abs(bins).argmin()))
    plt.figure()
    plt.grid(zorder=0)
    plt.title(args.model+" number of zeros per layer")
    plt.xlabel("Layer names")
    plt.ylabel("Zero count")
    plt.bar(layer_names,zero_count,zorder=3)
    plt.xticks(rotation=90)
    print(K)
    print(HW)
    print(C)
    print(N)
    plt.show()

# Generate VHDL code for each Conv2D layer with pretrained weights   
if args.gen:
    input_names = ['input']
    output_names = ['output']
    input_sizes = []
    #summary(model,(1, 3, 224, 224),col_names=["input_size", "output_size", "num_params", "mult_adds"])
    model_stats = summary(model,(1, 3, 224, 224),col_names=["input_size"])
    for layer_info in model_stats.summary_list:
        if not str(layer_info).find("Conv2d: 1") == -1 or not str(layer_info).find("Conv2d: 2") == -1 or not str(layer_info).find("Conv2d: 3") == -1:
            input_sizes.append(layer_info.input_size)
    i = 0
    for p in list(model.named_parameters()):
        if (not p[0].find("features") == -1 or not p[0].find("conv") == -1) and not p[0].find("weight") == -1:
            kernel = p[1].detach().cpu().numpy()
            # input_size -> (b, C, H, W)
            # kernel.shape -> (N,C,k,k)
            #print(p[0], ':', kernel.shape)
            input_size = input_sizes[i]
            hw = input_size[2]
            c = input_size[1]
            k = kernel.shape[2]
            n = kernel.shape[0]
            # Conv layer definition
            class Convkxk_Net(nn.Module):
                def __init__(self):
                    super(Convkxk_Net,self).__init__()
                    # batch size, n conv output, kernel size kxk, stride 1-1
                    self.conv1 = nn.Conv2d(c, n, k)
                    # loading pre-trained weights
                    self.conv1.weight = torch.nn.Parameter(p[1].detach().cpu())
                def forward(self, x):
                    # Maxpooling 2x2 and ReLu activation function
                    #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                    x = F.relu(self.conv1(x))
                    return x
            # Dummy input data
            input = torch.randn(1,c,hw,hw)
            # Convolution layer model
            convkxk_net = Convkxk_Net()
            print("Now generating ...")
            onnx_file = 'vhdl_generated/conv%dx%d/hw_%d/c_%d/conv%dx%d_%d_%d_%d' % (k, k, hw, c, k, k, hw, c, n) + '.onnx'
            torch.onnx.export(convkxk_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)
            i += 1
    print("ONNX files successfuly generated")

if args.eval:    
    # Image batch retrieval
    # Normalize and Resize input 
    loader = transforms.Compose([transforms.Resize(size=(224,224)),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),transforms.ConvertImageDtype(dtype=float)])
    transPILToTensor = transforms.PILToTensor()
    # Path to directory
    directory = 'images'
    # Batch empty initialization on GPU with CUDA
    batch = torch.empty(0,dtype=torch.float32,device='cuda')
    # Iterate over every file
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # Check if it is a file
        if os.path.isfile(f):
            # Open image
            image = Image.open(f)
            image_tensor = transPILToTensor(image).float()
            input = loader(image_tensor).cuda()
            input = input.unsqueeze(0)
            batch = torch.cat((batch,input),0)
    batch = batch.float()

    # Initialize GPU with a dummy batch computation
    model(batch)
    # Model profiling
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(batch)
    print(args.model," Profiler")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
