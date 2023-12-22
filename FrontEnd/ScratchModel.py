import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

class NeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(NeuralNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.relu3 = nn.ReLU()
        
        self.fc = nn.Linear(in_features=32*75*75,out_features=num_classes)
        
    def forward(self,input):
        output = self.conv1(input)
        output = self.relu1(output)
        
        output = self.pool(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.relu3(output)
        
        output = output.view(-1,32*75*75)
        output = self.fc(output)
        
        return output
    

def image_loader(image_name):
    loader = transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image