import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        

        # Fully Connected Layers
        self.fc1 = nn.Linear(15488, 1024)# The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc2 = nn.Linear(1024,  1024)
        self.fc3 = nn.Linear(1024,  136) # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs

        # Dropouts
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.drop4 = nn.Dropout(0.3)
       # self.drop5 = nn.Dropout(0.5)
      


    def forward(self, x):

        # First - Convolution + Activation + Pooling + Dropout
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
       # x = self.drop1(x)
        #print("First size: ", x.shape)

        # Second - Convolution + Activation + Pooling + Dropout
        x = self.drop1(self.pool(F.relu(self.conv2(x))))
        x = self.pool(x)
        # Third - Convolution + Activation + Pooling + Dropout
        x = self.drop2(self.pool(F.relu(self.conv3(x))))
        x = self.pool(x)
        # Flattening the layer
        x = x.view(x.size(0), -1)

        # First - Dense + Activation + Dropout
        x = self.drop3(F.relu(self.fc1(x)))
        #print("First dense size: ", x.shape)

        # Second - Dense + Activation + Dropout
        x = self.drop4(F.relu(self.fc2(x)))

        # Final Dense Layer
        x = self.fc3(x)
       
        return x
    

