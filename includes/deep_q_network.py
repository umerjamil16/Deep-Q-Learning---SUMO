import torch.nn as nn
import torch.nn.functional as F
import torch .optim as optim
import torch as T 
import os

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # The deep neural network
        self.fc1 = nn.Linear(input_dims, 512) #* keyword is used for unpacking list
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, n_actions)

        self.droput = nn.Dropout(0.2)
            
        self.sigmoid = nn.Sigmoid()

        # Defining optimizer
#        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))

        x = F.relu(self.fc2(x))
        x = self.droput(x)

        x = F.relu(self.fc3(x))
        x = self.droput(x)

        x = F.relu(self.fc4(x))
        x = self.droput(x)

        x = F.relu(self.fc5(x))

        actions = self.fc6(x)

        #actions = self.sigmoid(self.fc6(x))

        return actions


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    