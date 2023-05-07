import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

'''
'''
class Linear_QNet(nn.Module):
    '''
    Constructor

    Parameters
    ------------
    input_size      size of the input we are giving to the model
    hidden_size     size of the hidden layers
    output_size     size of the output from our model
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)


    '''
    Forward function
    
    Parameters
    -----------
    
    '''
    def forward(self, x):
        x = F.relu(self.linear1(x))     # first applies a linear transformation to the incoming data (x), then applies the rectified linear unit function element-wise
        x = self.linear2(x)

        return x


    '''
    
    Parameters
    -----------
    
    '''
    def save(self, file_name = 'model.pth'):
        # Create a folder path for our model
        model_folder_path = './model'

        # If the path doesn't exist, make a new directory for it
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Join the path and the filename
        file_name = os.path.join(model_folder_path, file_name)

        # Create a dictionary and map each layer to its parameter tensor. Then save it to the disk file
        torch.save(self.state_dict(), file_name)



class QTrainer:
    '''
    Constructor

    Parameters
    ------------
    model       instance of our model
    lr          learning rate
    gamma
    '''
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    '''
    Trains the next step for the model to take
    
    Parameters
    -----------
    state           state we are currently in
    action          action for the model to take
    reward          the reward we get for taking that action
    next_state      the next state 
    done            indicates whether we got a game over
    '''
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.float)
        #print(reward)
        reward = torch.tensor(reward, dtype = torch.float)
        # (n, x)

        # The shape of the state should be 1
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

            #1 : predicted Q values with current state
            pred = self.model(state)

            target = pred.clone()

            for index in range(len(done)):
                Q_new = reward[index]

                if not done[index]:
                    Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

                target[index][torch.argmax(action[index]).item()] = Q_new

            #2 : Q_new = r + y * max(next_predicted Q value) ->  only do this if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()
