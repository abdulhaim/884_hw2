import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
from dataset import ObjPushDataset
from push_env import PushingEnv

from logger import Logger
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
from plot_data import load_data, plot_data

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Model(nn.Module):

    def __init__(self, num_inputs, num_outputs):

        super(Model, self).__init__()

        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_outputs)

        self.mse = nn.MSELoss()
        self.optim = Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def optimize(self, input_feed, output):
        predicted_push_output = self(input_feed)
        loss = self.mse(predicted_push_output.double(), output)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.cpu().data)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, "{}.pth".format("forward")))

    def load(self, model_dir):
        self.load_state_dict(torch.load(os.path.join(model_dir, "{}.pth".format("forward"))))

    def infer(self, init_obj, goal_obj):
        feed = torch.cat((init_obj, goal_obj), axis=1)
        return self(feed)

def get_loss(loader, model):
    total_loss = 0
    for data in loader:
        obj1 = data['obj1']
        obj2 = data['obj2']
        push = data['push']
        input_feed = torch.cat((obj1.float(), push.float()), axis=1)
        
        predicted_output = model(input_feed)
        loss = model.mse(predicted_output.double(), obj2)    
        total_loss += float(loss.cpu().data)

    return total_loss/len(loader)

def train_forward():

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dir = 'push_dataset/train'
    test_dir = 'push_dataset/test'
    bsize = 64
    num_epochs = 5
    num_inputs = 6
    num_outputs = 2

    train_loader = DataLoader(ObjPushDataset(train_dir), batch_size=bsize, shuffle=True)
    valid_loader = DataLoader(ObjPushDataset(test_dir), batch_size=bsize, shuffle=True)  
    logger = Logger(logdir="runs/", run_name=f"{train_forward}-{time.ctime()}")
    model = Model(num_inputs,num_outputs)
    print("Model Architecture", model)

    for epoch in range(num_epochs):  
        for data in train_loader:
            obj1 = data['obj1']
            obj2 = data['obj2']
            push = data['push']

            input_feed = torch.cat((obj1.float(), push.float()), axis=1)
            loss = model.optimize(input_feed, obj2)

            logger.log_epoch("training_loss", loss,epoch)

    print("Done")
    print("Train Loss", get_loss(train_loader, model))
    print("Test Loss", get_loss(valid_loader,model))
    model.save("models")

    return model 

def call_cem(model, init_obj, goal_obj, env):

    start_x, start_y, end_x, end_y = env.sample_push(init_obj[0, 0], init_obj[0, 1])

    push_np = np.array([start_x, start_y, end_x, end_y])
    push = torch.FloatTensor(push_np).unsqueeze(0)

    goal_pred = model.infer(init_obj, push)
    goal_pred = goal_pred.detach().numpy().flatten()

    goal_obj = goal_obj.flatten()

    loss = np.linalg.norm(goal_obj - goal_pred)
    print("Final Loss", loss)

    return torch.from_numpy(push_np)

def generate_stats(model):
    env = PushingEnv(ifRender=False)

    env.go_home()
    env.reset_box()
    np.random.seed(1)

    start_x, start_y, end_x, end_y = env.sample_push(env.box_pos[0], env.box_pos[1])

    print("Initial", (start_x, start_y))
    print("Final", (end_x, end_y))

    init_obj, goal_obj = env.execute_push(start_x, start_y, end_x, end_y, "imgs_forward_ground/ground_truth")
    env.createGif("imgs_forward_ground/", "forward_ground")

    init_obj = torch.FloatTensor(init_obj).unsqueeze(0)
    goal_obj = torch.FloatTensor(goal_obj).unsqueeze(0)
        
    # Get push from your model. Your model can have a method like "push = self.model.infer(init_obj, goal_obj)"      
    push = call_cem(model, init_obj, goal_obj, env)
    push = push.detach().numpy()
    start_x, start_y, end_x, end_y = push

    env.reset_box()       
    env.execute_push(start_x, start_y, end_x, end_y, "imgs_forward_predict/prediction")
    env.createGif("imgs_forward_predict/", "forward_predict")

    final_obj = env.get_box_pose()[0][:2]
    goal_obj = goal_obj.numpy().flatten()
    loss = np.linalg.norm(final_obj-goal_obj)
    print(f"L2 Distance between final obj position and goal obj position is {loss}")

def plan_forward_model_extrapolate(env, model, seed=0):
    env.go_home()
    env.reset_box()
    np.random.seed(seed)
    init_obj = env.get_box_pose()[0][:2]
    push_ang = np.pi/3 + np.random.random()*np.pi/3
    
    if np.random.random() < 0.5:
        push_ang -= np.pi

    obj_x, obj_y = env.get_box_pose()[0][:2]            
    start_x, start_y, end_x, end_y = env.sample_push(obj_x=obj_x, obj_y=obj_y, push_len=0.1, push_ang=push_ang)
    _, goal_obj = env.execute_push(start_x, start_y, end_x, end_y, "imgs_forward_ground_extrapolate/ground_truth")
    env.createGif("imgs_forward_ground_extrapolate/", "forward_ground_extrapolate")

    ### Write code for visualization ###
    env.reset_box()        
    goal_obj = torch.FloatTensor(goal_obj).unsqueeze(0)
    init_obj = torch.FloatTensor(init_obj).unsqueeze(0)
    # Get push from your model. Your model can have a method like "push = self.model.infer(init_obj, goal_obj)"
    push = call_cem(model, init_obj, goal_obj, env)
    push = push[0].detach().numpy()
    # Your code should ideally call this twice: once at the start and once when you get intermediate state.
    start_x, start_y, end_x, end_y = push     
    env.execute_push(start_x, start_y, end_x, end_y, "imgs_forward_predict_extrapolate/prediction")
    env.createGif("imgs_forward_predict_extrapolate/", "forward_predict_extrapolate")

    final_obj = env.get_box_pose()[0][:2]
    goal_obj = goal_obj.numpy().flatten()
    loss = np.linalg.norm(final_obj-goal_obj)
    print(f"L2 Distance between final obj position and goal obj position is {loss}")
    return loss     

if __name__ == "__main__":

    model = train_forward()

    # Download data from tensorboard, then uncomment
    #data = load_data('losses_forward/')
    #plot_data(data)

    # NOTE: For CEM algorithm, we recommend sampling push angle and push length 
    # and then using the push angle and push length to calculate the initial and the final end-effector 
    # positions of the robot’s arm tip. You can look at sample-push function in push-env.py.
    loss = generate_stats(model)







