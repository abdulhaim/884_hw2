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

	def optimize(self, push_input, push_output):
		predicted_push_output = self(push_input)
		loss = self.mse(predicted_push_output, push_output)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		return float(loss.cpu().data)

	def save(self, model_dir):
		os.makedirs(model_dir, exist_ok=True)
		torch.save(self.state_dict(), os.path.join(model_dir, "{}.pth".format(self.name)))

	def load(self, model_dir):
		self.load_state_dict(torch.load(os.path.join(model_dir, "{}.pth".format(self.name))))

	def infer(self, init_obj, goal_obj):
		feed = torch.cat((init_obj, goal_obj), axis=1)
		return self(feed)

def get_loss(loader, model):
	total_loss = 0
	for data in loader:
		obj1 = data['obj1']
		obj2 = data['obj2']
		push_output = data['push']
		push_input = torch.cat((obj1, obj2), axis=1)
		
		predicted_push_output = model(push_input.float())
		loss = model.mse(predicted_push_output, push_output)		 
		total_loss += float(loss.cpu().data)

	return total_loss/len(loader)

def train_inverse():

	torch.manual_seed(1)
	np.random.seed(1)
	random.seed(1)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	train_dir = 'push_dataset/train'
	test_dir = 'push_dataset/test'
	bsize = 64
	num_epochs = 5
	num_inputs = 4
	num_outputs = 4

	train_loader = DataLoader(ObjPushDataset(train_dir), batch_size=bsize, shuffle=True)
	valid_loader = DataLoader(ObjPushDataset(test_dir), batch_size=bsize, shuffle=True)  
	logger = Logger(logdir="runs/", run_name=f"{train_inverse}-{time.ctime()}")
	model = Model(num_inputs,num_outputs)
	print("Model Architecture", model)

	for epoch in range(num_epochs):  
		

			obj1 = data['obj1']
			obj2 = data['obj2']
			push_output = data['push']

			push_input = torch.cat((obj1, obj2), axis=1)
			loss = model.optimize(push_input.float(), push_output)

			logger.log_epoch("training_loss", loss,epoch)

	print("Done")
	print("Train Loss", get_loss(train_loader, model))
	print("Test Loss", get_loss(valid_loader,model))

	return model 
if __name__ == "__main__":

	model = train_inverse()

	# Download data from tensorboard, then uncomment
	data = load_data('losses/')
	#plot_data(data)

	env = PushingEnv(ifRender=False)
	num_trials = 10
	errors = np.zeros(num_trials)
	# save one push
	errors[0] = env.plan_inverse_model(model)
	print("test loss:", errors[0])
	# try 10 random seeds
	# for seed in range(1,10):
	# 	errors[seed] = env.plan_inverse_model(model, seed=seed)
	# 	print("test loss:", errors[seed])
	
	# print("average loss:", np.mean(errors))




