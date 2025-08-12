import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from evolutionary_optimization import CMAEvolutionStrategy
from utils import load_data, save_model, load_model, setup_logger

# Constants and configuration
CONFIG = {
    'seed': 42,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'population_size': 100,
    'max_iter': 1000,
    'velocity_threshold': 0.5,
    'flow_theory_threshold': 0.8,
    'scaler': StandardScaler()
}

# Logger setup
logger = setup_logger('main_agent')

class AgentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.Series):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input': self.data.iloc[idx],
            'label': self.labels.iloc[idx]
        }

class AgentModel(nn.Module):
    def __init__(self):
        super(AgentModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, model: AgentModel, optimizer: Adam, loss_fn: nn.MSELoss):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, dataset: AgentDataset, batch_size: int, epochs: int):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in data_loader:
                inputs, labels = batch['input'], batch['label']
                inputs, labels = torch.tensor(inputs.values), torch.tensor(labels.values)
                inputs, labels = inputs.float(), labels.float()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                logger.info(f'Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}')

    def evaluate(self, dataset: AgentDataset):
        data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch['input'], batch['label']
                inputs, labels = torch.tensor(inputs.values), torch.tensor(labels.values)
                inputs, labels = inputs.float(), labels.float()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(data_loader)

class CMAESAgent:
    def __init__(self, model: AgentModel, optimizer: Adam, loss_fn: nn.MSELoss):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cma_es = CMAEvolutionStrategy(model.parameters(), 0.1)

    def train(self, dataset: AgentDataset, batch_size: int, epochs: int):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in data_loader:
                inputs, labels = batch['input'], batch['label']
                inputs, labels = torch.tensor(inputs.values), torch.tensor(labels.values)
                inputs, labels = inputs.float(), labels.float()
                self.cma_es.update(self.model.parameters())
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                logger.info(f'Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}')

    def evaluate(self, dataset: AgentDataset):
        data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch['input'], batch['label']
                inputs, labels = torch.tensor(inputs.values), torch.tensor(labels.values)
                inputs, labels = inputs.float(), labels.float()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(data_loader)

def main():
    # Load data
    data, labels = load_data()

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=CONFIG['seed'])

    # Scale data
    scaler = CONFIG['scaler']
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Create dataset and data loader
    train_dataset = AgentDataset(train_data, train_labels)
    test_dataset = AgentDataset(test_data, test_labels)
    train_data_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Create model, optimizer, and loss function
    model = AgentModel()
    optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()

    # Train model
    agent = Agent(model, optimizer, loss_fn)
    agent.train(train_dataset, CONFIG['batch_size'], CONFIG['epochs'])

    # Evaluate model
    loss = agent.evaluate(test_dataset)
    logger.info(f'Test Loss: {loss}')

    # Save model
    save_model(model, 'agent_model.pth')

if __name__ == '__main__':
    main()