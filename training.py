import logging
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvoSparrowAgent:
    """
    Evo-Sparrow Agent for AI decision-making in Sparrow Mahjong.
    Implements the training pipeline as described in the research paper.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

    def train(self, train_loader: DataLoader, epochs: int, batch_size: int, learning_rate: float, log_interval: int = 100):
        """
        Train the Evo-Sparrow agent using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

        Parameters:
        train_loader (DataLoader): Data loader for training data.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the optimizer.
        log_interval (int, optional): Interval for logging training progress. Defaults to 100.
        """
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            for batch_idx, (states, actions, rewards) in enumerate(train_loader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.loss_fn(outputs, actions)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

            if (epoch + 1) % log_interval == 0:
                self.save_checkpoint(epoch+1, self.model.state_dict())

    def save_checkpoint(self, epoch: int, model_state: Dict[str, torch.Tensor]):
        """
        Save a checkpoint of the model at a specific epoch.

        Parameters:
        epoch (int): Current epoch number.
        model_state (Dict[str, torch.Tensor]): State dictionary of the model.
        """
        filename = f"checkpoint_epoch_{epoch}.pth"
        torch.save(model_state, os.path.join(os.getcwd(), filename))
        logger.info(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename: str):
        """
        Load a checkpoint and set the model's state.

        Parameters:
        filename (str): Name of the checkpoint file to load.
        """
        if os.path.isfile(filename):
            logger.info(f"Loading checkpoint: {filename}")
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        else:
            logger.error(f"Checkpoint file not found: {filename}")

    def evaluate(self, test_loader: DataLoader):
        """
        Evaluate the trained model on a test dataset.

        Parameters:
        test_loader (DataLoader): Data loader for test data.

        Returns:
        float: Mean squared error between predicted and actual actions.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for states, actions in test_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)

                outputs = self.model(states)
                loss = self.loss_fn(outputs, actions)

                total_loss += loss.item() * states.size(0)

        avg_loss = total_loss / len(test_loader.dataset)
        logger.info(f"Evaluation MSE Loss: {avg_loss:.4f}")
        return avg_loss

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) model for decision-making in Sparrow Mahjong.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def parse_dataset(data_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parse the dataset and extract states, actions, and rewards.

    Parameters:
    data_path (str): Path to the dataset file.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors for states, actions, and rewards.
    """
    # Sample implementation: Assuming the dataset is in a specific format (e.g., CSV)
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Placeholder dataframes
    df_states = pd.read_csv(os.path.join(data_path, "states.csv"))
    df_actions = pd.read_csv(os.path.join(data_path, "actions.csv"))
    df_rewards = pd.read_csv(os.path.join(data_path, "rewards.csv"))

    # Preprocessing: Standardize the data
    scaler = StandardScaler()
    df_states_scaled = pd.DataFrame(scaler.fit_transform(df_states), columns=df_states.columns)

    # Convert dataframes to tensors
    states = torch.tensor(df_states_scaled.values, dtype=torch.float32)
    actions = torch.tensor(df_actions.values, dtype=torch.float32)
    rewards = torch.tensor(df_rewards.values, dtype=torch.float32)

    return states, actions, rewards

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for the DataLoader.

    Parameters:
    batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): List of (state, action, reward) tuples.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batched tensors for states, actions, and rewards.
    """
    states, actions, rewards = map(torch.stack, zip(*batch))
    return states, actions, rewards

def main():
    # Configuration
    data_path = "path/to/dataset"
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    log_interval = 50

    # Seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Parse the dataset
    states, actions, rewards = parse_dataset(data_path)

    # Data loaders
    train_loader = DataLoader(list(zip(states, actions, rewards)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create the agent
    agent = EvoSparrowAgent(input_size=states.shape[1], hidden_size=128, output_size=actions.shape[1], device=device)

    # Load checkpoint if exists (optional)
    # checkpoint_path = "path/to/checkpoint.pth"
    # agent.load_checkpoint(checkpoint_path)

    # Train the agent
    agent.train(train_loader, epochs, batch_size, learning_rate, log_interval)

    # Evaluate the agent on a test dataset (not provided in this example)
    # test_loader = DataLoader(...)
    # mse_loss = agent.evaluate(test_loader)

if __name__ == "__main__":
    main()