import logging
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
from config import Config
from utils import get_logger, get_config

# Set up logging
logger = get_logger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class implements the reward calculation and shaping algorithms
    described in the research paper. It uses the velocity-threshold and
    Flow Theory algorithms to calculate rewards.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config (Config): Configuration object.
        """
        self.config = config
        self.velocity_threshold = config.velocity_threshold
        self.flow_threshold = config.flow_threshold
        self.velocity_window = config.velocity_window
        self.flow_window = config.flow_window

    def calculate_velocity_reward(self, velocity: float) -> float:
        """
        Calculate the velocity reward.

        The velocity reward is calculated using the velocity-threshold
        algorithm. If the velocity is above the threshold, the reward
        is 1. Otherwise, the reward is 0.

        Args:
            velocity (float): Velocity value.

        Returns:
            float: Velocity reward.
        """
        if velocity > self.velocity_threshold:
            return 1.0
        else:
            return 0.0

    def calculate_flow_reward(self, flow: float) -> float:
        """
        Calculate the flow reward.

        The flow reward is calculated using the Flow Theory algorithm.
        The reward is calculated as the difference between the current
        flow and the average flow over the window.

        Args:
            flow (float): Flow value.

        Returns:
            float: Flow reward.
        """
        if self.flow_window > 0:
            avg_flow = np.mean(self.config.flow_values[-self.flow_window:])
            reward = flow - avg_flow
            return reward
        else:
            return 0.0

    def calculate_reward(self, velocity: float, flow: float) -> float:
        """
        Calculate the overall reward.

        The overall reward is a weighted sum of the velocity and flow
        rewards.

        Args:
            velocity (float): Velocity value.
            flow (float): Flow value.

        Returns:
            float: Overall reward.
        """
        velocity_reward = self.calculate_velocity_reward(velocity)
        flow_reward = self.calculate_flow_reward(flow)
        reward = self.config.velocity_weight * velocity_reward + self.config.flow_weight * flow_reward
        return reward

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward.

        The reward is shaped using the Flow Theory algorithm. The
        reward is clipped to a maximum value and then scaled to a
        range of [0, 1].

        Args:
            reward (float): Reward value.

        Returns:
            float: Shaped reward.
        """
        if reward > self.config.max_reward:
            reward = self.config.max_reward
        elif reward < self.config.min_reward:
            reward = self.config.min_reward
        reward = (reward - self.config.min_reward) / (self.config.max_reward - self.config.min_reward)
        return reward

class RewardSystemConfig:
    """
    Reward system configuration.

    This class represents the configuration for the reward system.
    It contains the velocity threshold, flow threshold, velocity
    window, flow window, velocity weight, flow weight, maximum
    reward, and minimum reward.
    """

    def __init__(self):
        """
        Initialize the reward system configuration.
        """
        self.velocity_threshold = 0.5
        self.flow_threshold = 0.5
        self.velocity_window = 10
        self.flow_window = 10
        self.velocity_weight = 0.5
        self.flow_weight = 0.5
        self.max_reward = 1.0
        self.min_reward = 0.0
        self.flow_values = []

def get_reward_system(config: Config) -> RewardSystem:
    """
    Get the reward system.

    Args:
        config (Config): Configuration object.

    Returns:
        RewardSystem: Reward system object.
    """
    reward_system = RewardSystem(config)
    return reward_system

def main():
    # Get the configuration
    config = get_config()

    # Get the reward system
    reward_system = get_reward_system(config)

    # Calculate the reward
    velocity = 0.7
    flow = 0.8
    reward = reward_system.calculate_reward(velocity, flow)
    shaped_reward = reward_system.shape_reward(reward)

    # Log the reward
    logger.info(f"Reward: {reward}")
    logger.info(f"Shaped Reward: {shaped_reward}")

if __name__ == "__main__":
    main()