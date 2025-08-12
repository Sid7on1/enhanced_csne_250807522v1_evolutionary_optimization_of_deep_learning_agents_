import logging
import random
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import deque
from abc import ABC, abstractmethod
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, config: Config):
        self.config = config
        self.memory_size = config.memory_size
        self.experience_replay_buffer = deque(maxlen=self.memory_size)
        self.current_experience_index = 0

    def add_experience(self, experience: Dict):
        self.experience_replay_buffer.append(experience)
        self.current_experience_index += 1

    def get_random_experience(self) -> Dict:
        if self.current_experience_index < self.memory_size:
            return random.choice(list(self.experience_replay_buffer))
        else:
            return random.choice(self.experience_replay_buffer)

    def get_experiences(self, batch_size: int) -> List[Dict]:
        if self.current_experience_index < batch_size:
            return random.sample(list(self.experience_replay_buffer), self.current_experience_index)
        else:
            return random.sample(list(self.experience_replay_buffer), batch_size)

    def get_experiences_with_indices(self, batch_size: int) -> Tuple[List[Dict], List[int]]:
        if self.current_experience_index < batch_size:
            experiences = random.sample(list(self.experience_replay_buffer), self.current_experience_index)
            indices = list(range(self.current_experience_index))
        else:
            experiences = random.sample(list(self.experience_replay_buffer), batch_size)
            indices = list(range(batch_size))
        return experiences, indices

    def get_experiences_with_indices_and_states(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict]]:
        if self.current_experience_index < batch_size:
            experiences = random.sample(list(self.experience_replay_buffer), self.current_experience_index)
            indices = list(range(self.current_experience_index))
            states = [experience['state'] for experience in experiences]
        else:
            experiences = random.sample(list(self.experience_replay_buffer), batch_size)
            indices = list(range(batch_size))
            states = [experience['state'] for experience in experiences]
        return experiences, indices, states

    def get_experiences_with_indices_and_states_and_actions(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict], List[Dict]]:
        if self.current_experience_index < batch_size:
            experiences = random.sample(list(self.experience_replay_buffer), self.current_experience_index)
            indices = list(range(self.current_experience_index))
            states = [experience['state'] for experience in experiences]
            actions = [experience['action'] for experience in experiences]
        else:
            experiences = random.sample(list(self.experience_replay_buffer), batch_size)
            indices = list(range(batch_size))
            states = [experience['state'] for experience in experiences]
            actions = [experience['action'] for experience in experiences]
        return experiences, indices, states, actions

    def get_experiences_with_indices_and_states_and_actions_and_rewards(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict], List[Dict], List[float]]:
        if self.current_experience_index < batch_size:
            experiences = random.sample(list(self.experience_replay_buffer), self.current_experience_index)
            indices = list(range(self.current_experience_index))
            states = [experience['state'] for experience in experiences]
            actions = [experience['action'] for experience in experiences]
            rewards = [experience['reward'] for experience in experiences]
        else:
            experiences = random.sample(list(self.experience_replay_buffer), batch_size)
            indices = list(range(batch_size))
            states = [experience['state'] for experience in experiences]
            actions = [experience['action'] for experience in experiences]
            rewards = [experience['reward'] for experience in experiences]
        return experiences, indices, states, actions, rewards

    def get_experiences_with_indices_and_states_and_actions_and_rewards_and_next_states(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict], List[Dict], List[float], List[Dict]]:
        if self.current_experience_index < batch_size:
            experiences = random.sample(list(self.experience_replay_buffer), self.current_experience_index)
            indices = list(range(self.current_experience_index))
            states = [experience['state'] for experience in experiences]
            actions = [experience['action'] for experience in experiences]
            rewards = [experience['reward'] for experience in experiences]
            next_states = [experience['next_state'] for experience in experiences]
        else:
            experiences = random.sample(list(self.experience_replay_buffer), batch_size)
            indices = list(range(batch_size))
            states = [experience['state'] for experience in experiences]
            actions = [experience['action'] for experience in experiences]
            rewards = [experience['reward'] for experience in experiences]
            next_states = [experience['next_state'] for experience in experiences]
        return experiences, indices, states, actions, rewards, next_states

class Experience:
    def __init__(self, state: Dict, action: Dict, reward: float, next_state: Dict):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class ExperienceReplayBuffer:
    def __init__(self, config: Config):
        self.config = config
        self.memory_size = config.memory_size
        self.experience_replay_buffer = deque(maxlen=self.memory_size)

    def add_experience(self, experience: Experience):
        self.experience_replay_buffer.append(experience)

    def get_random_experience(self) -> Experience:
        return random.choice(list(self.experience_replay_buffer))

    def get_experiences(self, batch_size: int) -> List[Experience]:
        return random.sample(list(self.experience_replay_buffer), batch_size)

    def get_experiences_with_indices(self, batch_size: int) -> Tuple[List[Experience], List[int]]:
        experiences = random.sample(list(self.experience_replay_buffer), batch_size)
        indices = list(range(batch_size))
        return experiences, indices

    def get_experiences_with_indices_and_states(self, batch_size: int) -> Tuple[List[Experience], List[int], List[Dict]]:
        experiences = random.sample(list(self.experience_replay_buffer), batch_size)
        indices = list(range(batch_size))
        states = [experience.state for experience in experiences]
        return experiences, indices, states

    def get_experiences_with_indices_and_states_and_actions(self, batch_size: int) -> Tuple[List[Experience], List[int], List[Dict], List[Dict]]:
        experiences = random.sample(list(self.experience_replay_buffer), batch_size)
        indices = list(range(batch_size))
        states = [experience.state for experience in experiences]
        actions = [experience.action for experience in experiences]
        return experiences, indices, states, actions

    def get_experiences_with_indices_and_states_and_actions_and_rewards(self, batch_size: int) -> Tuple[List[Experience], List[int], List[Dict], List[Dict], List[float]]:
        experiences = random.sample(list(self.experience_replay_buffer), batch_size)
        indices = list(range(batch_size))
        states = [experience.state for experience in experiences]
        actions = [experience.action for experience in experiences]
        rewards = [experience.reward for experience in experiences]
        return experiences, indices, states, actions, rewards

    def get_experiences_with_indices_and_states_and_actions_and_rewards_and_next_states(self, batch_size: int) -> Tuple[List[Experience], List[int], List[Dict], List[Dict], List[float], List[Dict]]:
        experiences = random.sample(list(self.experience_replay_buffer), batch_size)
        indices = list(range(batch_size))
        states = [experience.state for experience in experiences]
        actions = [experience.action for experience in experiences]
        rewards = [experience.reward for experience in experiences]
        next_states = [experience.next_state for experience in experiences]
        return experiences, indices, states, actions, rewards, next_states

class MemoryManager:
    def __init__(self, config: Config):
        self.config = config
        self.memory = Memory(config)
        self.experience_replay_buffer = ExperienceReplayBuffer(config)

    def add_experience(self, experience: Experience):
        self.memory.add_experience(experience.__dict__)
        self.experience_replay_buffer.add_experience(experience)

    def get_random_experience(self) -> Dict:
        return self.memory.get_random_experience()

    def get_experiences(self, batch_size: int) -> List[Dict]:
        return self.memory.get_experiences(batch_size)

    def get_experiences_with_indices(self, batch_size: int) -> Tuple[List[Dict], List[int]]:
        return self.memory.get_experiences_with_indices(batch_size)

    def get_experiences_with_indices_and_states(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict]]:
        return self.memory.get_experiences_with_indices_and_states(batch_size)

    def get_experiences_with_indices_and_states_and_actions(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict], List[Dict]]:
        return self.memory.get_experiences_with_indices_and_states_and_actions(batch_size)

    def get_experiences_with_indices_and_states_and_actions_and_rewards(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict], List[Dict], List[float]]:
        return self.memory.get_experiences_with_indices_and_states_and_actions_and_rewards(batch_size)

    def get_experiences_with_indices_and_states_and_actions_and_rewards_and_next_states(self, batch_size: int) -> Tuple[List[Dict], List[int], List[Dict], List[Dict], List[float], List[Dict]]:
        return self.memory.get_experiences_with_indices_and_states_and_actions_and_rewards_and_next_states(batch_size)