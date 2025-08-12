import logging
import threading
import time
import random
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiAgentCommunicator:
    """
    Multi-agent communicator class for facilitating communication and coordination between multiple agents.
    """
    def __init__(self, agent_ids: List[str], communication_range: float):
        self.agent_ids = agent_ids
        self.communication_range = communication_range
        self.agent_locations = {}  # Dict to store agent locations
        self.agent_data = {}  # Dict to store agent data
        self.communication_locks = {}  # Dict to store locks for thread safety
        self.message_queue = [] # Queue to store incoming messages
        self.message_lock = threading.Lock() # Lock for thread safety
        self.agent_locks = {agent_id: threading.Lock() for agent_id in agent_ids} # Locks for each agent for thread safety
        self.data_received = False # Flag to track data reception

    def update_agent_location(self, agent_id: str, location: Tuple[float, float]):
        """
        Updates the location of an agent.

        Args:
            agent_id (str): ID of the agent.
            location (Tuple[float, float]): New location of the agent.
        """
        with self.agent_locks[agent_id]:
            self.agent_locations[agent_id] = location

    def share_data_with_agents(self, sending_agent_id: str, data: Dict):
        """
        Shares data from one agent to other agents within the communication range.

        Args:
            sending_agent_id (str): ID of the agent sending the data.
            data (Dict): Data to be shared with other agents.
        """
        sending_agent_location = self.agent_locations[sending_agent_id]
        for agent_id, agent_location in self.agent_locations.items():
            if agent_id != sending_agent_id and self._is_within_range(sending_agent_location, agent_location, self.communication_range):
                with self.agent_locks[agent_id]:
                    self.agent_data[agent_id] = data
                    logging.info(f"Data shared with agent {agent_id} at location {agent_location}")

    def retrieve_data_from_agents(self, receiving_agent_id: str):
        """
        Retrieves data received by an agent from other agents within the communication range.

        Args:
            receiving_agent_id (str): ID of the agent retrieving the data.

        Returns:
            Dict: Data received from other agents.
        """
        with self.agent_locks[receiving_agent_id]:
            if receiving_agent_id in self.agent_data:
                data = self.agent_data[receiving_agent_id]
                self.agent_data.pop(receiving_agent_id)
                return data
            else:
                return None

    def _is_within_range(self, location1: Tuple[float, float], location2: Tuple[float, float], range: float) -> bool:
        """
        Checks if two locations are within a given range.

        Args:
            location1 (Tuple[float, float]): First location.
            location2 (Tuple[float, float]): Second location.
            range (float): Communication range.

        Returns:
            bool: True if the locations are within range, False otherwise.
        """
        distance = ((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2) ** 0.5
        return distance <= range

    def _handle_message(self, message: Dict):
        """
        Handles an incoming message from another agent and shares data if the sender is within communication range.

        Args:
            message (Dict): Message containing sender ID and data.
        """
        sender_id = message['sender_id']
        data = message['data']
        sender_location = self.agent_locations[sender_id]
        for agent_id, agent_location in self.agent_locations.items():
            if agent_id != sender_id and self._is_within_range(sender_location, agent_location, self.communication_range):
                with self.agent_locks[agent_id]:
                    self.agent_data[agent_id] = data
                    logging.info(f"Data received from agent {sender_id} and shared with agent {agent_id}")

    def receive_message(self, message: Dict):
        """
        Receives a message from another agent, queues it, and handles it asynchronously.

        Args:
            message (Dict): Message containing sender ID and data.
        """
        with self.message_lock:
            self.message_queue.append(message)

    def process_messages(self):
        """
        Processes queued messages asynchronously.
        """
        while True:
            if self.message_queue:
                with self.message_lock:
                    message = self.message_queue.pop(0)
                self._handle_message(message)
            else:
                time.sleep(0.1)

class Agent:
    """
    Base agent class with communication capabilities.
    """
    def __init__(self, agent_id: str, communicator: MultiAgentCommunicator, location: Tuple[float, float]):
        self.agent_id = agent_id
        self.communicator = communicator
        self.location = location
        self.communicator.update_agent_location(self.agent_id, self.location)
        self.data = None

    def send_data(self, data: Dict):
        """
        Sends data to other agents within communication range.

        Args:
            data (Dict): Data to be sent.
        """
        message = {'sender_id': self.agent_id, 'data': data}
        self.communicator.share_data_with_agents(self.agent_id, data)
        logging.info(f"Data sent by agent {self.agent_id} at location {self.location}")

    def receive_data(self):
        """
        Receives data from other agents within communication range.

        Returns:
            Dict: Data received from other agents.
        """
        self.data = self.communicator.retrieve_data_from_agents(self.agent_id)
        return self.data

class DataReceiverThread(threading.Thread):
    """
    Thread class for receiving data asynchronously.
    """
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def run(self):
        """
        Runs the thread to continuously receive data.
        """
        while True:
            time.sleep(random.uniform(0.1, 0.5))
            self.agent.receive_data()

# Example usage
if __name__ == "__main__":
    agent_ids = ['Agent1', 'Agent2', 'Agent3']
    communication_range = 10.0

    # Create communicator
    communicator = MultiAgentCommunicator(agent_ids, communication_range)

    # Create agents
    agent1 = Agent(agent_ids[0], communicator, (0, 0))
    agent2 = Agent(agent_ids[1], communicator, (5, 5))
    agent3 = Agent(agent_ids[2], communicator, (12, 0))

    # Start data receiver threads
    receiver_thread1 = DataReceiverThread(agent1)
    receiver_thread2 = DataReceiverThread(agent2)
    receiver_thread3 = DataReceiverThread(agent3)
    receiver_thread1.start()
    receiver_thread2.start()
    receiver_thread3.start()

    # Simulate agent communication
    agent1.send_data({'message': 'Hello from Agent 1!'})
    agent2.send_data({'message': 'Hello from Agent 2!'})
    agent3.send_data({'message': 'Hello from Agent 3!'})

    # Retrieve received data
    received_data1 = agent1.receive_data()
    received_data2 = agent2.receive_data()
    received_data3 = agent3.receive_data()

    # Print received data
    if received_data1:
        logging.info(f"Agent 1 received data: {received_data1}")
    if received_data2:
        logging.info(f"Agent 2 received data: {received_data2}")
    if received_data3:
        logging.info(f"Agent 3 received data: {received_data3}")

    # Clean up threads
    receiver_thread1.join()
    receiver_thread2.join()
    receiver_thread3.join()