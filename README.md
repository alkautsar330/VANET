# **SAC for VANET V2V Communication Optimization**

A PyTorch implementation of the Soft Actor-Critic (SAC) algorithm to optimize **Data Rate** and **Power Transmission**, tested on the [CANVAS-VANET](https://github.com/galihnnk/CANVAS-VANET/tree/main) simulation environment.

This repository provides the code for training and evaluating the SAC agent.

## **Installation**

To get started, clone the repository and install the required dependencies.

1. **Clone the repository:**
   ```
   git clone https://github.com/alkautsar330/VANET.git  
   cd VANET
   ```  
3. **Create a virtual environment (recommended):**
   ```
   python -m venv venv  
   source venv/bin/activate  # On Windows use `venv\Scripts\activate\`
   ```
5. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   *(Note: This project requires PyTorch, NumPy, and Matplotlib).*

## **Usage**

   The agent interacts with the CANVAS-VANET simulator, which acts as the environment. You must start the appropriate Python script *before* running the simulation.

### **Training**

   To train a new agent or continue training a saved one, run the `train.py` script. This script will start a server and wait for the VANET simulator to connect.
   ```
   python train.py
```
* The script will automatically save the model with the best performance in the` model/` directory.  
* Training logs (rewards, losses) are saved in the `logs/` directory.  
* Performance plots for the best episode are saved in the `plots/` directory.

### **Testing**

  To evaluate a trained agent, run the `test_agent.py` script. This script loads the saved models from the `model/` directory and runs the agent in a deterministic evaluation mode.
  ```
  python test_agent.py
```
* The script will wait for the VANET simulator to connect.  
* Performance metrics from the test run are saved as a CSV file in the `logs_test/` directory.  
* A summary plot of the test run is saved in the `plots_test/` directory.

## **Project Structure**
   ```
  .  
  ├── model/                  # Saved model checkpoints  
  ├── logs/                   # Training logs (rewards, losses, performance)  
  ├── plots/                  # Output plots for best training episodes  
  ├── logs_test/              # Logs from test runs  
  ├── plots_test/             # Output plots from test runs  
  ├── sac_torch.py            # Core SAC agent and network implementation  
  ├── train.py                # Main script for training the agent  
  ├── test_agent.py           # Script for evaluating a trained agent  
  ├── utils.py                # Utility functions for plotting results  
  └── README.md               # This file
   ```
### **File Descriptions:**

* `sac_torch.py`: Contains the core implementation of the Soft Actor-Critic (SAC) agent. This includes the `ActorNetwork` and `CriticNetwork` architectures (the "brain" of the RL agent), the `ReplayBuffer` for storing experiences, and the main `SACAgent` class that orchestrates the learning process.  
* `train.py`: The main script for training the agent. It initializes the agent and starts a server to manage the interaction with the CANVAS-VANET environment. It handles receiving state information, calculating rewards, sending actions, and triggering the agent's learning steps. It also manages all logging for training.  
* `test_agent.py`: Used to evaluate a pre-trained agent. This script loads a saved model and runs it in the environment deterministically (i.e., without exploration noise) to measure its performance.  
* `utils.py`: Contains utility functions, most notably `plot_training_summary`, which reads the CSV logs generated during training or testing and creates summary plots of key performance indicators (like CBR, SINR, throughput) and agent losses.
