import socket
import math
import torch
import numpy as np
import logging
import json
import os
import csv
import struct
import time 
from datetime import datetime
from sac_torch import SACAgent
from scipy.special import gamma 
import CANVAS_VANET_UNTESTED as vanet_env
from utils import plot_training_summary

# --- Configuration for Logging ---
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_RECEIVED_PATH = os.path.join(LOG_DIR, 'receive_data.log')
LOG_SENT_PATH = os.path.join(LOG_DIR, 'sent_data.log')
LOSS_LOG_PATH = os.path.join(LOG_DIR, 'loss_history.csv')
REWARD_LOG_PATH = os.path.join(LOG_DIR, 'rewards_history.csv')
BEST_PERFORMANCE_LOG_PATH = os.path.join(LOG_DIR, 'best_episode_performance.csv')


# --- Global Constants -------------------------------------------------------------------------------------
c = 3e8 # Speed of light in meters/second
MIN_TX_POWER_DBM = 10.0 
MAX_TX_POWER_DBM = 33.0 
MIN_MCS_INDEX = 0
MAX_MCS_INDEX = 9 
agent = SACAgent()
ieee_mapper = vanet_env.IEEE80211bdMapper()
# ----------------------------------------------------------------------------------------------------------

# --- NEW: Clean up previous log files for a fresh run ---
log_files_to_clean = [REWARD_LOG_PATH, LOSS_LOG_PATH, BEST_PERFORMANCE_LOG_PATH]
for log_file in log_files_to_clean:
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            logging.info(f"Removed previous log file: {log_file}")
        except OSError as e:
            logging.error(f"Error removing file {log_file}: {e}")

# --- Logging Functions ---
def log_data(log_path, data):
    """Appends data with a timestamp to a specified log file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")

def write_best_performance_metrics(performance_data, file_path=BEST_PERFORMANCE_LOG_PATH):
    """
    Overwrites the specified file with the performance data from the best episode.
    """
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'VehicleID', 'CBR', 'SNR', 'Strongest Neighbor','Strongest Neighbor MCS', 'Throughput', 'Latency', 'PER', 'PDR', 'Reward', 'Power_dBm', 'MCS','cbr_term', 'sinr_term', 'rate_term'])
            writer.writerows(performance_data)
        logging.info(f"Successfully wrote best performance metrics to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write best performance metrics: {e}")


def write_loss_history(step, actor_loss, critic_loss, alpha_loss, file_path=LOSS_LOG_PATH):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Step', 'Actor_Loss', 'Critic_Loss', 'Alpha_Loss'])
        writer.writerow([step, actor_loss, critic_loss, alpha_loss])

def write_reward_history(episode, avg_reward, file_path=REWARD_LOG_PATH):
    """Logs the average reward for a completed episode."""
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Episode', 'Average_Reward'])
        writer.writerow([episode, avg_reward])

# --- Helper Functions ---
def get_data_rate_and_snr_threshold(mcs_index):
    mcs_index = int(np.clip(round(mcs_index), MIN_MCS_INDEX, MAX_MCS_INDEX))
    mcs_config = ieee_mapper.mcs_table.get(mcs_index, ieee_mapper.mcs_table[0])
    snr_thresholds = ieee_mapper.snr_thresholds.get(mcs_index, ieee_mapper.snr_thresholds[0])
    data_rate = mcs_config['data_rate']
    snr_success_threshold = snr_thresholds['success']
    return data_rate, snr_success_threshold

def calculate_carrier_sense_range(p_dBm, mcs_index):
    _, required_snr_db = get_data_rate_and_snr_threshold(mcs_index)
    S_r_dBm = required_snr_db + vanet_env.SimulationConfig.interference_threshold_db
    beta = vanet_env.SimulationConfig.path_loss_exponent
    p_linear = 10**(p_dBm/10) * 1e-3
    S_linear = 10**(S_r_dBm/10) * 1e-3
    lambda_val = c / vanet_env.SimulationConfig.frequency
    A = (4 * np.pi / lambda_val)**2
    m = 1
    gamma_term1 = gamma(m + 1/beta)
    gamma_term2 = gamma(m)
    try:
        r_CS = (gamma_term1 / gamma_term2) * (S_linear * A * m / p_linear)**(-1/beta)
    except (ZeroDivisionError, OverflowError):
        r_CS = 0.0
    r_CS = max(0.0, r_CS)
    return r_CS

def calculate_vehicle_density(num_neighbors, power_dBm, mcs_index): 
    r_cs = calculate_carrier_sense_range(power_dBm, mcs_index)
    rho = num_neighbors / (2 * r_cs) if r_cs > 0 else 0
    return rho

def calculate_full_reward(CBR, sinr, mcs_index, strongest_neighbor_mcs): 
    # Weights for Reward Components
    MBL = 0.65
    omega_c = 2
    omega_s = 0.25
    omega_d = 0.1
    omega_e = 0.8

    # =================================================================
    # 1. CBR Term: Steadier climb, bonus zone, and heavy penalty
    # =================================================================
    cbr_term = 0.0
    error = CBR - MBL
    # positive at CBR = 0
    base_cbr = MBL - abs(CBR - MBL)
    if CBR > (MBL + 0.05):
        base_cbr -= 0.6 # This is a penalty for going too high, above 0.7 degrades faster
    cbr_term = omega_c * base_cbr

    # =================================================================
    # 2. SINR Term: A multi-objective approach to power control
    # =================================================================
    sinr_term = 0.0

    # Get the required SNR for the neighbor's MCS
    _, snr_required = get_data_rate_and_snr_threshold(strongest_neighbor_mcs)
    base_sinr = -abs(sinr - snr_required)
    if 0 <= sinr - snr_required:
        base_sinr = (2 ** (-(sinr - snr_required))) * 6
    sinr_term = omega_s * base_sinr
    # print(f"SNR Required for MCS {strongest_neighbor_mcs}: {snr_required} dB, Current SINR: {sinr} dB, Base SINR: {base_sinr}")

    # =================================================================
    # 3. Data Rate Term: Directly rewarding higher data rates
    # =================================================================
    data_rate, _ = get_data_rate_and_snr_threshold(mcs_index)
    
    rate_term = 0.0
    rate_term = omega_d * (data_rate ** omega_e)
    # =================================================================
    # Final Reward Calculation
    # =================================================================
    reward = cbr_term + sinr_term - rate_term
    components = {
        'cbr_term': cbr_term,
        'sinr_term': sinr_term,
        'rate_term': rate_term
    }
    final_reward = reward if np.isfinite(reward) else 0.0
    return float(final_reward), components

# --- Main Training Loop ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

checkpoint_dir = 'model'
os.makedirs(checkpoint_dir, exist_ok=True)

try:
    agent.load_models()
    logging.info("SAC models loaded successfully from default directory.")
except Exception as e:
    logging.warning(f"Failed to load models, starting with a newly initialized model. Error: {e}")

RL_SERVER_HOST = '127.0.0.1'
RL_SERVER_PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((RL_SERVER_HOST, RL_SERVER_PORT))
server.listen(vanet_env.NUMBER_OF_RUNS)
logging.info(f"Listening for vanet_env connection on {RL_SERVER_HOST}:{RL_SERVER_PORT}...")


best_score = -np.inf
learn_step_counter = 0
scores_history = []
# --- REMOVED: StateNormalizer is no longer used ---

# --- MODIFIED: Flag to control the main training loop ---
training_failed = False

# Main loop over episodes
for i_episode in range (1, vanet_env.NUMBER_OF_RUNS + 1):
    logging.info(f"Waiting for vanet_env connection for episode {i_episode} on {RL_SERVER_HOST}:{RL_SERVER_PORT}...")
    conn, addr = server.accept()
    logging.info(f"Environment simulator connected from {addr} for episode {i_episode}")
    
    episode_rewards = []
    episode_performance_data = []
    prev_states_per_veh = {}
    prev_actions_per_veh = {}

    logging.info(f"--- Starting Episode {i_episode}/{vanet_env.NUMBER_OF_RUNS} ---")

    while True:
        try:
            if os.path.exists("run.complete"):
                print(f"Episode {i_episode} is marked as done")
                print(f"Last Timestamp: {current_timestamp_sim}")
                break

            length_header = conn.recv(4)
            if not length_header:
                logging.info("Simulator has closed the connection. Ending training.") 
                break
            
            msg_length = struct.unpack('<I', length_header)[0]
            
            data = b''
            bytes_recd = 0
            while bytes_recd < msg_length:
                chunk = conn.recv(min(msg_length - bytes_recd, 4096))
                if not chunk:
                    raise socket.error("Socket connection broken during payload reception")
                data += chunk
                bytes_recd += len(chunk)

            batch_data = json.loads(data.decode('utf-8'))
            
            current_timestamp_sim = None
            responses = {}

            for veh_id, vehicle_info in batch_data.items():
                current_timestamp_sim = vehicle_info['timestamp']

                strongest_neighbor_mcs = vehicle_info.get('strongest_neighbor_mcs', -1)
                strongest_neighbor_id = vehicle_info.get('strongest_neighbor_id', 'None')
                
                current_power_tx = float(np.clip(vehicle_info.get('transmissionPower', MIN_TX_POWER_DBM), MIN_TX_POWER_DBM, MAX_TX_POWER_DBM))
                current_mcs = int(np.clip(round(vehicle_info.get('MCS', MIN_MCS_INDEX)), MIN_MCS_INDEX, MAX_MCS_INDEX))
                current_cbr = float(np.clip(vehicle_info.get('CBR', 0.0), 0.0, 1.0))
                current_sinr = float(np.clip(vehicle_info.get('SINR', -20.0), -20.0, 50.0))
                current_neighbors = int(np.clip(vehicle_info.get('neighbors', 0), 0, 100))
                vehicle_density = calculate_vehicle_density(current_neighbors, current_power_tx, current_mcs)
                
                # --- MODIFIED: Use the raw state directly ---
                current_state_raw = np.array([current_power_tx, current_mcs, current_cbr, vehicle_density, current_sinr], dtype=np.float32)

                if veh_id in prev_states_per_veh and prev_states_per_veh[veh_id] is not None:
                    # Retrieve previous state and action
                    prev_state_raw = prev_states_per_veh[veh_id]
                    prev_action = prev_actions_per_veh[veh_id]
                    
                    applied_prev_power = (prev_action[0] + 1) * (MAX_TX_POWER_DBM - MIN_TX_POWER_DBM) / 2 + MIN_TX_POWER_DBM
                    applied_prev_mcs = round((prev_action[1] + 1) * (MAX_MCS_INDEX - MIN_MCS_INDEX) / 2 + MIN_MCS_INDEX)
                    
                    reward, reward_components = calculate_full_reward(CBR=current_cbr, mcs_index=applied_prev_mcs, sinr=current_sinr, strongest_neighbor_mcs=strongest_neighbor_mcs)
                    
                    episode_rewards.append(reward)
                    
                    # --- MODIFIED: Pass raw states to the agent's memory ---
                    agent.remember(prev_state_raw, prev_actions_per_veh[veh_id], reward, current_state_raw, os.path.exists("run.complete"))
                    
                    performance_row = [
                        current_timestamp_sim, veh_id, current_cbr, current_sinr, strongest_neighbor_id, strongest_neighbor_mcs,
                        vehicle_info.get('throughput', 0.0), vehicle_info.get('latency', 0.0),
                        vehicle_info.get('PER', 0.0), vehicle_info.get('PDR', 0.0), reward,
                        applied_prev_power, applied_prev_mcs,
                        reward_components.get('cbr_term', 0),
                        reward_components.get('sinr_term', 0),
                        reward_components.get('rate_term', 0)
                    ]
                    episode_performance_data.append(performance_row)

                # --- MODIFIED: Choose action based on the raw state ---
                action_raw = agent.choose_action(current_state_raw)

                prev_states_per_veh[veh_id] = current_state_raw
                prev_actions_per_veh[veh_id] = action_raw
                
                new_power = float(np.clip((action_raw[0] + 1) * (MAX_TX_POWER_DBM - MIN_TX_POWER_DBM) / 2 + MIN_TX_POWER_DBM, MIN_TX_POWER_DBM, MAX_TX_POWER_DBM))
                new_mcs = int(np.clip(round((action_raw[1] + 1) * (MAX_MCS_INDEX - MIN_MCS_INDEX) / 2 + MIN_MCS_INDEX), MIN_MCS_INDEX, MAX_MCS_INDEX))
                new_beacon_rate = float(np.clip(vehicle_info.get('beaconRate', 10.0), 1.0, 20.0))

                responses[veh_id] = {"transmissionPower": new_power, "beaconRate": new_beacon_rate, "MCS": new_mcs}

            if agent.memory.mem_cntr >= agent.batch_size:
                actor_loss, critic_loss, alpha_loss = agent.learn()
                if actor_loss is not None: # All three will be None or a number
                    learn_step_counter += 1
                    write_loss_history(learn_step_counter, actor_loss, critic_loss, alpha_loss)
            
            response_to_send = {"vehicles": responses}
            response_data = json.dumps(response_to_send).encode('utf-8')
            response_length = len(response_data)
            length_header = struct.pack('<I', response_length)
            conn.sendall(length_header)
            conn.sendall(response_data)

        except (socket.error, json.JSONDecodeError, Exception) as e:
            logging.error(f"Error during episode {i_episode}: {e}", exc_info=True)
            training_failed = True
            break
        
    conn.close()
    logging.info(f"Connection for episode {i_episode} closed.")

    if training_failed:
        break # Exit the main 'for' loop

    if i_episode <= vanet_env.NUMBER_OF_RUNS:
        score = np.mean(episode_rewards) if episode_rewards else 0.0
        scores_history.append(score)
        
        logging.info(f"Episode {i_episode} | Final Score: {score:.2f} | Best score so far: {best_score:.2f}")
        write_reward_history(i_episode, score)

        if score > best_score:
            best_score = score
            agent.save_models()
            plot_output_dir = 'plots'
            best_csv_path = os.path.join(LOG_DIR, f'best_episode_{i_episode}_performance.csv')
            plot_filename = os.path.join(plot_output_dir, f'best_episode_{i_episode}_summary.png')
            logging.info(f"*** New best model saved with score: {best_score:.2f} at episode {i_episode} ***")
            write_best_performance_metrics(episode_performance_data, file_path=best_csv_path)
            plot_training_summary(
                performance_csv_path=best_csv_path, 
                loss_csv_path=LOSS_LOG_PATH, 
                figure_file=plot_filename
            )
        
        # Wait for the simulator to delete the flag file to start the next run
        logging.info("Waiting for simulator to begin the next episode...")
        while os.path.exists("run.complete"):
            time.sleep(1)

# --- End of all episodes ---
logging.info("Training finished.")
server.close()
logging.info("Server closed.")