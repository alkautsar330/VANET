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
# --- MODIFIED: Create a separate directory for test results ---
LOG_DIR = 'logs_test' 
os.makedirs(LOG_DIR, exist_ok=True)
TEST_PERFORMANCE_LOG_PATH = os.path.join(LOG_DIR, 'test_episode_performance.csv')


# --- Global Constants -------------------------------------------------------------------------------------
c = 3e8 # Speed of light in meters/second
MIN_TX_POWER_DBM = 10.0
MAX_TX_POWER_DBM = 33.0
MIN_MCS_INDEX = 0
MAX_MCS_INDEX = 9
agent = SACAgent() # Assumes SACAgent can be instantiated without all training params
ieee_mapper = vanet_env.IEEE80211bdMapper()
# ----------------------------------------------------------------------------------------------------------

# --- MODIFIED: Simplified logging for test performance ---
def write_test_performance_metrics(performance_data, file_path=TEST_PERFORMANCE_LOG_PATH):
    """
    Overwrites the specified file with the performance data from the test episode.
    """
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'VehicleID', 'CBR', 'SNR', 'Strongest Neighbor','Strongest Neighbor MCS', 'Throughput', 'Latency', 'PER', 'PDR', 'Power_dBm', 'MCS'])
            writer.writerows(performance_data)
        logging.info(f"Successfully wrote test performance metrics to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write test performance metrics: {e}")


# --- REMOVED: Loss and reward history functions are not needed for testing ---

# --- Helper Functions (get_data_rate_and_snr_threshold, calculate_carrier_sense_range, etc.) ---
# (Keep all helper functions as they are, they are needed for state calculation)
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
# --- Main Testing Logic ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

checkpoint_dir = 'model' # Directory where models are saved

try:
    # --- CRITICAL: Load the trained models ---
    agent.load_models()
    logging.info("Trained SAC models loaded successfully for testing.")
except Exception as e:
    logging.error(f"FATAL: Failed to load models from '{checkpoint_dir}'. Cannot run test. Error: {e}")
    exit() # Exit if models can't be loaded

RL_SERVER_HOST = '127.0.0.1'
RL_SERVER_PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((RL_SERVER_HOST, RL_SERVER_PORT))
server.listen(1) # Listen for a single test run
logging.info(f"Starting Agent for TESTING. Listening on {RL_SERVER_HOST}:{RL_SERVER_PORT}...")

# --- MODIFIED: Simplified loop for a single test episode ---
logging.info(f"Waiting for vanet_env connection for a test run...")
conn, addr = server.accept()
logging.info(f"Environment simulator connected from {addr} for testing.")

test_performance_data = []

try:
    while True:
        if os.path.exists("run.complete"):
            print(f"Test run is marked as done.")
            break

        length_header = conn.recv(4)
        if not length_header:
            logging.info("Simulator has closed the connection. Ending test.")
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
        
        responses = {}

        for veh_id, vehicle_info in batch_data.items():
            current_timestamp_sim = vehicle_info['timestamp']
            strongest_neighbor_id = vehicle_info.get('strongest_neighbor_id', 'None')
            strongest_neighbor_mcs = vehicle_info.get('strongest_neighbor_mcs', -1)
            
            current_power_tx = float(np.clip(vehicle_info.get('transmissionPower', MIN_TX_POWER_DBM), MIN_TX_POWER_DBM, MAX_TX_POWER_DBM))
            current_mcs = int(np.clip(round(vehicle_info.get('MCS', MIN_MCS_INDEX)), MIN_MCS_INDEX, MAX_MCS_INDEX))
            current_cbr = float(np.clip(vehicle_info.get('CBR', 0.0), 0.0, 1.0))
            current_sinr = float(np.clip(vehicle_info.get('SINR', -20.0), -20.0, 50.0))
            current_neighbors = int(np.clip(vehicle_info.get('neighbors', 0), 0, 100))
            vehicle_density = calculate_vehicle_density(current_neighbors, current_power_tx, current_mcs)
            
            current_state_raw = np.array([current_power_tx, current_mcs, current_cbr, vehicle_density, current_sinr], dtype=np.float32)

            # --- REMOVED: No reward calculation or agent.remember() needed for testing ---

            # --- MODIFIED: Choose action deterministically for evaluation ---
            # NOTE: Your SACAgent class needs to handle the `evaluate=True` flag.
            # This typically means it returns the mean of the policy distribution, not a sample.
            action_raw = agent.choose_action(current_state_raw, evaluate=True)

            new_power = float(np.clip((action_raw[0] + 1) * (MAX_TX_POWER_DBM - MIN_TX_POWER_DBM) / 2 + MIN_TX_POWER_DBM, MIN_TX_POWER_DBM, MAX_TX_POWER_DBM))
            new_mcs = int(np.clip(round((action_raw[1] + 1) * (MAX_MCS_INDEX - MIN_MCS_INDEX) / 2 + MIN_MCS_INDEX), MIN_MCS_INDEX, MAX_MCS_INDEX))
            new_beacon_rate = float(np.clip(vehicle_info.get('beaconRate', 10.0), 1.0, 20.0))

            responses[veh_id] = {"transmissionPower": new_power, "beaconRate": new_beacon_rate, "MCS": new_mcs}
            
            # --- MODIFIED: Log performance data without reward ---
            performance_row = [
                current_timestamp_sim, veh_id, current_cbr, current_sinr, strongest_neighbor_id, strongest_neighbor_mcs,
                vehicle_info.get('throughput', 0.0), vehicle_info.get('latency', 0.0),
                vehicle_info.get('PER', 0.0), vehicle_info.get('PDR', 0.0), 
                new_power, new_mcs
            ]
            test_performance_data.append(performance_row)

        # --- REMOVED: agent.learn() is not called during testing ---
        
        response_to_send = {"vehicles": responses}
        response_data = json.dumps(response_to_send).encode('utf-8')
        response_length = len(response_data)
        length_header = struct.pack('<I', response_length)
        conn.sendall(length_header)
        conn.sendall(response_data)

except (socket.error, json.JSONDecodeError, Exception) as e:
    logging.error(f"Error during test run: {e}", exc_info=True)

finally:
    conn.close()
    logging.info("Connection closed.")
    
    # Save the results from the test run
    if test_performance_data:
        # --- ADD THIS SECTION ---
        # 1. Define output paths
        plot_output_dir = 'plots_test'
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_filename = os.path.join(plot_output_dir, 'test_run_summary.png')

        # 2. Write the performance data to the CSV
        write_test_performance_metrics(test_performance_data, file_path=TEST_PERFORMANCE_LOG_PATH)
        logging.info(f"Test performance data saved to {TEST_PERFORMANCE_LOG_PATH}")

        # 3. Call the plotting function
        #    We pass `None` for the loss path since we don't have losses in testing.
        #    You may need to adjust `plot_training_summary` to handle this.
        logging.info(f"Generating test summary plot at {plot_filename}...")
        plot_training_summary(
            performance_csv_path=TEST_PERFORMANCE_LOG_PATH, 
            loss_csv_path=None,  # No loss data for testing
            figure_file=plot_filename
        )
        # --- END OF ADDED SECTION ---

    server.close()
    logging.info("Server closed. Testing finished.")