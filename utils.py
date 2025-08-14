import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np

def plot_training_summary(performance_csv_path, loss_csv_path, figure_file):
    """
    Reads logs from a specific best-performing episode and generates a 4x3 summary plot
    with separated metrics and losses for clarity.
    """
    performance_log = performance_csv_path
    loss_log = loss_csv_path

    logging.info(f"Reading best episode performance data from {performance_log}")
    logging.info(f"Reading loss data from {loss_log}")

    try:
        perf_df = pd.read_csv(performance_log)
        if perf_df.empty:
            logging.warning(f"Performance log file is empty: {performance_log}. Skipping plot.")
            return
        # Ensure PDR column exists, if not, create it from PER
        if 'PDR' not in perf_df.columns and 'PER' in perf_df.columns:
            perf_df['PDR'] = 1 - perf_df['PER']
            
        agg_perf_df = perf_df.drop(columns=['VehicleID', 'Strongest Neighbor']).groupby('Timestamp').mean().reset_index()
    except FileNotFoundError:
        logging.error(f"Best performance log file not found at {performance_log}. Cannot generate plots.")
        return
    except Exception as e:
        logging.error(f"Error reading performance log: {e}")
        return

    try:
        loss_df = pd.read_csv(loss_log)
    except FileNotFoundError:
        logging.warning(f"Loss log file not found at {loss_log}. Skipping loss plots.")
        loss_df = None
    except Exception as e:
        logging.error(f"Error reading loss log: {e}")
        loss_df = None

    # --- MODIFIED: Changed grid to 4x3 for separated loss plots ---
    fig, axes = plt.subplots(4, 3, figsize=(27, 28)) # Increased figure size
    fig.suptitle(f'Best Episode Performance Summary\nData from: {os.path.basename(performance_log)}', fontsize=20, y=0.98)
    sns.set_theme(style="whitegrid")
    
    rolling_window = max(1, len(agg_perf_df) // 20)

    # --- Row 1: Overall Performance ---

     # Plot 1: Cumulative Reward
    ax = axes[0, 0]
    if 'Reward' in agg_perf_df.columns:
        ax.plot(agg_perf_df['Timestamp'], agg_perf_df['Reward'].cumsum(), label='Cumulative Reward', color='g', linewidth=2)
        ax.set_title('Cumulative Reward in Best Episode', fontsize=14)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Reward Plot\nNot Available for Test Run', ha='center', va='center')
        ax.set_title('Cumulative Reward', fontsize=14)

    # --- CORRECTED INDENTATION ---
    # Plot 2: Average Throughput (Now correctly indented)
    ax = axes[0, 1]
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['Throughput'].rolling(window=rolling_window).mean(), label='Avg. Throughput (Smoothed)', color='b')
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['Throughput'], color='b', alpha=0.1, label='_nolegend_')
    ax.set_title('Average Throughput', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.legend()

    # Plot 3: Packet Delivery Ratio (PDR) (Now correctly indented)
    ax = axes[0, 2]
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['PDR'].rolling(window=rolling_window).mean(), label='Avg. PDR (Smoothed)', color='r')
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['PDR'], color='r', alpha=0.1, label='_nolegend_')
    ax.set_title('Packet Delivery Ratio (PDR)', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('PDR')
    ax.set_ylim(0, 1.05) # PDR is between 0 and 1
    ax.legend()
    
    # --- Row 2: Channel Conditions & Reward Breakdown ---

    # Plot 4: Average CBR
    ax = axes[1, 0]
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['CBR'].rolling(window=rolling_window).mean(), label='Avg. CBR (Smoothed)', color='purple')
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['CBR'], color='purple', alpha=0.1, label='_nolegend_')
    ax.set_title('Channel Busy Ratio (CBR)', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('CBR')
    ax.legend()

    # Plot 5: Average SINR
    ax = axes[1, 1]
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['SNR'].rolling(window=rolling_window).mean(), label='Avg. SINR (Smoothed)', color='orange')
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['SNR'], color='orange', alpha=0.1, label='_nolegend_')
    ax.set_title('Signal-to-Interference-plus-Noise Ratio (SINR)', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('SINR (dB)')
    ax.legend()
    
    # Plot 6: Reward Components
    ax = axes[1, 2]
    # Check if reward component columns exist
    if 'cbr_term' in agg_perf_df.columns:
        ax.plot(agg_perf_df['Timestamp'], agg_perf_df['cbr_term'].rolling(window=rolling_window).mean(), label='CBR Term (Smoothed)', color='tab:blue')
        sinr_col_name = 'sinr_bonus' if 'sinr_bonus' in agg_perf_df.columns else 'sinr_term'
        if sinr_col_name in agg_perf_df.columns:
            ax.plot(agg_perf_df['Timestamp'], agg_perf_df[sinr_col_name].rolling(window=rolling_window).mean(), label='SINR Term (Smoothed)', color='tab:orange')
        if 'rate_term' in agg_perf_df.columns:
            ax.plot(agg_perf_df['Timestamp'], agg_perf_df['rate_term'].rolling(window=rolling_window).mean(), label='Rate Term (Smoothed)', color='tab:green')
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    else:
        ax.text(0.5, 0.5, 'Reward Components\nNot Available for Test Run', ha='center', va='center')

    ax.set_title('Reward Components Over Time', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('Component Value')
    
    # --- Row 3: Agent Actions ---

    # Plot 7: Average Power Action
    ax = axes[2, 0]
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['Power_dBm'].rolling(window=rolling_window).mean(), label='Avg. Power (dBm, Smoothed)', color='darkcyan')
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['Power_dBm'], color='darkcyan', alpha=0.1, label='_nolegend_')
    ax.set_title('Average Power Action', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('Power (dBm)')
    ax.legend()

    # Plot 8: Average MCS Action
    ax = axes[2, 1]
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['MCS'].rolling(window=rolling_window).mean(), label='Avg. MCS (Smoothed)', color='saddlebrown')
    ax.plot(agg_perf_df['Timestamp'], agg_perf_df['MCS'], color='saddlebrown', alpha=0.1, label='_nolegend_')
    ax.set_title('Average MCS Action', fontsize=14)
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('MCS Index')
    ax.legend()
    
    # Disable the unused plot in this row
    axes[2, 2].axis('off')

    # --- Row 4: Training Losses (SEPARATED) ---
    loss_rolling = max(1, len(loss_df) // 50) if loss_df is not None and not loss_df.empty else 1
    
    # Plot 10: Actor Loss
    ax = axes[3, 0]
    if loss_df is not None and not loss_df.empty and 'Actor_Loss' in loss_df.columns:
        ax.plot(loss_df['Step'], loss_df['Actor_Loss'].rolling(window=loss_rolling).mean(), label='Actor Loss (Smoothed)', color='tab:blue')
        ax.set_title('Smoothed Actor Loss', fontsize=14)
        ax.set_xlabel('Learn Step')
        ax.set_ylabel('Loss')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Actor Loss\nNot Available', ha='center', va='center')
        ax.set_title('Actor Loss', fontsize=14)

    # Plot 11: Critic Loss
    ax = axes[3, 1]
    if loss_df is not None and not loss_df.empty and 'Critic_Loss' in loss_df.columns:
        ax.plot(loss_df['Step'], loss_df['Critic_Loss'].rolling(window=loss_rolling).mean(), label='Critic Loss (Smoothed)', color='tab:orange')
        ax.set_title('Smoothed Critic Loss', fontsize=14)
        ax.set_xlabel('Learn Step')
        ax.set_ylabel('Loss')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Critic Loss\nNot Available', ha='center', va='center')
        ax.set_title('Critic Loss', fontsize=14)
        
    # Plot 12: Alpha (Temperature) Loss
    ax = axes[3, 2]
    if loss_df is not None and not loss_df.empty and 'Alpha_Loss' in loss_df.columns:
        ax.plot(loss_df['Step'], loss_df['Alpha_Loss'].rolling(window=loss_rolling).mean(), label='Alpha Loss (Smoothed)', color='tab:red')
        ax.set_title('Smoothed Alpha (Temperature) Loss', fontsize=14)
        ax.set_xlabel('Learn Step')
        ax.set_ylabel('Loss')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Alpha Loss\nNot Available', ha='center', va='center')
        ax.set_title('Alpha Loss', fontsize=14)
        
    # --- Final Layout Adjustments ---
    for ax_row in axes:
        for ax_item in ax_row:
            ax_item.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to fit suptitle
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plt.savefig(figure_file)
    plt.close()
    logging.info(f"✅ Best episode summary plot saved to {figure_file}")