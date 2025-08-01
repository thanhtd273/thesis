import logging
import pickle
import time
import os
from collections import deque, defaultdict

import numpy as np
import joblib
from scapy.layers.inet import IP
from tensorflow.keras.models import load_model  # dùng tensorflow.keras
from scapy.all import sniff

# ---------- CONFIGURATION ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Time slot length in seconds (must match training preprocessing, e.g., 10 minutes = 600)
TIME_SLOT_SECONDS = 10
WINDOW_SIZE = 10
PREDICTION_THRESHOLD = 0.5  # xác suất > threshold coi là attack

# List of target nodes you want to run MM-WC detection for
# These should correspond to the node IDs used during training (e.g., integers or strings)
TARGET_NODES = ["454"]

# Mapping from IP in Mininet to target node ID
ip_to_node_map = {
    "10.0.0.2": "454",
}

# Base path where per-node model/scaler/etc are stored
MODEL_BASE_DIR = "models"  # structure: models/{node_id}/final_model.keras etc.

# ---------- STATE ----------
# For each observed node, maintain per-slot packet counts (could also be byte counts)
current_slot_start = None
slot_counts = defaultdict(int)  # key: node_id, value: count in current slot

# Sliding windows for each target node and its correlated peers: dict[node_id] -> deque of length WINDOW_SIZE of vectors
# Each entry is a dict of node volumes for that time-slot (self + selected correlated nodes)
windows = {}  # e.g., windows["454"] = deque(maxlen=WINDOW_SIZE)

# Loaded models/scalers/selected_nodes
models = {}
scalers = {}
selected_nodes_for_correlation = {}  # per target node: list including self

# ---------- UTILITIES ----------
def load_per_node_assets(node_id):
    """Load model, scaler, and selected correlated nodes for a target node."""
    if node_id in models:
        return  # already loaded
    model_dir = os.path.join(MODEL_BASE_DIR, str(node_id))
    try:
        model = load_model(os.path.join(model_dir, "final_model.keras"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        with open(os.path.join(model_dir, "selected_nodes_for_correlation.pkl"), "rb") as f:
            sel = pickle.load(f)
        # Ensure self is included
        if node_id not in sel:
            sel = [node_id] + sel
        models[node_id] = model
        scalers[node_id] = scaler
        selected_nodes_for_correlation[node_id] = sel
        windows[node_id] = deque(maxlen=WINDOW_SIZE)  # will hold list of dicts per slot
        logging.info(f"Loaded assets for node {node_id}, correlated set: {sel}")
    except Exception as e:
        logging.error(f"Failed to load assets for node {node_id} from {model_dir}: {e}")

def roll_time_slot(now):
    """Check if new slot started; if yes, flush previous and shift window."""
    global current_slot_start, slot_counts
    if current_slot_start is None:
        current_slot_start = now
        return

    if now - current_slot_start >= TIME_SLOT_SECONDS:
        # finalize this slot: for each target node, assemble the feature vector of this slot
        for target in TARGET_NODES:
            if target not in selected_nodes_for_correlation:
                continue
            sel = selected_nodes_for_correlation[target]  # list of node IDs used as correlation inputs
            # For this time-slot, build dict: key=node in sel, value=packet count (default 0)
            slot_vector = {node: slot_counts.get(node, 0) for node in sel}
            # Append to its sliding window
            windows[target].append(slot_vector)
        # reset slot
        slot_counts = defaultdict(int)
        # advance to next slot boundary (naively)
        current_slot_start += TIME_SLOT_SECONDS

def try_infer():
    for target in TARGET_NODES:
        if target not in windows:
            continue
        if len(windows[target]) < WINDOW_SIZE:
            continue  # thiếu full window

        sel = selected_nodes_for_correlation[target]  # list of correlated node IDs including self
        num_features = len(sel)
        # Build matrix (time_window, num_features)
        mat = np.zeros((WINDOW_SIZE, num_features), dtype=float)
        for t_idx, slot_dict in enumerate(windows[target]):
            for f_idx, node in enumerate(sel):
                mat[t_idx, f_idx] = slot_dict.get(node, 0)

        scaler = scalers.get(target)
        model = models.get(target)
        if scaler is None or model is None:
            continue

        # **CORRECTION**: scale per time-step, not flatten whole window
        # mat shape: (WINDOW_SIZE, num_features)
        try:
            scaled_steps = scaler.transform(mat)  # transforms each row independently
        except Exception as e:
            logging.error(f"Scaler transform failed for {target}: {e}")
            return

        # reshape to (1, time_window, num_features)
        input_seq = scaled_steps.reshape(1, WINDOW_SIZE, num_features)

        prob = float(model.predict(input_seq, verbose=0)[0][0])
        attacked = prob > PREDICTION_THRESHOLD
        if attacked:
            logging.warning(f"[Node {target}] DDoS detected! score={prob:.3f}")
        else:
            logging.info(f"[Node {target}] benign. score={prob:.3f}")


# ---------- PACKET HANDLER ----------
def packet_callback(pkt):
    global slot_counts
    now = time.time()
    roll_time_slot(now)

    if IP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        # Map observed IP to node IDs used in training; we count per source (you can extend to bytes)
        for ip in (src_ip, dst_ip):
            node_id = ip_to_node_map.get(ip)
            if node_id:
                slot_counts[node_id] += 1  # increment packet count (could also add len(pkt) for bytes)
    # After updating slot counts, attempt inference
    try_infer()

# ---------- MAIN ----------
def main():
    # Preload all target nodes assets
    for t in TARGET_NODES:
        load_per_node_assets(t)

    # Start sniffing (run indefinitely)
    iface = "broker-eth0"
    logging.info(f"Starting sniffing on interface {iface}")
    sniff(iface=iface, prn=packet_callback, store=False)

if __name__ == "__main__":
    main()
