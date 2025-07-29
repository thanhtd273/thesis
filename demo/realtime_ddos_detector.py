import logging, pickle, joblib, ipaddress
import numpy as np
from keras.src.saving import load_model
from scapy.layers.inet import IP
from scapy.sendrecv import sniff

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Download scaler and training model
scaler = joblib.load('scaler.pkl')             # Standardized storage set
model = load_model('final_model.keras')        # Training model has been trained

# Download the relevant button list (if any)
try:
    with open('selected_nodes_for_correlation.pkl','rb') as f:
        selected_nodes = pickle.load(f)
    logging.info(f"Correlated nodes: {selected_nodes}")
except:
    selected_nodes = []
    logging.info("No correlated nodes file found.")

# Map (if needed) from IP to button ID in mininet
ip_to_node_map = {
    # '10.0.0.1': 'n1',
    # '10.0.0.2': 'n2',
    # ...
}

window_size = 10
packet_buffer = []

def process_window(packets):
    # Prepare features [IAT, src_int, dst_int, length]
    features = []
    for i, (ts, src, dst, length) in enumerate(packets):
        start_time = packets[0][0]
        iat = ts - start_time if i > 0 else 0
        src_int = int(ipaddress.ip_address(src))
        dst_int = int(ipaddress.ip_address(dst))
        features.append([iat, src_int, dst_int, length])

    X = np.array(features, dtype=float)
    X_scaled = scaler.transform(X)  # Standardize

    # Thêm đặc trưng tương quan (nếu có)
    if selected_nodes:
        corr_counts = {node: 0 for node in selected_nodes}
        for ts, src, dst, length in packets:
            node = ip_to_node_map.get(src) or ip_to_node_map.get(dst)
            if node in corr_counts:
                corr_counts[node] += 1
        corr_vals = [corr_counts[node] for node in selected_nodes]
        corr_matrix = np.tile(corr_vals, (window_size, 1))
        X_scaled = np.hstack((X_scaled, corr_matrix))

    # Dự đoán bằng LSTM
    input_seq = np.expand_dims(X_scaled, axis=0)  # (1, time_steps, features)
    prob = float(model.predict(input_seq)[0][0])
    attacked = prob > 0.5
    if attacked:
        logging.warning(f"Possible DDoS attack detected! Score={prob:.3f}")
    else:
        logging.info(f"No attack. Score={prob:.3f}")

def packet_callback(pkt):
    if IP in pkt:
        ts = pkt.time
        src = pkt[IP].src; dst = pkt[IP].dst
        length = len(pkt)
        packet_buffer.append((ts, src, dst, length))
        if len(packet_buffer) >= window_size:
            window = packet_buffer.copy()
            process_window(window)
            packet_buffer.clear()

# Start the package on the appropriate interface (for example, 'HX-ETH0' or 'ETH0')
sniff(iface="eth0", prn=packet_callback, store=0)
