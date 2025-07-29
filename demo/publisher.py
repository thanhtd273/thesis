import paho.mqtt.client as mqtt
import time
import random
import json

# Cấu hình MQTT
broker_address = "10.0.0.1"  # IP của MQTT broker
topic = "iot/sensor/data"
client_id = f"publisher-{random.randint(1, 1000)}"

# Kết nối đến broker
client = mqtt.Client(client_id=client_id)
client.connect(broker_address, 1883, 60)

# Gửi dữ liệu mô phỏng cảm biến
while True:
    try:
        data = {
            "device_id": client_id,
            "timestamp": time.time(),
            "temperature": random.uniform(20.0, 30.0),
            "humidity": random.uniform(40.0, 80.0)
        }
        message = json.dumps(data)
        client.publish(topic, message)
        print(f"Published: {message}")
        time.sleep(2)  # Gửi mỗi 2 giây
    except Exception as e:
        print(f"Error publishing: {e}")
        time.sleep(5)
