import random
import paho.mqtt.client as mqtt
import json

# Cấu hình MQTT
broker_address = "10.0.0.1"
topic = "iot/sensor/data"
client_id = f"subscriber-{random.randint(1, 1000)}"

# Callback khi nhận được message
def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())
        print(f"Received: {data} on topic {message.topic}")
    except Exception as e:
        print(f"Error processing message: {e}")

# Kết nối đến broker
client = mqtt.Client(client_id=client_id)
client.on_message = on_message
client.connect(broker_address, 1883, 60)
client.subscribe(topic)

# Lắng nghe message
client.loop_forever()
