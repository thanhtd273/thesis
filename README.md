### Install necessary tools

```
sudo apt update
sudo apt install -y mininet
pip3 install ryu
pip3 install tensorflow keras joblib numpy pandas scikit-learn
sudo apt install -y mosquitto mosquitto-clients python3-pip
pip3 install paho-mqtt
```
### Mininet
```
sudo apt install mininet -y
sudo apt install -y openvswitch-testcontroller
sudo ln -s /usr/bin/ovs-testcontroller /usr/bin/controller
sudo mn -c
```
### Shut down port :6653
```
sudo lsof -i :6653
sudo kill -9 1182
```