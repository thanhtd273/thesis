from mininet.net import Mininet
from mininet.node import OVSController
from mininet.cli import CLI
from mininet.node import Node
from mininet.log import setLogLevel, info

def iot_topology():
    net = Mininet(controller=OVSController)

    info("*** Adding controller\n")
    net.addController('c0')

    info("*** Adding switch\n")
    s1 = net.addSwitch('s1')

    info("*** Adding NAT (to connect to Internet)\n")
    nat = net.addHost('nat0', cls=Node, ip='10.0.0.254', inNamespace=False)
    net.addLink(nat, s1)

    info("*** Configuring NAT\n")
    nat.cmd('iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE')
    nat.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')

    info("*** Adding MQTT broker\n")
    broker = net.addHost('broker', ip='10.0.0.1')
    broker.cmd('apt update && apt install -y mosquitto mosquitto-clients')
    broker.cmd('mkdir -p /etc/mosquitto/conf.d')
    broker.cmd('echo "listener 1883" > /etc/mosquitto/conf.d/custom.conf')
    broker.cmd('echo "allow_anonymous true" >> /etc/mosquitto/conf.d/custom.conf')
    broker.cmd('pkill mosquitto; mosquitto -c /etc/mosquitto/conf.d/custom.conf -d')

    num_iot_nodes = 5
    iot_nodes = []
    for i in range(num_iot_nodes):
        node = net.addHost(f'iot{i+1}', ip=f'10.0.0.{i+2}')
        iot_nodes.append(node)

    num_attackers = 2
    attackers = []
    for i in range(num_attackers):
        attacker = net.addHost(f'attacker{i+1}', ip=f'10.0.0.{i+7}')
        attackers.append(attacker)

    info("*** Creating links\n")
    net.addLink(broker, s1)
    for node in iot_nodes:
        net.addLink(node, s1)
    for attacker in attackers:
        net.addLink(attacker, s1)

    info("*** Starting network\n")
    net.start()

    nat.cmd('sysctl -w net.ipv4.ip_forward=1')
    nat.cmd("iptables -F")
    nat.cmd('iptables -t nat -F')
    internet_iface  = 'ens33'
    nat.cmd(f'iptables -t nat -A POSTROUTING -o {internet_iface} -j MASQUERADE')
    nat.cmd('iptables -A FORWARD -i s1-eth1 -j ACCEPT')
    nat.cmd(f'iptables -A FORWARD -o {internet_iface} -m state --state RELATED,ESTABLISHED -j ACCEPT')

    for h in iot_nodes + attackers + [broker]:
        h.cmd('ip route add default via 10.0.0.254')
        h.cmd('echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" > /etc/resolv.conf')

    info("*** Installing Mosquitto on broker\n")
    broker.cmd('apt update && apt install -y mosquitto mosquitto-clients')
    broker.cmd('mosquitto -d')

    info("*** Installing depenedencies on iot nodes\n")
    for node in iot_nodes:
        node.cmd("apt update && apt install -y mosquitto-clients python3-pip")
        node.cmd("pip3 install paho-mqtt")

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    iot_topology()
