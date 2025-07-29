from scapy.all import sniff

def packet_callback(packet):
    if packet.haslayer("IP"):
        ip_layer = packet["IP"]
        print(f"[+] New packet: {ip_layer.src} -> {ip_layer.dst}")
        print(f"    Protocol: {ip_layer.proto}, Packet size: {len(packet)} bytes")
        print("-" * 50)


sniff(iface = "eth0", prn = packet_callback, store = False)