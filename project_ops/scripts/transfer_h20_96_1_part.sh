#!/usr/bin/env bash
# Resume one archive part, reconnecting periodically on the lossy migration link.
set -euo pipefail

part=${1:?Expected a migration archive part}
case "$part" in
    /data/zym/roboparty_train/migration/h20_96_1_20261012/parts/assets.part.[0-9][0-9][0-9][0-9]|\
    /data/zym/roboparty_train/migration/h20_96_1_20261012/parts/environment.part.[0-9][0-9][0-9][0-9]) ;;
    *) echo "Unexpected archive part: $part" >&2; exit 2 ;;
esac
test -f "$part"

for attempt in $(seq 1 20); do
    if timeout --signal=INT --kill-after=10s 120s rsync -a --partial --append-verify \
        -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o IPQoS=none -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -i /home/zym/.ssh/id_rsa -p 51820' \
        "$part" zym@61.49.53.32:/data/zym/roboparty_train/migration/h20_96_1_20261012/parts/; then
        echo "PASS $(basename "$part") attempt=$attempt"
        exit 0
    fi
    echo "RETRY $(basename "$part") attempt=$attempt" >&2
done
echo "Failed after 20 attempts: $part" >&2
exit 1
