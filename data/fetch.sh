cd "$(dirname "$0")/euroc"
for pair in "vicon_room1:02ecda9a-298f-498b-970c-b7c44334d880:6042259865" "vicon_room2:ea12bc01-3677-4b4c-853d-87c7870b8c44:0" "machine_hall:7b2419c1-62b5-4714-b7f8-485e5fe3e5fe:0"; do
  name=${pair%%:*}; rest=${pair#*:}; id=${rest%%:*}
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    code=$(curl -fsSL -A "Mozilla/5.0" -C - -o "$name.zip" -w "%{http_code}" "https://www.research-collection.ethz.ch/server/api/core/bitstreams/$id/content")
    echo "$name attempt $attempt: http $code, $(wc -c < $name.zip) bytes"
    case "$code" in 200|206|416) break;; esac
    sleep $((attempt * 60))
  done
done
echo finished
