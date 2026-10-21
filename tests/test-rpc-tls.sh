#!/usr/bin/env bash
# Mutual TLS transport: accepted peer, refusals (wrong CA, unlisted client, wrong server pin, plaintext
# either way, revoked fingerprint), byte-exact transfers. Every client run is bounded: a hang fails.
set -uo pipefail

server=$1
client=$2
dir=$(mktemp -d)
port=$((42000 + $$ % 10000))
pids=()
cleanup() {
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
    rm -rf "$dir"
}
trap cleanup EXIT
fail=0

ca() { # name
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 1 -subj "/CN=$1" \
        -keyout "$dir/$1.key" -out "$dir/$1.crt" -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,keyCertSign >/dev/null 2>&1
}
leaf() { # name ca
    openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -subj "/CN=$1" \
        -keyout "$dir/$1.key" -out "$dir/$1.csr" >/dev/null 2>&1
    printf "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth,clientAuth\n" > "$dir/$1.ext"
    openssl x509 -req -in "$dir/$1.csr" -CA "$dir/$2.crt" -CAkey "$dir/$2.key" -CAcreateserial -days 1 \
        -extfile "$dir/$1.ext" -out "$dir/$1.crt" >/dev/null 2>&1
}
fp() { openssl x509 -in "$dir/$1.crt" -outform DER | sha256sum | cut -d' ' -f1; }

ca caA; ca caB
leaf server caA; leaf client caA; leaf other caA; leaf rogue caB
fp client > "$dir/allow"

wait_port() {
    for _ in $(seq 200); do (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0; sleep 0.05; done
    return 1
}

"$server" --device CPU -H 127.0.0.1 -p "$port" --tls-cert "$dir/server.crt" --tls-key "$dir/server.key" \
    --tls-ca "$dir/caA.crt" --tls-allow "$dir/allow" > "$dir/tls_server.log" 2>&1 &
pids+=($!)
"$server" --device CPU -H 127.0.0.1 -p $((port + 1)) > "$dir/plain_server.log" 2>&1 &
pids+=($!)
wait_port "$port" && wait_port $((port + 1)) || { echo "servers did not start"; cat "$dir"/*.log; exit 1; }

run() { # expect(ok|refused) label cert endpoint [MiB]
    local expect=$1 label=$2 who=$3 ep=$4 mib=${5:-16}
    ( GGML_RPC_TLS_CERT="$dir/$who.crt" GGML_RPC_TLS_KEY="$dir/$who.key" GGML_RPC_TLS_CA="$dir/caA.crt" \
        exec timeout 60 "$client" "$ep" "$mib" > "$dir/run.log" 2>&1 )
    local rc=$?
    local why
    why=$( { grep -h -m1 -E "TLS:|failed|crashed" "$dir/run.log"; tail -n 1 "$dir/tls_server.log"; } | tr '\n' ' ' | cut -c1-200)
    if [ $rc -eq 124 ]; then
        echo "FAIL $label: client hung"; fail=1
    elif [ "$expect" = ok ] && [ $rc -ne 0 ]; then
        echo "FAIL $label: expected success, rc=$rc"; tail -5 "$dir/run.log"; fail=1
    elif [ "$expect" = refused ] && [ $rc -eq 0 ]; then
        echo "FAIL $label: expected refusal, got through"; fail=1
    else
        echo "ok   $label (rc=$rc) $([ "$expect" = refused ] && echo "[$why]")"
    fi
}

S=$(fp server)
run ok      "accepted client, 64 MiB byte-exact"   client "tls://$S@127.0.0.1:$port" 64
grep -q "Accepted TLS client $(fp client)" "$dir/tls_server.log" && echo "ok   server logged the client fingerprint" \
    || { echo "FAIL server did not log the client fingerprint"; fail=1; }
grep -q "TLS: read failed" "$dir/tls_server.log" && { echo "FAIL a normal disconnect was logged as a failed read (no close_notify)"; fail=1; } \
    || echo "ok   a normal disconnect ends with close_notify"

# a peer trickling a handshake one byte per second must be dropped at the handshake deadline (10 s),
# not kept alive by each byte arriving before a per-receive timeout
start=$SECONDS
(
    exec 3<>"/dev/tcp/127.0.0.1/$port" || exit 0
    printf '\x16\x03\x01\x02\x00' >&3 # a TLS handshake record header announcing 512 bytes
    for _ in $(seq 40); do printf 'A' >&3 2>/dev/null || exit 0; sleep 1; done
) 2>/dev/null
held=$((SECONDS - start))
if [ "$held" -le 20 ]; then
    echo "ok   a trickling handshake is dropped at the deadline (${held} s)"
else
    echo "FAIL a trickling handshake held the accept loop for ${held} s"; fail=1
fi

run refused "client from another CA"               rogue  "tls://$S@127.0.0.1:$port"
run refused "client not in the allowlist"          other  "tls://$S@127.0.0.1:$port"
run refused "wrong server pin"                     client "tls://$(fp other)@127.0.0.1:$port"
run refused "plaintext client to the TLS server"   client "127.0.0.1:$port"
run refused "TLS client to a plaintext server"     client "tls://$S@127.0.0.1:$((port + 1))"
: > "$dir/allow"
run refused "revoked (allowlist emptied)"          client "tls://$S@127.0.0.1:$port"
fp client > "$dir/allow"
run ok      "re-allowed, server still serving"     client "tls://$S@127.0.0.1:$port"
run ok      "plain server unaffected"              client "127.0.0.1:$((port + 1))"
exit $fail
