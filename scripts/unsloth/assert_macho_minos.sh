#!/usr/bin/env bash
# Pre-publish gate for the Unsloth macOS llama.cpp prebuilt. Fails the build
# unless every shipped Mach-O declares a minimum macOS <= the pinned deployment
# target (so it dyld-loads on that floor or newer), carries the expected arch slice, and
# actually launches. This is what keeps a runner/SDK bump from silently shipping
# a minos=26 binary that fails on older Macs.
#
# Usage: assert_macho_minos.sh <bin_dir> <expect_arch> [max_minos]
#   expect_arch: arm64 | x86_64        max_minos: default 14.0
set -uo pipefail

BIN_DIR="${1:?bin dir required}"
EXPECT_ARCH="${2:?expected arch required}"
MAX_MINOS="${3:-14.0}"

fail() { echo "::error::$*"; exit 1; }
# Compare dotted major.minor as major*100+minor (14.0 -> 1400).
ver_key() { local v="${1%%-*}"; awk -F. '{printf "%d", $1*100 + ($2==""?0:$2)}' <<<"$v"; }
MAX_KEY="$(ver_key "$MAX_MINOS")"

command -v vtool >/dev/null 2>&1 || fail "vtool not found (Xcode command line tools required)"

# macOS ships bash 3.2, which has no `mapfile`; read into the array portably.
MACHOS=()
while IFS= read -r _macho; do MACHOS+=("$_macho"); done < <(find "$BIN_DIR" -type f \( -name '*.dylib' -o -name 'llama-server' -o -name 'llama-quantize' -o -name 'llama-cli' \) 2>/dev/null)
[ "${#MACHOS[@]}" -gt 0 ] || fail "no Mach-O binaries found under $BIN_DIR"

for macho in "${MACHOS[@]}"; do
  minos="$(vtool -show-build "$macho" 2>/dev/null | awk '/minos/{print $2; exit}')"
  [ -n "$minos" ] || fail "$(basename "$macho") has no LC_BUILD_VERSION/minos"
  if [ "$(ver_key "$minos")" -gt "$MAX_KEY" ]; then
    fail "$(basename "$macho") minos=$minos exceeds deployment target $MAX_MINOS"
  fi
  if ! lipo -archs "$macho" 2>/dev/null | tr ' ' '\n' | grep -qx "$EXPECT_ARCH"; then
    fail "$(basename "$macho") is missing the $EXPECT_ARCH slice (got: $(lipo -archs "$macho" 2>/dev/null))"
  fi
done
echo "static check passed: ${#MACHOS[@]} Mach-O files, all minos<=$MAX_MINOS, arch=$EXPECT_ARCH"

# Runtime launch forces dyld to resolve every linked dylib (incl. Metal).
for tool in llama-cli llama-quantize; do
  bin="$(find "$BIN_DIR" -type f -name "$tool" 2>/dev/null | head -1)"
  [ -n "$bin" ] || fail "$tool not found under $BIN_DIR"
done
CLI="$(find "$BIN_DIR" -type f -name llama-cli | head -1)"
QUANT="$(find "$BIN_DIR" -type f -name llama-quantize | head -1)"
"$CLI" --version   >/dev/null 2>&1 || fail "llama-cli failed to launch (dyld load / symbol error)"
"$QUANT" --help    >/dev/null 2>&1 || fail "llama-quantize failed to launch (dyld load / symbol error)"
echo "runtime launch passed: llama-cli --version and llama-quantize --help both ran"
