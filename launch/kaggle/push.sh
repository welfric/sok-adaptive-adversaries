#!/usr/bin/env bash
# Push one Kaggle kernel from this repo.
#
#   launch/kaggle/push.sh launch/kaggle/adversarial_robustness/blacklight
#   launch/kaggle/push.sh launch/kaggle/poisoning/flame
#
# The kernel folders hold placeholders so no username or commit is checked in:
#   __KAGGLE_USERNAME__  -> your Kaggle username (KAGGLE_USERNAME, else kaggle.json)
#   __SOK_COMMIT__       -> this repo's HEAD commit, "-dirty" if the tree has changes
# The filled-in copy goes to a temp folder; the repo is not modified.
set -euo pipefail

kernel_dir="${1:?usage: push.sh <kernel folder>}"
# Fixed: all runs use a T4 so numbers are comparable. "NvidiaTeslaT4" is Kaggle's
# T4 x2 option (there is no single-T4 option); run.py uses GPU 0 only.
accelerator="NvidiaTeslaT4"

[ -f "$kernel_dir/kernel-metadata.json" ] || { echo "no kernel-metadata.json in $kernel_dir" >&2; exit 1; }

# Prefer the repo's .venv (kaggle >= 2.x, which has --accelerator; needs Python >= 3.11).
repo_root="$(git rev-parse --show-toplevel)"
if [ -x "$repo_root/.venv/bin/kaggle" ]; then kaggle="$repo_root/.venv/bin/kaggle"
elif command -v kaggle >/dev/null; then kaggle="kaggle"
else echo "kaggle CLI not found: see launch/README.md, one-time setup" >&2; exit 1; fi

# Username: KAGGLE_USERNAME, else legacy kaggle.json, else the logged-in CLI config.
user="${KAGGLE_USERNAME:-}"
if [ -z "$user" ]; then
  for f in "$HOME/.kaggle/kaggle.json" "$HOME/.config/kaggle/kaggle.json"; do
    if [ -f "$f" ]; then
      user="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["username"])' "$f")"
      break
    fi
  done
fi
if [ -z "$user" ]; then
  user="$("$kaggle" config view 2>/dev/null | sed -n 's/^[- ]*username:[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -1)"
  [ "$user" = "None" ] && user=""
fi
[ -n "$user" ] || { echo "could not find your Kaggle username: run 'kaggle auth login' or set KAGGLE_USERNAME" >&2; exit 1; }

commit="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then commit="${commit}-dirty"; fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -R "$kernel_dir/." "$tmp/"
for f in "$tmp"/*; do
  [ -f "$f" ] || continue
  sed -i.bak -e "s/__KAGGLE_USERNAME__/$user/g" -e "s/__SOK_COMMIT__/$commit/g" "$f"
  rm -f "$f.bak"
done

slug="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$tmp/kernel-metadata.json")"
echo "pushing $slug (commit $commit, accelerator $accelerator)"
"$kaggle" kernels push -p "$tmp" --accelerator "$accelerator"
echo
shown="kaggle"; [ "$kaggle" = "$repo_root/.venv/bin/kaggle" ] && shown=".venv/bin/kaggle"
echo "status:  $shown kernels status $slug"
echo "logs:    $shown kernels logs $slug"
echo "results: $shown kernels output $slug -p results/kaggle/$(basename "$slug") -o"
