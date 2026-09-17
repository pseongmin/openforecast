#!/bin/bash
# Install the GitHub Actions workflows and repository secrets for the Metaculus bot.
# Requires: a GitHub token with `repo` AND `workflow` scope in $GH_TOKEN (or $PSEONGMIN_GITHUB_TOKEN),
# and the bot credentials in the environment. Idempotent: safe to re-run.
set -euo pipefail
REPO="${REPO:-pseongmin/openforecast}"
TOKEN="${GH_TOKEN:-${PSEONGMIN_GITHUB_TOKEN:-}}"
[ -n "$TOKEN" ] || { echo "no token: set GH_TOKEN or PSEONGMIN_GITHUB_TOKEN"; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "1/3 installing workflow files"
mkdir -p "$ROOT/.github/workflows"
cp "$ROOT"/ci/workflows/*.yaml "$ROOT/.github/workflows/"
cd "$ROOT"
git add .github/workflows
git commit -qm 'ci: install Metaculus bot workflows' || echo "  (nothing to commit)"
git push -q origin main || { echo "  push refused — the token still lacks the 'workflow' scope"; exit 3; }

echo "2/3 setting repository secrets"
PUBKEY_JSON=$(curl -sf -H "Authorization: Bearer $TOKEN" "https://api.github.com/repos/$REPO/actions/secrets/public-key")
set_secret() {
  local name="$1" value="${2:-}"
  [ -n "$value" ] || { echo "  skip $name (not set in the environment)"; return 0; }
  python3 - "$name" "$value" <<'PY'
import base64, json, os, sys, urllib.request
from nacl import encoding, public  # PyNaCl
name, value = sys.argv[1], sys.argv[2]
repo, token, pubkey = os.environ["REPO"], os.environ["TOKEN"], json.loads(os.environ["PUBKEY_JSON"])
sealed = public.SealedBox(public.PublicKey(pubkey["key"].encode(), encoding.Base64Encoder())).encrypt(value.encode())
body = json.dumps({"encrypted_value": base64.b64encode(sealed).decode(), "key_id": pubkey["key_id"]}).encode()
req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/secrets/{name}", data=body, method="PUT",
                             headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req) as r:
    print(f"  {name}: HTTP {r.status}")
PY
}
export REPO TOKEN PUBKEY_JSON
set_secret METACULUS_TOKEN "${METACULUS_TOKEN:-}"
set_secret OPENROUTER_API_KEY "${OPENROUTER_API_KEY:-}"
set_secret ASKNEWS_CLIENT_ID "${ASKNEWS_CLIENT_ID:-}"
set_secret ASKNEWS_SECRET "${ASKNEWS_SECRET:-}"

echo "3/3 next: run the smoke test"
echo "  gh workflow run 'Metaculus — test bot (bot-testing-area)' --repo $REPO"
echo "  then check the bot profile on Metaculus for posted forecasts."
