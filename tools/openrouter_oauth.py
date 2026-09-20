"""Obtain an OpenRouter API key through its headless OAuth (PKCE) flow.

No browser is needed on this machine and the secret half (the code verifier)
never leaves it. Two steps:

    python tools/openrouter_oauth.py start          # prints a URL to open anywhere
    python tools/openrouter_oauth.py finish <CODE>  # exchanges the on-screen code for a key

The key is written to ~/.env as OPENROUTER_API_KEY (file mode 600) and checked
with GET /api/v1/key. The authorization code is single-use and expires after
ten minutes, so run `finish` promptly.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import urllib.request
from pathlib import Path

STATE = Path.home() / ".openrouter_pkce_state"
ENV = Path.home() / ".env"
KEY_LABEL = "openforecast-bot"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def start() -> int:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    STATE.write_text(verifier)
    STATE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    url = (
        "https://openrouter.ai/auth?code_challenge=" + challenge
        + "&code_challenge_method=S256&key_label=" + KEY_LABEL
    )
    print("Open this URL in any browser, sign in / sign up, authorize, then copy the code shown:")
    print(url)
    print("\nThen run:  python tools/openrouter_oauth.py finish <CODE>")
    return 0


def finish(code: str) -> int:
    if not STATE.exists():
        print("no pending authorization: run `start` first", file=sys.stderr)
        return 2
    verifier = STATE.read_text().strip()
    body = json.dumps({"code": code.strip(), "code_verifier": verifier, "code_challenge_method": "S256"}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/auth/keys", data=body, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "openforecast/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            key = json.load(r).get("key")
    except urllib.error.HTTPError as exc:
        print(f"exchange failed: HTTP {exc.code} {exc.read()[:200].decode(errors='ignore')}", file=sys.stderr)
        return 3
    if not key:
        print("exchange returned no key", file=sys.stderr)
        return 3
    _write_env("OPENROUTER_API_KEY", key)
    STATE.unlink(missing_ok=True)
    return verify(key)


def _write_env(name: str, value: str) -> None:
    text = ENV.read_text() if ENV.exists() else ""
    line = f'{name}="{value}"'
    pattern = re.compile(rf"^{name}=.*$", re.M)
    text = pattern.sub(line, text) if pattern.search(text) else (text.rstrip("\n") + "\n" + line + "\n")
    ENV.write_text(text)
    ENV.chmod(stat.S_IRUSR | stat.S_IWUSR)


def verify(key: str) -> int:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}", "User-Agent": "openforecast/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r).get("data", {})
    print(f"key stored as OPENROUTER_API_KEY (prefix {key[:6]}...); label={data.get('label')} "
          f"usage={data.get('usage')} limit={data.get('limit')} free_tier={data.get('is_free_tier')}")
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "start":
        return start()
    if len(sys.argv) >= 3 and sys.argv[1] == "finish":
        return finish(sys.argv[2])
    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        env = dict(re.findall(r'^([A-Z0-9_]+)="?([^"\n]*)"?$', ENV.read_text(), re.M))
        return verify(env["OPENROUTER_API_KEY"]) if env.get("OPENROUTER_API_KEY") else 2
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
