# CI workflows (install by hand)

GitHub refuses workflow files pushed with a token that lacks the `workflow` scope, so the
files in `workflows/` are kept here and copied into `.github/workflows/` from the GitHub
web UI (or pushed with a token that has the scope):

```
mkdir -p .github/workflows && cp ci/workflows/*.yaml .github/workflows/
```

Secrets the Metaculus workflows read (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|---|---|
| `METACULUS_TOKEN` | bot account token from the FutureEval participate page |
| `OPENROUTER_API_KEY` | model access (tournament credits or your own key) |
| `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET` | optional news search |

Variable `OPENFORECAST_MODEL` (optional) overrides the default model id.
