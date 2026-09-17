# Data policy

This project is developed in a clean-room workspace and may use **only**:

- Public benchmark inputs distributed by the competition organizers.
- Public, freely redistributable data APIs: Yahoo Finance, FRED (public CSV endpoint),
  DBnomics, Wikipedia/Wikimedia, Manifold Markets, Kalshi public market data.
- Public documentation and open-source libraries under their own licenses.

It must **never** contain, import, or derive from:

- Any employer-owned dataset, model, pipeline, credential, or internal identifier.
- Any private trading account, position, or performance record.
- Any non-public repository, internal host, path, or service.

`tools/leak_guard.py` runs before every commit and fails on any of the forbidden markers.
