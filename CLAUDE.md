# Project Rules

- Farm is LIVE on Kraken Futures — do NOT make breaking changes to zones.py, structure.py, backtest.py, live/ without verifying stats
- After any backtest logic change, verify: ~27,300 trades, ~+14,760R, PF ~2.11 on full portfolio (8 coins)
- Run tests: `python3 main.py --wfo --days=1460` (BTC WFO), `python3 main.py --test` (portfolio)
- Do NOT re-test dead ends (see memory) — BE, partial close, filters, 15m, lookback optimization
- OPTIMAL_PARAMS changes require full WFO validation
