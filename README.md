# Supply & Demand Zone Strategy

Automated supply/demand zone trading strategy for Kraken Futures. Detects impulse-based zones, scores them (trend, structure, liquidity, volume), and trades reversals with dynamic TP targeting.

## Quick Start

```bash
# 1. Cache historical data (once, or to refresh)
python3 cache_data.py                     # All portfolio coins
python3 cache_data.py --coins=BTC,ETH     # Specific coins

# 2. Test strategy on portfolio
python3 main.py --test                    # Full portfolio (cached data)
python3 main.py --test --coins=BTC,ETH    # Specific coins
python3 main.py --test --mc              # + Monte Carlo stress test

# 3. Walk-forward optimization (BTC)
python3 main.py --wfo --days=1460         # 4 years, parallel
python3 main.py --wfo --serial            # Single-threaded (debug)

# 4. BTC scan + chart
python3 main.py                           # Quick scan (365 days)
python3 main.py --days=365                # Custom lookback

# 5. Live trading
python3 -m live.farm --coins=AVAX,ETH     # Default: 0.5% risk, 4 max trades
python3 -m live.farm --coins=BTC --risk-pct=1.0 --max-trades=3
```

## Portfolio (sorted by R/Mo)

AVAX, UNI, LINK, LTC, BNB, SOL, ETH, BTC

## Project Structure

```
main.py              CLI entry point (--test, --wfo, scan)
backtest.py          Backtest engine (walkforward_backtest)
zones.py             Zone detection (impulse + structure + liquidity)
structure.py         Market structure (swings, ATR, trend, BOS)
config.py            Strategy parameters + OPTIMAL_PARAMS
data.py              Kraken Futures data fetching (ccxt)
wfo.py               Walk-forward optimization engine
reporting.py         Reports, charts, CSV export
cache_data.py        Download + cache OHLCV as parquet

live/
  farm.py            Trading farm (zone detection + order management)
  kraken_client.py   Kraken Futures API client
```

## Strategy Parameters (OPTIMAL_PARAMS)

| Param | Value | Description |
|-------|-------|-------------|
| SL_ATR_MULT | 1.0 | Stop loss = zone edge - 1.0x ATR |
| TP_RR | 2.5 | Take profit target (risk:reward) |
| MAX_OPEN_TRADES | 4 | Max concurrent positions |
| ZONE_CLUSTER_PCT | 0.8% | Merge zones within 0.8% |
| MIN_RR | 1.5 | Minimum RR to take trade |

## Key Metrics (--test output)

| Metric | Description |
|--------|-------------|
| Trades | Total closed trades |
| WR | Win rate (%) |
| Total R | Cumulative R-multiples (net of fees) |
| Avg RR | Average winning trade size |
| PF | Profit Factor (gross wins / gross losses) |
| MaxDD | Maximum drawdown in R |
| R/DD | Total R / MaxDD (risk-adjusted return) |
| R/Mo | Total R / months (time-normalized return, sort key) |
| Fees | Total fees paid in R |
| Dur | Median trade duration |
| Neg Mo | Negative months / total months |
