# Data

`aapl.csv` - Apple Inc. daily OHLC price data, used only as an **out-of-distribution
test** in `notebooks/05_historical_validation.ipynb` (the real price path is replayed
as the simulator's fundamental).

Source: the public [`plotly/datasets`](https://github.com/plotly/datasets) repository
(`finance-charts-apple.csv`). Included here as a small, redistributable sample so the
notebook runs without a network fetch. If it is ever removed, the notebook falls back
to a synthetic fat-tailed series automatically.

The loader (`quant_mm.sim.load_price_series`) accepts any CSV with a close-price
column, so a LOBSTER mid-price series or a crypto tape drops straight in.
