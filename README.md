# Alpha Engine — Demo Dashboard

Interactive demo of the Alpha Engine systematic trading platform.

**Backtest tabs** show sample data generated from synthetic factor strategies.
**Paper Trading tabs** connect to Alpaca (bring your own API keys).

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Alpaca Paper Trading

Create `.streamlit/secrets.toml`:

```toml
ALPACA_API_KEY = "your_paper_key"
ALPACA_API_SECRET = "your_paper_secret"
```

Get paper trading keys at [alpaca.markets](https://alpaca.markets).

## Links

- [Project Page](https://mananoza.ai/projects/alpha_engine/)
- [Source (private)](https://github.com/ozamanan/alpha_engine)
