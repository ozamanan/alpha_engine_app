"""
Alpha Engine — Dashboard Demo

Backtest tabs use pre-computed sample data.
Paper Trading tab connects to Alpaca (bring your own API keys).
"""

import asyncio
import os
from datetime import datetime, date, timezone
from decimal import Decimal

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Alpha Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ #
# Async helper
# ------------------------------------------------------------------ #
def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
for key in [
    "backtest_results", "broker", "account", "positions",
    "order_history", "manual_orders",
]:
    if key not in st.session_state:
        st.session_state[key] = None
if "manual_orders" not in st.session_state:
    st.session_state.manual_orders = []


# ------------------------------------------------------------------ #
# Sample data generation (deterministic)
# ------------------------------------------------------------------ #
@st.cache_data
def generate_sample_data():
    """Generate realistic synthetic backtest results."""
    np.random.seed(42)
    dates = pd.bdate_range("2016-01-04", "2024-12-31")
    n = len(dates)

    strategies = {
        "Momentum": {"mu": 0.00065, "sigma": 0.014, "sharpe": 0.82},
        "Value": {"mu": 0.00035, "sigma": 0.011, "sharpe": 0.56},
        "Quality": {"mu": 0.00040, "sigma": 0.010, "sharpe": 0.71},
        "Multi-Factor": {"mu": 0.00055, "sigma": 0.011, "sharpe": 0.89},
    }

    results = {}
    for name, params in strategies.items():
        returns = np.random.normal(params["mu"], params["sigma"], n)
        # Add regime effects
        returns[500:600] -= 0.015  # drawdown period
        returns[1000:1020] -= 0.025  # covid crash
        returns[1020:1100] += 0.008  # recovery

        ret_series = pd.Series(returns, index=dates)
        equity = (1 + ret_series).cumprod() * 100_000

        cum = (1 + ret_series).cumprod()
        dd = cum / cum.cummax() - 1
        total_ret = float(cum.iloc[-1] - 1)
        years = n / 252
        cagr = float((1 + total_ret) ** (1 / years) - 1)
        vol = float(ret_series.std() * np.sqrt(252))
        sharpe = float(ret_series.mean() / ret_series.std() * np.sqrt(252))
        sortino_downside = ret_series[ret_series < 0].std() * np.sqrt(252)
        sortino = float(ret_series.mean() * 252 / sortino_downside) if sortino_downside > 0 else 0
        max_dd = float(dd.min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        win_rate = float((ret_series > 0).sum() / len(ret_series))

        results[name] = {
            "returns": ret_series,
            "equity": equity,
            "metrics": {
                "total_return": total_ret,
                "cagr": cagr,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "max_drawdown": max_dd,
                "volatility": vol,
                "win_rate": win_rate,
            },
        }

    # SPY benchmark
    spy_ret = np.random.normal(0.00045, 0.012, n)
    spy_ret[1000:1020] -= 0.030
    spy_ret[1020:1080] += 0.006
    benchmark = pd.Series(spy_ret, index=dates)

    return results, benchmark


# ------------------------------------------------------------------ #
# Plotting helpers
# ------------------------------------------------------------------ #
def plot_equity_curves(results, benchmark=None):
    fig = go.Figure()
    for name, data in results.items():
        eq = data["equity"]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=name, mode="lines"
        ))
    if benchmark is not None:
        spy_eq = (1 + benchmark).cumprod() * 100_000
        fig.add_trace(go.Scatter(
            x=spy_eq.index, y=spy_eq.values, name="SPY",
            mode="lines", line=dict(dash="dash", color="gray"),
        ))
    fig.update_layout(
        title="Equity Curves", xaxis_title="Date",
        yaxis_title="Portfolio Value ($)", hovermode="x unified",
        height=500, yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def plot_drawdowns(results):
    fig = go.Figure()
    for name, data in results.items():
        cum = (1 + data["returns"]).cumprod()
        dd = cum / cum.cummax() - 1
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, name=name,
            mode="lines", fill="tozeroy",
        ))
    fig.update_layout(
        title="Drawdowns", xaxis_title="Date",
        yaxis_title="Drawdown (%)", hovermode="x unified",
        height=400, yaxis=dict(ticksuffix="%"),
    )
    return fig


def plot_rolling_sharpe(results, window=252):
    fig = go.Figure()
    for name, data in results.items():
        ret = data["returns"]
        rolling = ret.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, name=name, mode="lines"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio",
        xaxis_title="Date", yaxis_title="Sharpe Ratio",
        hovermode="x unified", height=400,
    )
    return fig


def plot_correlation_heatmap(results):
    ret_df = pd.DataFrame({n: d["returns"] for n, d in results.items()})
    corr = ret_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Return Correlations",
    )
    fig.update_layout(height=400)
    return fig


def plot_monthly_returns(returns, name):
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({
        "year": monthly.index.year, "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="return")
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    pivot.columns = month_names[:len(pivot.columns)]
    fig = px.imshow(
        pivot * 100, text_auto=".1f", color_continuous_scale="RdYlGn",
        zmin=-10, zmax=10, title=f"{name} — Monthly Returns (%)",
        labels=dict(color="%"),
    )
    fig.update_layout(height=350)
    return fig


def plot_sensitivity_heatmap():
    lookbacks = [3, 6, 9, 12]
    top_ns = [10, 15, 20, 25, 30]
    np.random.seed(99)
    sharpe = np.array([
        [0.45, 0.52, 0.61, 0.58, 0.50],
        [0.55, 0.65, 0.72, 0.68, 0.60],
        [0.60, 0.70, 0.82, 0.78, 0.65],
        [0.58, 0.68, 0.80, 0.75, 0.62],
    ])
    fig = px.imshow(
        sharpe, text_auto=".2f", color_continuous_scale="Viridis",
        x=[str(n) for n in top_ns], y=[str(l) for l in lookbacks],
        title="Sensitivity: Sharpe Ratio",
        labels=dict(x="Top N", y="Lookback (months)", color="Sharpe"),
    )
    fig.update_layout(height=400)
    return fig


def fmt_pct(v):
    return f"{v * 100:.2f}%" if not pd.isna(v) else "N/A"


def fmt_ratio(v):
    return f"{v:.2f}" if not pd.isna(v) else "N/A"


# ------------------------------------------------------------------ #
# Alpaca helpers (paper trading)
# ------------------------------------------------------------------ #
def get_alpaca_client():
    """Get Alpaca TradingClient from secrets, env, or session state."""
    api_key = (
        os.getenv("ALPACA_API_KEY")
        or st.secrets.get("ALPACA_API_KEY", "")
        or st.session_state.get("alpaca_key", "")
    )
    api_secret = (
        os.getenv("ALPACA_API_SECRET")
        or st.secrets.get("ALPACA_API_SECRET", "")
        or st.session_state.get("alpaca_secret", "")
    )
    if not api_key or not api_secret:
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(api_key=api_key, secret_key=api_secret, paper=True)


async def fetch_account(client):
    loop = asyncio.get_running_loop()
    acct = await loop.run_in_executor(None, client.get_account)
    return {
        "equity": Decimal(str(acct.equity)),
        "cash": Decimal(str(acct.cash)),
        "buying_power": Decimal(str(acct.buying_power)),
        "portfolio_value": Decimal(str(acct.portfolio_value)),
        "long_market_value": Decimal(str(acct.long_market_value)),
    }


async def fetch_positions(client):
    loop = asyncio.get_running_loop()
    positions = await loop.run_in_executor(
        None, client.get_all_positions
    )
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pnl": float(p.unrealized_pl),
            "cost_basis": float(p.cost_basis),
        }
        for p in positions
    ]


async def fetch_open_orders(client):
    from alpaca.trading.requests import GetOrdersRequest
    loop = asyncio.get_running_loop()
    request = GetOrdersRequest(status="open")
    orders = await loop.run_in_executor(None, client.get_orders, request)
    return [
        {
            "symbol": o.symbol,
            "side": str(o.side).split(".")[-1],
            "qty": float(o.qty),
            "type": str(o.order_type or o.type).split(".")[-1],
            "status": str(o.status).split(".")[-1],
            "submitted_at": o.submitted_at,
        }
        for o in orders
    ]


async def submit_order(client, symbol, side, qty, order_type, limit_price=None):
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    sd = OrderSide.BUY if side == "BUY" else OrderSide.SELL

    if order_type == "MARKET":
        request = MarketOrderRequest(
            symbol=symbol, qty=qty, side=sd, time_in_force=TimeInForce.DAY,
        )
    else:
        request = LimitOrderRequest(
            symbol=symbol, qty=qty, side=sd,
            time_in_force=TimeInForce.DAY, limit_price=limit_price,
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, client.submit_order, request)
    return {
        "symbol": result.symbol,
        "side": str(result.side).split(".")[-1],
        "qty": float(result.qty),
        "filled_qty": float(result.filled_qty or 0),
        "filled_avg_price": float(result.filled_avg_price) if result.filled_avg_price else None,
        "status": str(result.status).split(".")[-1],
    }


async def cancel_all(client):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, client.cancel_orders)
    return len(result) if result else 0


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
st.sidebar.title("Alpha Engine")
st.sidebar.caption("Systematic Trading Research")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Demo mode**: Backtest tabs show sample data. "
    "Paper Trading connects to Alpaca if API keys are provided."
)
st.sidebar.markdown(
    "[View Project](https://mananoza.ai/projects/alpha_engine/) · "
    "[GitHub](https://github.com/ozamanan/alpha_engine)"
)

# ------------------------------------------------------------------ #
# Tabs
# ------------------------------------------------------------------ #
(
    tab_compare, tab_wf, tab_sensitivity, tab_drawdown,
    tab_holdings, tab_paper, tab_monitor,
) = st.tabs([
    "Strategy Comparison", "Walk-Forward", "Sensitivity", "Drawdowns",
    "Holdings", "Paper Trading", "Portfolio Monitor",
])

sample_results, sample_benchmark = generate_sample_data()

# ------------------------------------------------------------------ #
# Tab 1: Strategy Comparison
# ------------------------------------------------------------------ #
with tab_compare:
    if st.button("Run Backtests", key="run_bt_btn", type="primary"):
        st.session_state.backtest_results = sample_results

    results = st.session_state.backtest_results

    if results is None:
        st.info("Click **Run Backtests** to see strategy performance.")
    else:
        # Metrics table
        st.subheader("Performance Metrics")
        metrics_rows = {}
        for name, data in results.items():
            m = data["metrics"]
            metrics_rows[name] = {
                "Total Return": fmt_pct(m["total_return"]),
                "CAGR": fmt_pct(m["cagr"]),
                "Sharpe": fmt_ratio(m["sharpe_ratio"]),
                "Sortino": fmt_ratio(m["sortino_ratio"]),
                "Calmar": fmt_ratio(m["calmar_ratio"]),
                "Max Drawdown": fmt_pct(m["max_drawdown"]),
                "Volatility": fmt_pct(m["volatility"]),
                "Win Rate": fmt_pct(m["win_rate"]),
            }
        st.dataframe(
            pd.DataFrame(metrics_rows), use_container_width=True
        )

        # Equity curves
        st.plotly_chart(
            plot_equity_curves(results, sample_benchmark),
            use_container_width=True,
        )

        # Rolling Sharpe + Correlation
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_rolling_sharpe(results), use_container_width=True
            )
        with col2:
            st.plotly_chart(
                plot_correlation_heatmap(results), use_container_width=True
            )

        # Monthly returns
        st.subheader("Monthly Returns")
        selected = st.selectbox(
            "Strategy", list(results.keys()), key="monthly_sel"
        )
        if selected:
            st.plotly_chart(
                plot_monthly_returns(results[selected]["returns"], selected),
                use_container_width=True,
            )

# ------------------------------------------------------------------ #
# Tab 2: Walk-Forward
# ------------------------------------------------------------------ #
with tab_wf:
    wf_strat = st.selectbox(
        "Strategy", list(sample_results.keys()), key="wf_sel"
    )

    if st.button("Run Walk-Forward", key="wf_btn"):
        np.random.seed(hash(wf_strat) % 2**31)
        n_windows = 6
        windows = [f"W{i+1}" for i in range(n_windows)]
        train_sharpes = np.random.uniform(0.5, 1.2, n_windows)
        test_sharpes = train_sharpes * np.random.uniform(0.4, 0.9, n_windows)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train (IS)", x=windows, y=train_sharpes))
        fig.add_trace(go.Bar(name="Test (OOS)", x=windows, y=test_sharpes))
        fig.update_layout(
            barmode="group", title="Walk-Forward: Train vs Test Sharpe",
            yaxis_title="Sharpe Ratio", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Aggregate Out-of-Sample Performance")
        agg_df = pd.DataFrame({
            "Metric": ["sharpe_ratio", "cagr", "max_drawdown"],
            "Mean": [f"{test_sharpes.mean():.3f}", "12.4%", "-18.3%"],
            "Std": [f"{test_sharpes.std():.3f}", "4.1%", "5.2%"],
            "Min": [f"{test_sharpes.min():.3f}", "6.1%", "-28.7%"],
            "Max": [f"{test_sharpes.max():.3f}", "19.2%", "-9.1%"],
        })
        st.dataframe(agg_df.set_index("Metric"), use_container_width=True)

# ------------------------------------------------------------------ #
# Tab 3: Sensitivity
# ------------------------------------------------------------------ #
with tab_sensitivity:
    st.subheader("Momentum Parameter Sensitivity")
    st.caption("Sweeps lookback period and top-N positions")
    if st.button("Run Sensitivity Sweep", key="sens_btn"):
        st.plotly_chart(plot_sensitivity_heatmap(), use_container_width=True)

# ------------------------------------------------------------------ #
# Tab 4: Drawdowns
# ------------------------------------------------------------------ #
with tab_drawdown:
    results = st.session_state.backtest_results
    if results:
        st.plotly_chart(
            plot_drawdowns(results), use_container_width=True
        )
    else:
        st.info("Run the Strategy Comparison tab first.")

# ------------------------------------------------------------------ #
# Tab 5: Holdings
# ------------------------------------------------------------------ #
with tab_holdings:
    results = st.session_state.backtest_results
    if results:
        # Sample sector allocation
        sectors = [
            "Technology", "Healthcare", "Financials", "Consumer",
            "Industrials", "Energy", "Utilities",
        ]
        np.random.seed(77)
        dates = pd.bdate_range("2016-01-01", "2024-12-31", freq="ME")
        sector_data = {}
        for s in sectors:
            base = np.random.uniform(0.08, 0.20)
            sector_data[s] = base + np.random.normal(0, 0.02, len(dates))

        fig = go.Figure()
        for sector in sectors:
            fig.add_trace(go.Scatter(
                x=dates, y=np.clip(sector_data[sector], 0, 1) * 100,
                name=sector, mode="lines", stackgroup="one",
            ))
        fig.update_layout(
            title="Sector Exposure Over Time",
            xaxis_title="Date", yaxis_title="Weight (%)",
            hovermode="x unified", height=450,
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top holdings
        st.subheader("Top Holdings (Latest Rebalance)")
        top_symbols = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
            "AVGO", "LLY", "JPM", "V", "UNH", "XOM",
            "MA", "PG", "COST", "HD", "MRK", "ABBV",
            "WMT", "CRM",
        ]
        weights = np.random.dirichlet(np.ones(20)) * 100
        top_df = pd.DataFrame({
            "Symbol": top_symbols,
            "Weight": [f"{w:.1f}%" for w in sorted(weights, reverse=True)],
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the Strategy Comparison tab first.")

# ------------------------------------------------------------------ #
# Tab 6: Paper Trading
# ------------------------------------------------------------------ #
with tab_paper:
    client = get_alpaca_client()

    if client is None:
        st.subheader("Connect to Alpaca Paper Trading")
        st.caption(
            "Enter your Alpaca paper trading API keys. "
            "Get them free at [alpaca.markets](https://alpaca.markets)."
        )
        key_col1, key_col2 = st.columns(2)
        with key_col1:
            entered_key = st.text_input(
                "API Key", type="password", key="input_alpaca_key"
            )
        with key_col2:
            entered_secret = st.text_input(
                "API Secret", type="password", key="input_alpaca_secret"
            )
        if st.button("Connect", key="connect_keys_btn", type="primary"):
            if entered_key and entered_secret:
                st.session_state.alpaca_key = entered_key
                st.session_state.alpaca_secret = entered_secret
                st.rerun()
            else:
                st.error("Both API Key and Secret are required.")
    else:
        # Connect / refresh
        if st.button("Connect / Refresh", key="connect_btn"):
            st.session_state.account = run_async(fetch_account(client))
            st.session_state.positions = run_async(fetch_positions(client))

        acct = st.session_state.account
        positions = st.session_state.positions

        if acct is not None:
            # Account metrics
            st.subheader("Account")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Equity", f"${float(acct['equity']):,.2f}")
            m2.metric("Cash", f"${float(acct['cash']):,.2f}")
            m3.metric("Buying Power", f"${float(acct['buying_power']):,.2f}")
            m4.metric(
                "Long Market Value",
                f"${float(acct['long_market_value']):,.2f}",
            )

            # Positions + P&L charts
            st.subheader("Current Positions")
            active = [
                p for p in positions if p["qty"] > 0
            ]
            if active:
                pos_df = pd.DataFrame([
                    {
                        "Symbol": p["symbol"],
                        "Qty": p["qty"],
                        "Entry": f"${p['entry_price']:,.2f}",
                        "Current": f"${p['current_price']:,.2f}",
                        "Mkt Value": f"${p['market_value']:,.2f}",
                        "P&L": f"${p['unrealized_pnl']:,.2f}",
                        "P&L %": (
                            f"{p['unrealized_pnl'] / p['cost_basis'] * 100:+.2f}%"
                            if p["cost_basis"] != 0 else "—"
                        ),
                    }
                    for p in sorted(active, key=lambda x: x["symbol"])
                ])

                st.dataframe(
                    pos_df, use_container_width=True, hide_index=True
                )

                chart_l, chart_r = st.columns(2)
                with chart_l:
                    pnl_vals = [p["unrealized_pnl"] for p in active]
                    fig_pnl = go.Figure(go.Bar(
                        x=[p["symbol"] for p in active],
                        y=pnl_vals,
                        marker_color=[
                            "#2ecc71" if v >= 0 else "#e74c3c"
                            for v in pnl_vals
                        ],
                        text=[f"${v:+,.0f}" for v in pnl_vals],
                        textposition="outside",
                    ))
                    fig_pnl.update_layout(
                        title="Unrealized P&L by Position",
                        yaxis_title="P&L ($)",
                        yaxis=dict(tickformat="$,.0f"),
                        height=350, showlegend=False,
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

                with chart_r:
                    fig_alloc = go.Figure(go.Pie(
                        labels=[p["symbol"] for p in active],
                        values=[p["market_value"] for p in active],
                        hole=0.4, textinfo="label+percent",
                        textposition="outside",
                    ))
                    fig_alloc.update_layout(
                        title="Portfolio Allocation",
                        height=350, showlegend=False,
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True)
            else:
                st.info("No open positions")

            st.divider()

            # Manual trade
            st.subheader("Manual Trade")
            st.caption("Trade any ticker")

            tc1, tc2, tc3, tc4 = st.columns([2, 1, 1, 1])
            with tc1:
                manual_symbol = st.text_input(
                    "Symbol", placeholder="e.g. AAPL, TSLA",
                    key="manual_symbol",
                ).strip().upper()
            with tc2:
                manual_side = st.selectbox(
                    "Side", ["BUY", "SELL"], key="manual_side"
                )
            with tc3:
                manual_qty = st.number_input(
                    "Quantity", value=1, min_value=1, step=1,
                    key="manual_qty",
                )
            with tc4:
                manual_type = st.selectbox(
                    "Type", ["MARKET", "LIMIT"], key="manual_type"
                )

            manual_limit = None
            if manual_type == "LIMIT":
                manual_limit = st.number_input(
                    "Limit Price ($)", value=0.01, min_value=0.01,
                    step=0.01, format="%.2f", key="manual_limit",
                )

            if st.button("Submit Order", key="manual_btn"):
                if not manual_symbol:
                    st.error("Enter a symbol")
                else:
                    try:
                        with st.spinner(
                            f"Submitting {manual_side} {manual_qty} "
                            f"{manual_symbol}..."
                        ):
                            result = run_async(submit_order(
                                client, manual_symbol, manual_side,
                                manual_qty, manual_type, manual_limit,
                            ))

                        st.session_state.manual_orders.append(result)

                        if result["status"] == "filled":
                            st.success(
                                f"Filled: {manual_side} {result['filled_qty']} "
                                f"{manual_symbol} @ "
                                f"${result['filled_avg_price']:,.2f}"
                            )
                        else:
                            st.info(
                                f"Order **{result['status']}** — "
                                f"{manual_side} {manual_qty} {manual_symbol}. "
                                f"Will fill when market opens."
                            )
                    except Exception as e:
                        st.error(f"Order failed: {e}")
        else:
            st.info("Click **Connect / Refresh** to load account data.")

# ------------------------------------------------------------------ #
# Tab 7: Portfolio Monitor
# ------------------------------------------------------------------ #
with tab_monitor:
    client = get_alpaca_client()

    if client is None:
        st.info("Enter your Alpaca API keys in the **Paper Trading** tab to connect.")
    else:
        if st.button("Refresh", key="refresh_btn"):
            st.session_state.account = run_async(fetch_account(client))
            st.session_state.positions = run_async(fetch_positions(client))

        acct = st.session_state.account
        positions = st.session_state.positions

        if acct:
            total_pnl = sum(
                p["unrealized_pnl"] for p in (positions or [])
            )
            active = [p for p in (positions or []) if p["qty"] > 0]

            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Portfolio Value",
                f"${float(acct['portfolio_value']):,.2f}",
            )
            s2.metric(
                "Unrealized P&L",
                f"${total_pnl:,.2f}",
                delta=f"{total_pnl:+,.2f}",
            )
            s3.metric("Positions", len(active))

            # Positions
            st.subheader("Positions")
            if active:
                mon_df = pd.DataFrame([
                    {
                        "Symbol": p["symbol"],
                        "Qty": p["qty"],
                        "Avg Entry": f"${p['entry_price']:,.2f}",
                        "Current": f"${p['current_price']:,.2f}",
                        "Mkt Value": f"${p['market_value']:,.2f}",
                        "P&L": f"${p['unrealized_pnl']:,.2f}",
                        "P&L %": (
                            f"{p['unrealized_pnl'] / p['cost_basis'] * 100:+.2f}%"
                            if p["cost_basis"] != 0 else "—"
                        ),
                    }
                    for p in sorted(active, key=lambda x: x["symbol"])
                ])
                st.dataframe(
                    mon_df, use_container_width=True, hide_index=True
                )
            else:
                st.info("No open positions")

            # Open orders from broker
            st.subheader("Open Orders (Broker)")
            try:
                open_orders = run_async(fetch_open_orders(client))
            except Exception:
                open_orders = []

            if open_orders:
                st.caption(
                    f"{len(open_orders)} pending — "
                    f"will fill at market open"
                )
                oo_df = pd.DataFrame([
                    {
                        "Symbol": o["symbol"],
                        "Side": o["side"].upper(),
                        "Qty": o["qty"],
                        "Type": o["type"].upper(),
                        "Status": o["status"].upper(),
                        "Submitted": (
                            o["submitted_at"].strftime("%Y-%m-%d %H:%M")
                            if o["submitted_at"] else "—"
                        ),
                    }
                    for o in open_orders
                ])
                st.dataframe(
                    oo_df, use_container_width=True, hide_index=True
                )

                if st.button(
                    "Cancel All Open Orders", key="cancel_all_btn"
                ):
                    with st.spinner("Canceling..."):
                        n = run_async(cancel_all(client))
                    st.success(f"Canceled {n} orders")
                    st.rerun()
            else:
                st.info("No pending orders")

            # Session order history
            if st.session_state.manual_orders:
                st.subheader("Orders (This Session)")
                oh_df = pd.DataFrame(st.session_state.manual_orders)
                st.dataframe(
                    oh_df, use_container_width=True, hide_index=True
                )

            # Reconciliation
            st.subheader("Reconciliation")
            if st.button("Check Positions", key="recon_btn"):
                fresh = run_async(fetch_positions(client))
                st.session_state.positions = fresh
                st.success(
                    f"Synced {len([p for p in fresh if p['qty'] > 0])} "
                    f"positions from broker"
                )
        else:
            st.info("Click **Refresh** to load account data.")
