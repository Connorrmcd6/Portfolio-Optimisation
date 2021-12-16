import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from configs import tickers_list, train_start, train_stop, test_start, test_stop


df = yf.download(tickers_list, train_start, train_stop)['Adj Close']
df = df.dropna()

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)