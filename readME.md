# 🏦 Introducing PyBroker 📈
### PyBroker is a free and open-source Python framework that was designed with machine learning in mind and supports training machine learning models using your favorite ML framework.

# 🗝️ Key Features of PyBroker
### A lightning-fast backtesting engine built using NumPy and accelerated with Numba.
### The ability to create and execute trading rules and models across multiple instruments with ease.
### Access to historical data from Alpaca and Yahoo Finance, or from your own data provider.
### The option to train and backtest models using Walkforward Analysis, which simulates how the strategy would perform during actual trading.
### More reliable trading metrics that use randomized bootstrapping to provide more accurate results.
### Caching of downloaded data, indicators, and models to speed up your development process.
### Parallelized computations that enable faster performance.

```python 
pip install lib-pybroker
```

# Walkforward Analysis 👨‍💼
### PyBroker utilizes a robust algorithm called Walkforward Analysis to perform backtesting. Walkforward Analysis essentially divides your historical data into multiple time windows and then walks forward in time, emulating the process of executing and retraining the strategy with new data in the real world.

### During Walkforward Analysis, your model is initially trained on the earliest window and evaluated on the test data in that window. As the algorithm moves forward to evaluate the next time window, the test data from the previous window is added to the training data. This process is repeated until all the time windows are evaluated.

### Walkforward Analysis is also useful in addressing the issue of data mining and overfitting by testing your strategy on out-of-sample data.

# Walkforaward Analysis Example
### Let’s take a look at example code for an indicator that calculates the difference between close prices and a moving average (CMMA). This indicator could be helpful for a mean reversion strategy:
```python
import pybroker
import numpy as np
from numba import njit

def cmma(bar_data, lookback):

    @njit  # Enable Numba JIT.
    def vec_cmma(values):
        # Initialize the result array.
        n = len(values)
        out = np.array([np.nan for _ in range(n)])

        # For all bars starting at lookback:
        for i in range(lookback, n):
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            # Subtract the moving average from value.
            out[i] = values[i] - ma
        return out

    # Calculate with close prices.
    return vec_cmma(bar_data.close)
```
### We then register the indicator function with PyBroker and specify the lookback parameter as 20 days (bars):
```python
cmma_20 = pybroker.indicator('cmma_20', cmma, lookback=20)
```

### Next, we want to build a model that predicts the next day’s return using our 20-day CMMA indicator. A simple linear regression is a good approach to start with, and we can use the LinearRegression model from scikit-learn:

### The train_slr function uses the 20-day CMMA as the input feature, or predictor, for the LinearRegression model. The function then fits the LinearRegression model to the training data for that stock symbol.

### The final output of the train_slr function is the trained LinearRegression model for that specific stock symbol. PyBroker will use this model to predict the next day’s return of the stock during the backtest. The train_slr function will be called for each stock symbol, and the trained models will be used to predict the next day’s return for each individual stock.

### Then we register our training function with PyBroker, passing our cmma_20 indicator as training input:
```pthon
model_slr = pybroker.model(name='slr', fn=train_slr, indicators=[cmma_20])
```

### Now, let’s implement trading rules that generate buy and sell signals from our slr model:
```python
def hold_long(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('slr')[-1] > 0:
            ctx.buy_shares = 100
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('slr')[-1] < 0:
            ctx.sell_shares = 100
```

### The hold_long function opens a long position when the model predicts a positive return for the next bar, and then closes the position when the model predicts a negative return.

### The ctx.preds(‘slr’) method is used to access the predictions made by the 'slr' model for the current stock symbol being executed in the function. The predictions are stored in a NumPy array, and the most recent prediction for the current stock symbol is accessed using ctx.preds('slr')[-1], which is the model’s prediction of the next day’s return.

### We create a Strategy object that will train our model and run our trading rules on NVDA and AMD using data downloaded from Yahoo Finance:
```python
from pybroker import Strategy, StrategyConfig, YFinance

config = StrategyConfig(bootstrap_sample_size=100)
strategy = Strategy(YFinance(), '3/1/2017', '3/1/2022', config)
strategy.add_execution(hold_long, ['NVDA', 'AMD'], models=model_slr)
```

### Finally, we run our backtest using the Walkforward Analysis algorithm, using 3 time windows, each with a 50/50 train/test data split:
```python
result = strategy.walkforward(
    warmup=20,
    windows=3, 
    train_size=0.5, 
    calc_bootstrap=True
)
```
### The result contains trades and performance metrics from the backtest. There are 35 evaluation metrics in total, but here is a sample of a few:
```python
result.metrics_df
```
```python
result.bootstrap.conf_intervals
```

# Drawdown 👩‍💻
### PyBroker also uses the bootstrap method to calculate the maximum drawdown of the strategy. The probabilities of the drawdown not exceeding certain values, represented in cash and percentage of portfolio equity, are displayed below:
```python
result.bootstrap.drawdown_conf        
```
# Conclusion 🔚
### Obviously, our strategy needs a lot of improvement! But this should give you an understanding of how to train and evaluate a model in PyBroker.

### Please keep in mind that before conducting regression analysis, it is important to verify certain assumptions such as homoscedasticity, normality of residuals, etc. I have not provided the details for these assumptions here for the sake of brevity and recommend that you perform this exercise on your own.

### We are also not limited to just building linear regression models in PyBroker. We can train other model types such as gradient boosted machines, neural networks, or any other architecture that we choose with our preferred ML framework.

### With this knowledge, you can start building and testing your own models and trading strategies in PyBroker, and begin exploring the vast possibilities that this framework offers! Furthermore, I have written additional tutorials on using PyBroker and general algorithmic trading concepts that can be found on https://www.pybroker.com.

