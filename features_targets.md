Here goes the target features
 
# Things to try

TODO:
* lepsi labeling
* continue here https://claude.ai/chat/b3ee78b6-9662-4f25-95f0-ecac4a78a41b
* try model with other symbols
* rey different retraining options (even hourly)

Features:
- add datetime features (useful for rush hour model)
- add MT features as columns
- use convolutional networks to create features (https://www.youtube.com/watch?v=6wK4q8QvsV4)
Enhance model:
* multi target see xgb doc
* use SL with target price, with validy for few seconds
* how handle imbalanced datase https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html


Target:
- maybe add manual labeling

# Features

```python

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """Prepare enhanced features from input df with focus on predictive potential"""
        features = pd.DataFrame(index=df.index)

        # Original ohlcv added to features
        features['close'] = df['close']
        features['volume'] = df['volume']
        features['trades_count'] = df['trades']
        features['buy_volume'] = df['buyvolume']
        features['sell_volume'] = df['sellvolume']
        features['high'] = df['high']
        features['low'] = df['low']
        # features['log_return'] = np.log(features['close'] / features['close'].shift(1))
        # features['returns_1'] = features['close'].pct_change()
        # features['returns_5'] = features['close'].pct_change(5)
        # features['returns_20'] = features['close'].pct_change(20)

        def get_fib_windows():
            """
            #TODO based on real time (originally for 1s bars)

            Generate Fibonacci sequence windows up to ~1 hour (3600 seconds)
            Returns sequence: 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584
            """
            fib_windows = [3, 5]
            while fib_windows[-1] < 3600/60:
                next_fib = fib_windows[-1] + fib_windows[-2]
                if next_fib > 3600/60:
                    break
                fib_windows.append(next_fib)
            return fib_windows

        fib_windows = get_fib_windows()
        
        # Base price and returns
        features['log_return'] = np.log(features['close'] / features['close'].shift(1))
        features['price_velocity'] = (features['close'] - features['close'].shift(1)) / 1.0  # per second
        features['price_acceleration'] = features['price_velocity'] - features['price_velocity'].shift(1)
        
        # Fibonacci-based features
        for window in fib_windows:
            # Price features
            features[f'log_return_{window}s'] = np.log(features['close'] / features['close'].shift(window))
            features[f'volatility_{window}s'] = features['log_return'].rolling(window).std()
            features[f'range_{window}s'] = (features['high'].rolling(window).max() - 
                                        features['low'].rolling(window).min()) / features['close']
            
            # Volume features
            features[f'volume_momentum_{window}s'] = (
                features['volume'].rolling(window).mean() / 
                features['volume'].rolling(window * 2).mean()
            )
            
            features[f'buy_volume_momentum_{window}s'] = (
                features['buy_volume'].rolling(window).mean() / 
                features['buy_volume'].rolling(window * 2).mean()
            )
            
            features[f'sell_volume_momentum_{window}s'] = (
                features['sell_volume'].rolling(window).mean() / 
                features['sell_volume'].rolling(window * 2).mean()
            )
            
            # Trade features
            features[f'trade_intensity_{window}s'] = (
                features['trades_count'].rolling(window).mean() / 
                features['trades_count'].rolling(window * 2).mean()
            )
            
            features[f'avg_trade_size_{window}s'] = (
                features['volume'].rolling(window).sum() / 
                features['trades_count'].rolling(window).sum()
            )
            
            # Order flow features
            features[f'cum_volume_delta_{window}s'] = (
                features['buy_volume'] - features['sell_volume']
            ).rolling(window).sum()
            
            features[f'volume_pressure_{window}s'] = (
                features['buy_volume'].rolling(window).sum() / 
                features['sell_volume'].rolling(window).sum()
            )
            
            # Price efficiency
            features[f'price_efficiency_{window}s'] = (
                np.abs(features['close'] - features['close'].shift(window)) /
                (features['high'].rolling(window).max() - features['low'].rolling(window).min())
            )
            
            # Moving averages and their crosses
            features[f'sma_{window}s'] = features['close'].rolling(window).mean()
            if window > 5:  # Create MA crosses with shorter timeframe
                features[f'ma_cross_5_{window}s'] = (
                    features['close'].rolling(5).mean() - 
                    features['close'].rolling(window).mean()
                )
        
        # MA-based features
        ma_lengths = [5, 10, 20, 50]
        for length in ma_lengths:
            # Regular MAs
            features[f'ma_{length}'] = features['close'].rolling(length).mean()
            
            # MA slopes (rate of change)
            features[f'ma_{length}_slope'] = features[f'ma_{length}'].pct_change(3)
            
            # Price distance from MA
            features[f'price_ma_{length}_dist'] = (features['close'] - features[f'ma_{length}']) / features[f'ma_{length}']
            
            # MA crossovers
            if length > 5:
                features[f'ma_5_{length}_cross'] = (features['ma_5'] - features[f'ma_{length}']) / features[f'ma_{length}']
        
        # MA convergence/divergence
        features['ma_convergence'] = ((features['ma_5'] - features['ma_20']).abs() / 
                                    features['ma_20'].rolling(10).mean())
        
        # Volatility features using MAs
        features['ma_volatility'] = features['ma_5'].rolling(10).std() / features['ma_20']
        
        # MA momentum
        features['ma_momentum'] = (features['ma_5'] / features['ma_5'].shift(5) - 1) * 100
          
        
        # Cleanup and feature selection
        features = features.replace([np.inf, -np.inf], np.nan)
        
        lookback = 1000
        if len(features) > lookback:
            rolling_corr = features.iloc[-lookback:].corr().abs()
            upper = rolling_corr.where(np.triu(np.ones(rolling_corr.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            print(f"Column highly correlated - maybe drop? {to_drop} ")
            #features = features.drop(columns=to_drop)
        
        feature_columns = list(features.columns)
        print(f"Features shape before dropna: {features.shape}")
        
        return features.dropna(), feature_columns
```



# Targets
## Unbalanced classes

```python
from xgboost import XGBClassifier

# Compute scale_pos_weight
n_0 = sum(y_train == 0)
n_1 = sum(y_train == 1)
scale_pos_weight = n_0 / n_1

model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
```


```python
    def create_target_regressor(self, df: pd.DataFrame) -> pd.Series:
        """
        https://claude.ai/chat/8e7fe81c-ddbe-4e64-9af0-2bc4764fc5f0

        Creates enhanced target variable using adaptive returns based on market conditions.
        Key improvements:
        1. Multi-timeframe momentum approach
        2. Volume-volatility regime adaptation
        3. Trend-following vs mean-reversion regime detection
        4. Noise reduction through sophisticated filtering
        
        Parameters:
        -----------
        df : pd.DataFrame
            Features df containing required columns: 'close', 'volume', volatility features
        
        Returns:
        --------
        pd.Series
            Enhanced target variable with cross-day targets removed
        """

        future_bars= self.config.forward_bars

        future_ma_fast = df['close'].shift(-future_bars).rolling(5).mean()
        
        # Calculate forward returns (original approach)
        forward_returns = df['close'].shift(-future_bars) / df['close'] - 1
        
        target =  forward_returns

       # 6. Advanced noise reduction
        # Use exponential moving standard deviation for dynamic thresholds
        target_std = target.ewm(span=50, min_periods=20).std()
        
        # Adaptive thresholds based on rolling standard deviation
        upper_clip = 2.5 * target_std
        lower_clip = -2.5 * target_std
        
        # Apply soft clipping using hyperbolic tangent
        target = target_std * np.tanh(target / target_std)
        
        # Final hard clips for extreme outliers
        target = target.clip(lower=lower_clip, upper=upper_clip)


        # 7. Remove cross-day targets and intraday seasonality
        target = self.remove_crossday_targets(target, df, future_bars)

        #only 10% of extreme values from both sides are kept
        #target = target.where((target > target.quantile(0.9)) | (target < target.quantile(0.1)), 0)

        print("after target generation", target.index[[0, -1]])
        
        return target
```