- [FETCHING DATA](#fetching-data)
- [DISCOVERY](#discovery)
- [DATA/WRAPPER](#datawrapper)
  - [create WRAPPER manually](#create-wrapper-manually)
- [RESAMPLING](#resampling)
  - [config](#config)
- [REALIGN](#realign)
  - [REALIGN\_CLOSING accessors](#realign_closing-accessors)
- [SIGNALS](#signals)
  - [ENTRIES/EXITS time based](#entriesexits-time-based)
  - [STOPS](#stops)
  - [OHLCSTX modul](#ohlcstx-modul)
  - [WINDOW OPEN/CLOSE](#window-openclose)
  - [END OF DAY EXITS](#end-of-day-exits)
- [DF/SR ACCESSORS](#dfsr-accessors)
  - [Generic](#generic)
  - [SIGNAL ACCESSORS](#signal-accessors)
  - [RANKING - partitioning](#ranking---partitioning)
  - [Base Accessors](#base-accessors)
- [Stoploss/Takeprofit](#stoplosstakeprofit)
  - [SL - ATR based](#sl---atr-based)
  - [EXIT after time](#exit-after-time)
  - [CALLBACKS -](#callbacks--)
    - [MEMORY](#memory)
- [INDICATORS DEV](#indicators-dev)
- [FAV INDICATORS](#fav-indicators)
- [GROUPING - SPLITTING](#grouping---splitting)
- [CHARTING](#charting)
- [MULTIACCOUNT](#multiaccount)
- [CUSTOM SIMULATION](#custom-simulation)
- [ANALYSIS](#analysis)
  - [ROBUSTNESS](#robustness)
- [UTILS](#utils)


```python
import vectorbtpro as vbt
from lightweight_charts import Panel, chart, PlotDFAccessor, PlotSRAccessor
t15data = None

if not hasattr(pd.Series, 'lw'):
    pd.api.extensions.register_series_accessor("lw")(PlotSRAccessor)

if not hasattr(pd.DataFrame, 'lw'):
    pd.api.extensions.register_dataframe_accessor("lw")(PlotDFAccessor)
```


# FETCHING DATA  
 #fetching from remote db
from lib.db import Connection
SYMBOL = "BAC"
SCHEMA = "ohlcv_1s" #time based 1s other options ohlcv_vol_200 (volume based ohlcv with resolution of 200), ohlcv_renko_20 (renko with 20 bricks size) ...
DB = "market_data"

con = Connection(db_name=DB, default_schema=SCHEMA, create_db=True)
basic_data = con.pull(symbols=[SYMBOL], schema=SCHEMA,start="2024-08-01", end="2024-08-08", tz_convert='America/New_York')


#Fetching from YAHOO
symbols = ["AAPL", "MSFT", "AMZN", "TSLA", "AMD", "NVDA", "SPY", "QQQ", "META", "GOOG"]
data = vbt.YFData.pull(symbols, start="2024-09-28", end="now", timeframe="1H", missing_columns="nan")

#endregion

# DISCOVERY

#get parameters of method
vbt.IF.list_locations() #lists categories
vbt.IF.list_indicators(pattern="vbt") #all in category vbt
vbt.IF.list_indicators("*sma")
vbt.phelp(vbt.indicator("talib:MOM").run)


# DATA/WRAPPER

Available [methods for data](http://5.161.179.223:8000/vbt-doc/api/data/base/index.html#vectorbtpro.data.base.Data)

**Main data container** (servees as a wrapper for symbol oriented or feature oriented data)
```python
data.transform()
data.dropna()
data.feature_oriented vs data.symbol_oriented #returns True/False if cols are features or symbols
data.data #dictionary either feature oriented or
data.ohlcv #OHLCV mixin filters only ohlcv feature and offers methods http://5.161.179.223:8000/vbt-doc/api/data/base/index.html#vectorbtpro.data.base.OHLCDataMixin
data.base #base mixin - implicit offers functions wrapper methods  http://5.161.179.223:8000/vbt-doc/api/data/base/index.html#vectorbtpro.data.base.BaseDataMixin
    - data.symbol_wrapper
    - data.feature_wrapper
    - data.features


show(t1data.data["BAC"])

#display returns on top of ohlcv
t1data.ohlcv.data["BAC"].lw.plot(left=[(t1data.returns, "returns")], precision=4)
```

## create WRAPPER manually

[wrapper methods](http://5.161.179.223:8000/vbt-doc/api/base/wrapping/index.html#vectorbtpro.base.wrapping.ArrayWrapper)

```python
#create wrapper from existing objects
wrapper = data.symbol_wrapper # one column for each symbol
wrapper = data.get_symbol_wrapper() # symbol - level, one column for each symbol  (BAC a pod tim series)
wrapper = data.get_feature_wrapper() #feature level, one column for each feature (open,high...)
wrapper = df.vbt.wrapper

#Create an empty array with the same shape, index, and columns as in another array
new_float_df = wrapper.fill(np.nan)  
new_bool_df = wrapper.fill(False)  
new_int_df = wrapper.fill(-1)  

#display df/series
from itables import show
show(t1data.close)
```

# RESAMPLING
## config
```python
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro import _typing as tp
from vectorbtpro.generic import nb as generic_nb

_feature_config: tp.ClassVar[Config] = HybridConfig(
    {
        "buyvolume": dict(
            resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                resampler,
                generic_nb.sum_reduce_nb,
            )
        ),
        "sellvolume": dict(
            resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                resampler,
                generic_nb.sum_reduce_nb,
            )
        ),
        "trades": dict(
            resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                resampler,
                generic_nb.sum_reduce_nb,
            )
        )
    }
)

basic_data._feature_config = _feature_config
```
ddd
#1s to 1T
t1data = basic_data[['open', 'high', 'low', 'close', 'volume','vwap','buyvolume','trades','sellvolume']].resample("1T")
t1data = t1data.transform(lambda df: df.between_time('09:30', '16:00').dropna())

#using resampler (with more control over target index)
resampler_s = vbt.Resampler(target_data.index, source_data.index, source_freq="1T", target_freq="1s")
basic_data.resample(resampler_s)


# REALIGN

`REALIGN` method - runs on data object (OHLCV) - (open feature realigns leftbound, rest of features rightboud) .resample("1T").first().ffill()

```python
ffill=True = same frequency as t1data.index
ffill=False = keeps original frequency but moved to where data are available ie. instead of 15:30 to 15:44 for 15T bar
t15data_realigned = t15data.realign(t1data.index, ffill=True, freq="1T") #freq - target frequency
```

## REALIGN_CLOSING accessors
```python
t15data_realigned_close = t15data.close.vbt.realign_closing(t1data.index, ffill=True, freq="1T")
t15data_realigned_open = t15data.open.vbt.realign_open(t1data.index, ffill=True, freq="1T")
```

#realign_closing accessor just calls
#return self.realign(*args, source_rbound=False, target_rbound=False, **kwargs)
#realign opening
#return self.realign(*args, source_rbound=True, target_rbound=True, **kwargs)

#using RESAMPLER
#or
resampler_s = vbt.Resampler(t15data.index, t1data.index, source_freq="1T", target_freq="1s")
t15close_realigned_with_resampler = t1data.data["BAC"].realign_closing(resampler_s)


# SIGNALS
```python
cond1 = data.get("Low") < bb.lowerband
#comparing with previous value
cond2 = bandwidth > bandwidth.shift(1)
#comparing with value week ago  
cond2 = bandwidth > bandwidth.vbt.ago("7d")
mask = cond1 & cond2
mask.sum()
```

## ENTRIES/EXITS time based
```python
#create entries/exits based on open of first symbol
entries = pd.DataFrame.vbt.signals.empty_like(data.open.iloc[:,0]) 

#create entries/exits based on symbol level
symbol_wrapper = data.get_symbol_wrapper()
entries = symbol_wrapper.fill(False)
exits = symbol_wrapper.fill(False)

entries.vbt.set(
    True, 
    every="W-MON", 
    at_time="00:00:00",
    indexer_method="bfill",  # this time or after
    inplace=True
)
exits.vbt.set(
    True, 
    every="W-MON", 
    at_time="23:59:59", 
    indexer_method="ffill",  # this time or before
    inplace=True
)
```


## STOPS
[doc from_signal](http://5.161.179.223:8000/vbt-doc/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 

- StopExitPrice (Which price to use when exiting a position upon a stop signal?)
- StopEntryPrice (Which price to use as an initial stop price?)
                  
price = close.vbt.wrapper.fill()
price[entries] = entry_price
price[exits] = exit_price

## OHLCSTX modul
 - exit signal generator based on price and stop values
[doc](ttp://5.161.179.223:8000/vbt-doc/api/signals/generators/ohlcstx/index.html)


## WINDOW OPEN/CLOSE




## END OF DAY EXITS
```python
sr = t1data.data["BAC"]
last_n_daily_rows = sr.groupby(sr.index.date).tail(4) #or N last rows
second_last_daily_row = sr.groupby(sr.index.date).nth(-2) #or Nth last row
second_last_two_rows = sr.groupby(sr.index.date).apply(lambda x: x.iloc[-3:-1]).droplevel(0) #or any slice of rows
#create exit array
exits = t1data.get_symbol_wrapper().fill(False)
exits.loc[last_n_daily_rows.index] = True
#visualize
t1data.ohlcv.data["BAC"].lw.plot(right=[(t1data.close,"close",exits)], size="s")

#REGULAR EXITS -EVERY HOUR/D/WEEK exits
exits.vbt.set(
    True, 
    every="H" # "min" "2min" "2H" "W-MON"+at time "D"+time
    #at_time="23:59:59", 
    indexer_method="ffill",  # this time or before
    inplace=True
)

```

# DF/SR ACCESSORS 

## Generic
- for common taks ([docs](http://5.161.179.223:8000/vbt-doc/api/generic/accessors/index.html#vectorbtpro.generic.accessors.GenericAccessor)) 

 `rolling_apply` - runs custom function over a rolling window of a fixed size (number of bars or frequency)

`expanding_apply` - runs custome function over expanding the window from the start of the data to the current poin

```python
from numba import njit
mean_nb = njit(lambda a: np.nanmean(a))
hourly_anchored_expanding_mean = t1data.close.vbt.rolling_apply("1H", mean_nb) #ROLLING to FREQENCY or with fixed windows rolling_apply(10,mean_nb)
t1data.ohlcv.data["BAC"].lw.plot(right=[(t1data.close,"close"),(hourly_anchored_expanding_mean, "hourly_anchored_expanding_mean")], size="s")
#NOTE for anchored "1D" frequency - it measures timedelta that means requires 1 day between reseting (16:00 end of market, 9:30 start - not a full day, so it is enOugh to set 7H)

df['a'].vbt.overlay_with_heatmap(df['b']).show()
```

## SIGNAL ACCESSORS
- http://5.161.179.223:8000/vbt-doc/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsAccessor


## RANKING - partitioning
```python
#pos_rank -1 when False, 0, 1 ... for consecutive Trues, allow_gaps defautlne False
# sample_mask = pd.Series([True, True, False, True, True])
ranked = sample_mask.vbt.signals.pos_rank()
ranked == 1 #select each second signal in each partition
ranked = sample_mask.vbt.signals.pos_rank(allow_gaps=True)
(ranked > -1) & (ranked % 2 == 1) #Select each second signal globally

entries.vbt.signals.first() #selects only first entries in each group
entries.vbt.signals.from_nth(n) # pos_rank >= n in each group, all from Nth

#AFTER - with variants _after which resets partition each reset array
#maximum number of exit signals after each entry signal
exits.vbt.signals.pos_rank_after(entries, reset_wait=0).max() + 1 #Count is the maximum rank plus one since ranks start with zero. We also assume that an entry signal comes before an exit signal if both are at the same timestamp by passing reset_wait=0.

entries.vbt.signals.total_partitions


#partition_pos_rank - all members of each partition have the same rank
ranked = sample_mask.vbt.signals.partition_pos_rank(allow_gaps=True) #0,0,-1,1,1
ranked == 1 # the whole second partition
```

## Base Accessors
* low level accessors - http://5.161.179.223:8000/vbt-doc/api/base/accessors/index.html#vectorbtpro.base.accessors.BaseAccessor

```python
exits.vbt.set(
    True, 
    every="W-MON", 
    at_time="23:59:59", 
    indexer_method="ffill",  # this time or before
    inplace=True
)
```

# Stoploss/Takeprofit
[doc StopOrders](http://5.161.179.223:8000/vbt-doc/documentation/portfolio/from-signals/index.html#stop-orders)

## SL - ATR based
```
atr = data.run("atr").atr
pf = vbt.Portfolio.from_signals(
    data,
    entries=entries,
    sl_stop=atr / sub_data.close
)
```

## EXIT after time
using [from signals](http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#from-signals)

```python
f = vbt.PF.from_signals(..., td_stop="7 days")  
pf = vbt.PF.from_signals(..., td_stop=pd.Timedelta(days=7))
pf = vbt.PF.from_signals(..., td_stop=td_arr) 
#EXIT at time
pf = vbt.PF.from_signals(..., dt_stop="16:00")  #exit at 16 and later
pf = vbt.PF.from_signals(..., dt_stop=datetime.time(16, 0))
pf = vbt.PF.from_signals(  #exit last bar before
    ..., 
    dt_stop="16:00", 
    arg_config=dict(dt_stop=dict(last_before=True))
)
```

## CALLBACKS - 
  - a signal function (`signal_func_nb`)
    - can dynamically generate signals (True, True, False,False)
    - runs at beginning of bar
 - an adjustment function (`adjust_func_nb`) - [doc](http://5.161.179.223:8000/vbt-doc/documentation/portfolio/from-signals/#adjustment)
    - runs only if signal function above was not provided, but entry,exit arrays
    - runs before default signal function [ls_signal_func_nb](http://5.161.179.223:8000/vbt-doc/api/portfolio/nb/from_signals/index.html#vectorbtpro.portfolio.nb.from_signals.ls_signal_func_nb)
        - can change pending limit orders etc.
 - a post-signal function (`post_signal_func_nb`)
 - post-segment function (`post_segment_func_nb`)

all of them are accessing [SignalContext](http://5.161.179.223:8000/vbt-doc/api/portfolio/enums/index.html#vectorbtpro.portfolio.enums.SignalContext) `(c)` as named tuple 
SignalContaxt (contains various metrics) such as:
* last_limit_info - 1D with latest limit order per column
* order_counts
* last_return ...
    
"""

### MEMORY
save an information piece at one timestamp and re-use at a later timestamp when using [callbacks memory](http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks)

Usecases:
* [MULTIPLE simultaneuos LIMIT ORDERS at TIME](http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks)

* [IGNORE ENTRIES number of DAYS after losing trade](http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks) - signal function 




# INDICATORS DEV

```python
#REGISTER CUSTOM INDICATOR
vbt.IndicatorFactory.register_custom_indicator(
    SupportResistance,
    name="SUPPRES",
    location=None,
    if_exists='raise'
)

#RUN INDICATOR on DATA WRAPPER
cdlbreakaway = s1data.run(vbt.indicator("talib:CDLHAMMER"), skipna=True, timeframe=["12s"])

#FROM EXPRESSION http://5.161.179.223:8000/vbt-doc/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr
WMA = vbt.IF(
    class_name='WMA',
    input_names=['close'],
    param_names=['window'],
    output_names=['wma']
).from_expr("wm_mean_nb(close, window)")

wma = WMA.run(t1data.close, window=10)
wma.wma
```


# FAV INDICATORS 

```python
#for TALIB indicator always use skipna=True

#TALIB INDICATORS can do realing closing : timeframe=["1T"]
mom_multi = vbt.indicator("talib:MOM").run(t1data.close, timeperiod=5, timeframe=["1T","5T"], skipna=True) #returned 5T can be directly compared with 1T

#ANCHORED indciators vbt.indicator("talib:MOM") becomes AnchoredIndicator("talib:MOM", anchor="D") - freq of pd.Grouper
from ttools import AnchoredIndicator
mom_anch_d = AnchoredIndicator("talib:MOM", anchor='30min').run(t1data.data["BAC"].close, timeperiod=10)
mom = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close, timeperiod=10, skipna=True)
t1data.ohlcv.data["BAC"].lw.plot(auto_scale=[mom_anch_d, mom])

#FIBO RETRACEMENT
fibo = vbt.indicator("technical:FIBONACCI_RETRACEMENTS").run(t1data.close, skipna=True)
#fibo.fibonacci_retracements

fibo_plusclose = t1data.close + fibo.fibonacci_retracements
fibo_minusclose = t1data.close - fibo.fibonacci_retracements
#fibo_plusclose
Panel(
            auto_scale=[fibo_plusclose["BAC"]],
            ohlcv=(t1data.ohlcv.data["BAC"],),
            histogram=[],
            right=[(fibo_plusclose["BAC"],),(fibo_minusclose["BAC"],)],
            left=[],
            middle1=[(fibo.fibonacci_retracements["BAC"],"fibonacci_retracements")],
            middle2=[]
            ).chart(size="xs")

#CHOPINESS indicator
chopiness = vbt.indicator("technical:CHOPINESS").run(s1data.open, s1data.high, s1data.low, s1data.close, t1data.volume, skipna=True)
s1data.ohlcv.data["BAC"].lw.plot(auto_scale=[chopiness])

#anchored VWAP
t1vwap_h = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="H")
t1vwap_h_real = t1vwap_h.vwap.vbt.realign_closing(resampler_s)

#BBANDS = vbt.indicator("pandas_ta:BBANDS")
mom_anch_d = AnchoredIndicator("talib:MOM", anchor='30min').run(t1data.data["BAC"].close, timeperiod=10)
mom = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close, timeperiod=10, skipna=True)
#macd = vbt.indicator("talib:MACD").run(t1data.data["BAC"].close) #, timeframe=["1T"]) #, 
t1data.ohlcv.data["BAC"].lw.plot(auto_scale=[mom_anch_d, mom])

```

# GROUPING - SPLITTING

```python
#SPLITTER - splitting wrapper based on index
#http://5.161.179.223:8000/vbt-doc/tutorials/cross-validation/splitter/index.html#anchored
daily_splitter = vbt.Splitter.from_grouper(t1data.index, "D", split=None) #DOES contain last DAY

daily_splitter = vbt.Splitter.from_ranges( #doesnt contain last DY
    t1data.index,
    every="D",  
    split=None
)
daily_splitter.stats()
daily_splitter.plot()
daily_splitter.coverage()

#TAKING and APPLY MANUALLY - run UDF on ALL takes and concatenates
taken = daily_splitter.take(t1data)
inds = []
for series in taken:
    mom = vbt.indicator("talib:MOM").run(series.close, timeperiod=10, skipna=True)
    inds.append(mom)
mom_daily = vbt.base.merging.row_stack_merge(inds) #merge
mom = vbt.indicator("talib:MOM").run(t1data.close, timeperiod=10, skipna=True)
t1data.ohlcv.data["BAC"].lw.plot(left=[(mom_daily, "daily_splitter"),(mom, "original mom")]) #OHLCV with indicators on top

#TAKING and APPLY AUTOMATIC
daily_splitter = vbt.Splitter.from_grouper(t1data.index, "D", split=None) #DOES contain last DAY
def indi_run(sr):
    return vbt.indicator("talib:MOM").run(sr.close, timeperiod=10, skipna=True)

res = daily_splitter.apply(indi_run, vbt.Takeable(t1data), merge_func="row_stack", freq="1T") 


#use of IDX accessor (docs:http://5.161.179.223:8000/vbt-doc/api/base/accessors/index.html#vectorbtpro.base.accessors.BaseIDXAccessor)
daily_grouper = t1data.index.vbt.get_grouper("D")

#grouper instance can be iterated over
for name, indices in daily_grouper.iter_groups():
    print(name, indices)

#PANDAS GROUPING - series/df grouping resulting in GroupBySeries placeholder that can be aggregated(sum, mean), transformed iterated over or fitlered
for name, group in t1data.data["BAC"].close.groupby(pd.Grouper(freq='D')):
    print(name, group)
```


# CHARTING
Using [custom lightweight-charts-python](https://github.com/drew2323/lightweight-charts-python)

```python
#LW df/sr accessor
t1data.ohlcv.data["BAC"].lw.plot(left=[(mom_multi, "mom_multi")]) #OHLCV with indicators on top


t5data.ohlcv.data["BAC"].lw.plot(
    left=[(mom_multi.real, "mom"),(mom_multi_beztf, "mom_beztf"), (mom_5t_orig, "mom_5t_orig"), (mom_5t_orig_realigned, "mom_5t_orig_realigned")],
    right=[(t1data.data["BAC"].close, "t1 close"),(t5data.data["BAC"].close, "t5 close")],
    size="s") #.loc[:,(20,"1T","BAC")]

#SINGLE PANEL
Panel(
            auto_scale=[cdlbreakaway],
            ohlcv=(t1data.ohlcv.data["BAC"],entries),
            histogram=[],
            right=[],
            left=[],
            middle1=[],
            middle2=[]
            ).chart(size="xs")

#MULTI PANEL
pane1 = Panel(
    #auto_scale=[mom_multi, mom_multi_1t],
    #ohlcv=(t1data.data["BAC"],),  #(series, entries, exits, other_markers)
    #histogram=[(order_imbalance_allvolume, "oivol")], # [(series, name, "rgba(53, 94, 59, 0.6)", opacity)]
    right=[(t1data.data["BAC"].close,"close 1T"),(t5data.data["BAC"].close,"close 5T"),(mom_multi_1t.close, "mom multi close")], # [(series, name, entries, exits, other_markers)]
    left=[(mom_multi, "mom_multi"), (mom_multi_1t, "mom_multi_1t")],
    #middle1=[],
    #middle2=[],
    #xloc="2024-02-12 09:30",
    precision=3
)
pane2 = Panel(....)

ch = chart([pane1, pane2], size="s")
```


# MULTIACCOUNT
Simultaneous LONG and short (hedging)
In vbt position requires one column of data, so hedging is possible by using two columns representing the same asset but different directions,
then stack both portfolio together [column stacking](http://5.161.179.223:8000/vbt-doc/features/productivity/#column-stacking)
pf_join = vbt.PF.column_stack((pf1, pf2), group_by=True)

# CUSTOM SIMULATION

# ANALYSIS
## ROBUSTNESS
```python
pf_stats.sort_values(by='Sharpe Ratio', ascending=False).iloc[::-1].vbt.heatmap().show() #works when there are more metrics
```


#endregion

# UTILS

```python
#MEMORY
sr.info()


#peak memory usage, running once
with vbt.MemTracer() as tracer:
    my_pipeline()

print(tracer.peak_usage())

#CACHE
vbt.print_cache_stats()  
vbt.print_cache_stats(vbt.PF) 

vbt.flush() #clear cache and collect garbage
vbt.clear_cache(pf) #of specific


#TIMING
#running once
with vbt.Timer() as timer:
    my_pipeline()

print(timer.elapsed())

#multiple times
print(vbt.timeit(my_pipeline))

#NUMBA
#numba doesnt return error when indexing out of bound, this raises the error
import os
os.environ["NUMBA_BOUNDSCHECK"] = "1"
```