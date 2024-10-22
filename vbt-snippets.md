- [DEBUGGING](#debugging)
- [FETCHING DATA](#fetching-data)
  - [REINDEX to main session](#reindex-to-main-session)
  - [indexing](#indexing)
  - [Data manipulation](#data-manipulation)
- [DISCOVERY](#discovery)
- [DATA/WRAPPER](#datawrapper)
  - [create WRAPPER manually](#create-wrapper-manually)
- [RESAMPLING](#resampling)
  - [config](#config)
- [REALIGN](#realign)
  - [REALIGN\_CLOSING accessors](#realign_closing-accessors)
- [SIGNALS](#signals)
  - [Comparing](#comparing)
  - [GENERATE SIGNALS IRERATIVELY (numba)](#generate-signals-ireratively-numba)
  - [or as indicators](#or-as-indicators)
  - [ENTRIES/EXITS time based](#entriesexits-time-based)
  - [STOPS](#stops)
  - [OHLCSTX Module](#ohlcstx-module)
  - [Entry Window and Forced Exit Window](#entry-window-and-forced-exit-window)
  - [END OF DAY EXITS](#end-of-day-exits)
  - [REGULAR EXITS](#regular-exits)
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
- [Portfolio](#portfolio)
  - [delta format](#delta-format)
  - [CALLBACKS](#callbacks)
  - [Staticization](#staticization)
- [INDICATORS DEV](#indicators-dev)
  - [Custom ind](#custom-ind)
    - [register custom ind](#register-custom-ind)
    - [VWAP anchored example](#vwap-anchored-example)
    - [Use ttols indicators](#use-ttols-indicators)
- [FAV INDICATORS](#fav-indicators)
- [GROUPING](#grouping)
- [SPLITTING](#splitting)
- [CHARTING](#charting)
- [MULTIACCOUNT](#multiaccount)
- [CUSTOM SIMULATION](#custom-simulation)
- [ANALYSIS](#analysis)
  - [ROBUSTNESS](#robustness)
- [UTILS](#utils)
- [Market calendar](#market-calendar)


```python
import vectorbtpro as vbt
from lightweight_charts import Panel, chart, PlotDFAccessor, PlotSRAccessor
t15data = None

if not hasattr(pd.Series, 'lw'):
    pd.api.extensions.register_series_accessor("lw")(PlotSRAccessor)

if not hasattr(pd.DataFrame, 'lw'):
    pd.api.extensions.register_dataframe_accessor("lw")(PlotDFAccessor)
```

# DEBUGGING
prints which arguments are being passed to apply_func.

```python
def apply_func(*args, **kwargs):
    for i, arg in enumerate(args):
        print("arg {}: {}".format(i, type(arg)))
    for k, v in kwargs.items():
        print("kwarg {}: {}".format(k, type(v)))
    raise NotImplementedError

RollCov = vbt.IF(
    class_name='RollCov',
    input_names=['ts1', 'ts2'],
    param_names=['w'],
    output_names=['rollcov'],
).with_apply_func(apply_func, select_params=False)

ollCov.run(ts1, ts2, [2, 3], some_arg="some_value")
```


# FETCHING DATA  
```python
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


#Fetching from local cache
dir = DATA_DIR + "/notebooks/"
import os
files = [f for f in os.listdir(dir) if f.endswith(".parquet")]
print('\n'.join(map(str, files)))
file_name = "ohlcv_df-BAC-2024-10-03T09:30:00-2024-10-16T16:00:00-['4', '7', 'B', 'C', 'F', 'O', 'P', 'U', 'V', 'W', 'Z']-100.parquet"
ohlcv_df = pd.read_parquet(dir+file_name,engine='pyarrow')
basic_data = vbt.Data.from_data(vbt.symbol_dict({"BAC": ohlcv_df}), tz_convert=zoneNY)
basic_data.wrapper.index.normalize().nunique() #numdays

#Fetching Trades and Aggregating custom OHLCV
TBD
```

## REINDEX to main session
Get trading days main sessions from `pandas_market_calendars` and reindex fetched data to main session only. 
```python
import vectorbtpro as vbt

# Start and end dates to use across both the calendar and data fetch
start=data.index[0].to_pydatetime()
end=tata.index[-1].to_pydatetime()
timeframe="1m"

import pandas_market_calendars as mcal
# Get the NYSE calendar
nyse = mcal.get_calendar("NYSE")
# Get the market hours data
market_hours = nyse.schedule(start_date=start, end_date=end, tz=nyse.tz)
#market_hours = market_hours.tz_localize(nyse.tz)
# Create a DatetimeIndex at our desired frequency for that schedule. Because the calendar hands back the end of
# the window, you need to subtract that size timeframe to get back to the start
market_klines = mcal.date_range(market_hours, frequency=timeframe) - pd.Timedelta(timeframe)

testData = vbt.YFData.fetch(['MSFT'], start=start, end=end, timeframe=timeframe, tz_convert="US/Eastern")
# Finally, take our DatetimeIndex and use that to pull just the data we're interested in (and ensuring we have rows
# for any empty klines in there, which helps for some time based algorithms that need to have time not exist outside
# of market hours)
testData = testData.transform(lambda x: x.reindex(market_klines))
```
## indexing
```python
entries.vbt.xloc[slice("2024-08-01","2024-08-03")].obj.info()

data.xloc[slice("9:30","10:00")] #targeting only morning rush
```

## Data manipulation

```python
#add/rename/delete symbols
s12_data = s12_data.rename_symbols("BAC", "BAC-LONG")
s12_data = s12_data.add_symbol("BAC-SHORT", s12_data.data["BAC-LONG"])
s12_adata.symbols
s12_data = s12_data.remove_symbols(["BAC-SHORT"])
```
# DISCOVERY

```python
#get parameters of method
vbt.IF.list_locations() #lists categories
vbt.IF.list_indicators(pattern="vbt") #all in category vbt
vbt.IF.list_indicators("*sma")
vbt.phelp(vbt.indicator("talib:MOM").run)
```


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
## Comparing
```python
dvla = np.round(div_vwap_lin_angle.real,4) #ROUNDING to 4 decimals

long_entries = tts.isrisingc(dvla,3).vbt & div_vwap_cum.div_below(0) #strictly rising for 3 bars
short_entries = tts.isfalling(dvla,3).vbt & div_vwap_cum.div_above(0) #strictly falling for 3 bars
long_entries = tts.isrising(dvla,3)#rising for 3 bars including equal values
short_entries = tts.isfalling(dvla,3)#falling for 3 bars including equal values

cond1 = data.get("Low") < bb.lowerband
#comparing with previous value
cond2 = bandwidth > bandwidth.shift(1)
#comparing with value week ago  
cond2 = bandwidth > bandwidth.vbt.ago("7d")
mask = cond1 & cond2
mask.sum()

#creating
bandwidth = (bb.upperband - bb.lowerband) / bb.middleband
mask = bandwidth.vbt > vbt.Param([0.15, 0.3], name="threshold")  #broadcasts and create combinations (for scalar params only)

#same but for arrays
mask = bandwidth.vbt.combine(
    [0.15, 0.3],  #values elements (scalars or array)
    combine_func=np.greater, 
    keys=pd.Index([0.15, 0.3], name="threshold")  #keys for the multiindex
)
mask.sum()
```

## GENERATE SIGNALS IRERATIVELY (numba)

Used for 1D. For multiple symbol create own indicator instead.
```python
@njit  
def generate_mask_1d_nb(  #required arrays as inputs
    high, low,  
    uband, mband, lband,  
    cond2_th, cond4_th  
):
    out = np.full(high.shape, False)  
    
    for i in range(high.shape[0]):  
        
        bandwidth = (uband[i] - lband[i]) / mband[i]
        cond1 = low[i] < lband[i]
        cond2 = bandwidth > cond2_th
        cond3 = high[i] > uband[i]
        cond4 = bandwidth < cond4_th
        signal = (cond1 and cond2) or (cond3 and cond4)  
        out[i] = signal  
        
    return out

mask = generate_mask_1d_nb(
    data.get("High")["BTCUSDT"].values,  
    data.get("Low")["BTCUSDT"].values,
    bb.upperband["BTCUSDT"].values,
    bb.middleband["BTCUSDT"].values,
    bb.lowerband["BTCUSDT"].values,
    0.30,
    0.15
)
symbol_wrapper = data.get_symbol_wrapper()
mask = symbol_wrapper["BTCUSDT"].wrap(mask)  
mask.sum()
```

or create extra numba function to iterate over columns

```python
@njit
def generate_mask_nb(  
    high, low,
    uband, mband, lband,
    cond2_th, cond4_th
):
    out = np.empty(high.shape, dtype=np.bool_)  
    
    for col in range(high.shape[1]):  
        out[:, col] = generate_mask_1d_nb(  
            high[:, col], low[:, col],
            uband[:, col], mband[:, col], lband[:, col],
            cond2_th, cond4_th
        )
        
    return out

mask = generate_mask_nb(
    vbt.to_2d_array(data.get("High")),  
    vbt.to_2d_array(data.get("Low")),
    vbt.to_2d_array(bb.upperband),
    vbt.to_2d_array(bb.middleband),
    vbt.to_2d_array(bb.lowerband),
    0.30,
    0.15
)
mask = symbol_wrapper.wrap(mask)
mask.sum()
```


## or as indicators

Works on columns.

```python
MaskGenerator = vbt.IF(  
    input_names=["high", "low", "uband", "mband", "lband"],
    param_names=["cond2_th", "cond4_th"],
    output_names=["mask"]
).with_apply_func(generate_mask_1d_nb, takes_1d=True)  
mask_generator = MaskGenerator.run(  
    data.get("High"),
    data.get("Low"),
    bb.upperband,
    bb.middleband,
    bb.lowerband,
    [0.3, 0.4],
    [0.1, 0.2],
    param_product=True  
)
mask_generator.mask.sum()
```


## ENTRIES/EXITS time based
```python
#create entries/exits based on open of first symbol
entries = pd.DataFrame.vbt.signals.empty_like(data.open.iloc[:,0]) 
exits = pd.DataFrame.vbt.signals.empty_like(entries)

#OR create entries/exits based on symbol level if needed (for each columns)
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

## OHLCSTX Module
 - exit signal generator based on price and stop values
[doc](ttp://5.161.179.223:8000/vbt-doc/api/signals/generators/ohlcstx/index.html)


## Entry Window and Forced Exit Window
Applying `entry window `range (denoted by minutes from the session start) to `entries` and applying `forced exit window` to `exits`.

`create_mask_from_window` with param `use_cal=True` (default) uses market calendar data for each day to denote session start and end. When disabled it uses just fixed 9:30-16:00 for each day.

```python
from ttools import create_mask_from_window

entry_window_opens = 3 #in minutes from start of the market
entry_window_closes = 388
forced_exit_start = 387
forced_exit_end = 390

#create mask based on main session that day
entry_window_opened = create_mask_from_window(entries, entry_window_opens, entry_window_closes, use_cal=True)
#limit entries to the window
entries = entries & entry_window_opened

#create forced exits mask
forced_exits_window = create_mask_from_window(exits, forced_exit_start, forced_exit_end, use_cal=True)

#add forced_exits to exits
exits = exits | forced_exits_window
```

## END OF DAY EXITS
Another way of eod exits according to number of bars at the end of the session. Assuming the last rows each day represents end of the market.

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

#which is ALTERNATIVE to
exits = create_mask_from_window(t1data.close, 387, 390, use_cal=False)
t1data.ohlcv.data["BAC"].lw.plot(right=[(t1data.close,"close",exits)], size="s")
```
## REGULAR EXITS
Time based.
```python
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
For common taks ([docs](http://5.161.179.223:8000/vbt-doc/api/generic/accessors/index.html#vectorbtpro.generic.accessors.GenericAccessor)) 

*  `rolling_apply` - runs custom function over a rolling window of a fixed size (number of bars or frequency)

* `expanding_apply` - runs custome function over expanding the window from the start of the data to the current poin

```python
from numba import njit
mean_nb = njit(lambda a: np.nanmean(a))
hourly_anchored_expanding_mean = t1data.close.vbt.rolling_apply("1H", mean_nb) #ROLLING to FREQENCY or with fixed windows rolling_apply(10,mean_nb)
t1data.ohlcv.data["BAC"].lw.plot(right=[(t1data.close,"close"),(hourly_anchored_expanding_mean, "hourly_anchored_expanding_mean")], size="s")
#NOTE for anchored "1D" frequency - it measures timedelta that means requires 1 day between reseting (16:00 end of market, 9:30 start - not a full day, so it is enOugh to set 7H)

#HEATMAP OVERLAY
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

# Portfolio

group_by=True to put all columns to the same group and cash_sharing=True to share capital among them

```python
pf = vbt.Portfolio.from_signals(
    close=s12_data.close,
    entries=long_entries_cln,
    exits=long_exits,
    short_entries=short_entries_cln,
    short_exits=short_exits,
    sl_stop=0.3,
    tp_stop = 0.4,
    delta_format = vbt.pf_enums.DeltaFormat.Percent100, #(Absolute, Percent, Percent100, Target)
    fees=0.0167/100,
    freq="12s") #sl_stop=sl_stop, tp_stop = sl_stop,, tsl_stop
```

## delta format

```python
vbt.pf_enums.DeltaFormat:
    Absolute: int = 0
    Percent: int = 1
    Percent100: int = 2
    Target: int = 3
```

## CALLBACKS

Callbacks functions can be used to place/alter entries/exits and various other things dynamically based on simulation status.
All of them contain [SignalContext](http://5.161.179.223:8000/vbt-doc/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext) and also can include custom Memory.

Importan SignalContact attributes:
* `c.i` - current index
* `c.index` - time index numpy
* `c.last_pos_info[c.col] ` - named tuple of last position info 
`{'names': ['id', 'col', 'size', 'entry_order_id', 'entry_idx', 'entry_price', 'entry_fees', 'exit_order_id', 'exit_idx', 'exit_price', 'exit_fees', 'pnl', 'return', 'direction', 'status', 'parent_id']`

Callback functions:
- signal_func_nb - place/alter entries/exits
- adjust_sl_func_nb - adjust SL at each time stamp

For exit dependent entries, the entries can be preprocessed in `signal_func_nb` see [callbacks](http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks) in cookbok or [signal function](http://5.161.179.223:8000/vbt-doc/documentation/portfolio/from-signals/index.html#signal-function)in doc

```python
#@njit
def signal_func_nb(c, entries, exits, short_entries, short_exits, cooldown):
    entry = vbt.pf_nb.select_nb(c, entries) #get current value
    exit = vbt.pf_nb.select_nb(c, exits)
    short_entry = vbt.pf_nb.select_nb(c, short_entries)
    short_exit = vbt.pf_nb.select_nb(c, short_exits)
    if not vbt.pf_nb.in_position_nb(c): # short for c.last_position == 0
        if vbt.pf_nb.has_orders_nb(c):  
            if c.last_pos_info[c.col]["pnl"] < 0:  #positive pnl on last reade
                last_exit_idx = c.last_pos_info[c.col]["exit_idx"]  #exit_idx
                if c.index[c.i] - c.index[last_exit_idx] < cooldown:
                    return False, exit, False, short_exit #disable all entries
    return entry, exit, short_entry, short_exit

"""
c.last_pos_info[c.col] 
   - is namedtuple {'names': ['id', 'col', 'size', 'entry_order_id', 'entry_idx', 'entry_price', 'entry_fees', 'exit_order_id', 'exit_idx', 'exit_price', 'exit_fees', 'pnl', 'return', 'direction', 'status', 'parent_id']
"""                                    
                                      
pf = vbt.Portfolio.from_signals(
    close=s12_data.close,
    entries=long_entries_cln,
    exits=long_exits,
    short_entries=short_entries_cln,
    short_exits=short_exits,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"), 
        vbt.Rep("exits"),
        vbt.Rep("short_entries"),
        vbt.Rep("short_exits"),
        vbt.dt.to_ns(vbt.timedelta("12s"))*5  # any timedelta, 12s meaning bars - TODO bar count
    ),
    sl_stop=0.3,
    tp_stop = 0.4,
    delta_format = vbt.pf_enums.DeltaFormat.Percent100, #(Absolute, Percent, Percent100, Target)
    fees=0.0167/100,
    freq="12s",
    jitted=False,
    statiticized=True) #sl_stop=sl_stop, tp_stop = sl_stop,, tsl_stop

```

Tips:  
- To avoid waiting for the compilation, remove the `@njit` decorator from `signal_func_nb` and pass `jitted=False` to from_signals in order to disable Numba

## Staticization
Callbacks make function uncacheable, 
to overcome that
- define the callback in external file `signal_func_nb.py`

```python
@njit
def signal_func_nb(c, fast_sma, slow_sma):  
    long = vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma)
    short = vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma)
    return long, False, short, False
```

and then use use `staticized=True`

```python
data = vbt.YFData.pull("BTC-USD")
pf = vbt.PF.from_signals(
    data,
    signal_func_nb="signal_func_nb.py",  
    signal_args=(vbt.Rep("fast_sma"), vbt.Rep("slow_sma")),
    broadcast_named_args=dict(
        fast_sma=data.run("sma", 20, hide_params=True, unpack=True), 
        slow_sma=data.run("sma", 50, hide_params=True, unpack=True)
    ),
    staticized=True  
)
```
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
## Custom ind

```python
#simple
from numba import jit
@jit
def apply_func(high, low, close):
    return (high + low + close + close) / 4

HLCC4 = vbt.IF(
    class_name='hlcc4',
    input_names=['high', 'low', 'close'],
    output_names=['out']
).with_apply_func(
    apply_func,
    timeperiod=10, #single default
    high=vbt.Ref('close')) #default from another input)

ind = HLCC4.run(s12_data.high, s12_data.low, s12_data.close)

#1D apply function
import talib

def apply_func_1d(close, timeperiod):
    return talib.SMA(close.astype(np.double), timeperiod)

SMA = vbt.IF(
    input_names=['ts'],
    param_names=['timeperiod'],
    output_names=['sma']
).with_apply_func(apply_func_1d, takes_1d=True)

sma = SMA.run(ts, [3, 4])
sma.sma

#with grouping and keep_pd (inputs are pd.series)
def apply_func(ts, group_by):
    return ts.vbt.demean(group_by=group_by)

Demeaner = vbt.IF(
    input_names=['ts'],
    param_names=['group_by'],
    output_names=['out']
).with_apply_func(apply_func, keep_pd=True) #if takes_1D it sends pd.series, otherwise df with symbol as columns

ts_wide = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [5, 4, 3, 2, 1],
    'c': [3, 2, 1, 2, 3],
    'd': [1, 2, 3, 2, 1]
}, index=generate_index(5))
demeaner = Demeaner.run(ts_wide, group_by=[(0, 0, 1, 1), True])
demeaner.out
```
### register custom ind
[indicator registration](http://5.161.179.223:8000/vbt-doc/cookbook/indicators/#registration)

```python
vbt.IF.register_custom_indicator(sma_indicator) #name=classname
vbt.IF.register_custom_indicator(sma_indicator, "rolling:SMA")

vbt.IF.deregister_custom_indicator("rolling:SMA")
```
### VWAP anchored example

```python
import numpy as np
from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.utils.template import RepFunc

def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute reset frequency by group lens. It is array of number of elements of each group."""
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()

@jit(nopython=True)
def vwap_cum(high, low, close, volume, group_lens):
    #anchor based grouping - prepare group indexes
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    #prepare output
    out = np.full(volume.shape, np.nan, dtype=np.float_)

    hlcc4 = (high + low + close + close) / 4

    #iterate over groups
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        to_i = group_end_idxs[group]
        nom_cumsum = 0
        denum_cumsum = 0
        #for each group do this (it is just np.cumsum(hlcc4 * volume) / np.sum(volume) iteratively)
        for i in range(from_i, to_i):
            nom_cumsum += volume[i] * hlcc4[i]
            denum_cumsum += volume[i]
            if denum_cumsum == 0:
                out[i] = np.nan
            else:
                out[i] = nom_cumsum / denum_cumsum
    return out

vwap_ind = vbt.IF(
    class_name='CUVWAP',
    input_names=['high', 'low', 'close', 'volume'],
    param_names=['anchor'],
    output_names=['vwap']
).with_apply_func(vwap_cum,
                takes_1d=True,
                param_settings=dict(
                    anchor=dict(template=RepFunc(substitute_anchor)),
                ),
                anchor="D",
                )

%timeit vwap_cum = vwap_ind.run(s12_data.high, s12_data.low, s12_data.close, s12_data.volume, anchor="min")
vbt.IF.register_custom_indicator(vwap_ind) 
```
### Use ttols indicators

```python
from ttools.vbtindicators import register_custom_inds
register_custom_inds(if_exists="skip") #register all, skip or override when exists
#register_custom_inds("CVWAP", "skip") #register one, skip if exists
#register_custom_inds() #deregister all
vbt.IF.list_indicators("ttools")

vwap_cum = vbt.indicator("ttools:CUVWAP").run(s12_data.high, s12_data.low, s12_data.close, s12_data.volume, anchor="D")
vwap_cum.vwap

div_vwap_cum = vbt.indicator("ttools:DIVERGENCE").run(s12_data.close, vwap_cum_d.vwap, divtype=vbt.Default(valeu="reln"), hide_default=True) #hide default levels

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

# GROUPING

Group wrapper index based on freq:

```python
#returns array of number of elements in each consec group
group_lens = s12_data.wrapper.get_index_grouper("D").get_group_lens()
#
group_end_idxs = np.cumsum(group_lens) #end indices of each group
group_start_idxs = group_end_idxs - group_lens #start indices of each group

out = np.full(volume.shape, np.nan, dtype=np.float_)

#iterate over groups
for group in range(len(group_lens)):
    from_i = group_start_idxs[group]
    to_i = group_end_idxs[group]

    #iterate over elements of the group
    for i in range(from_i, to_i):
        out[i] = np.nan
return out

```


# SPLITTING

```python
#SPLITTER - splitting wrapper based on index
#http://5.161.179.223:8000/vbt-doc/tutorials/cross-validation/splitter/index.html#anchored
#based on GROUPER
daily_splitter = vbt.Splitter.from_grouper(t1data.index, "D", split=None) #DOES contain last DAY

daily_splitter = vbt.Splitter.from_ranges( #doesnt contain last DY
    t1data.index,
    every="D",  
    split=None
)
daily_splitter.stats()
daily_splitter.plot()
daily_splitter.coverage()
daily_splitter.get_bounds(index_bounds=True) #shows the exact times
daily_splitter.get_bounds_arr()
daily_splitter.get_range_coverage(relative=True)

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

#RELOAD module in ipynb
%load_ext autoreload
%autoreload 2

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

#in notebook
%timeit function(x)
%% time
function(x)

#NUMBA
#numba doesnt return error when indexing out of bound, this raises the error
import os
os.environ["NUMBA_BOUNDSCHECK"] = "1"
```

# Market calendar

```python
from pandas.tseries.offsets import CustomBusinessDay
from pandas_market_calendars import get_calendar

# Get the NYSE trading calendar
nyse = get_calendar('NYSE')
# Create a CustomBusinessDay object using the NYSE trading calendar
custom_bd = CustomBusinessDay(holidays=nyse.holidays().holidays, weekmask=nyse.weekmask, calendar=nyse)
```
