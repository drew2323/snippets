import vectorbtpro as vbt
from lightweight_charts import Panel, chart, PlotDFAccessor, PlotSRAccessor

#region DATA
 #fetching from remote db
from lib.db import Connection
SYMBOL = "BAC"
SCHEMA = "ohlcv_1s" #time based 1s other options ohlcv_vol_200 (volume based ohlcv with resolution of 200), ohlcv_renko_20 (renko with 20 bricks size) ...
DB = "market_data"

con = Connection(db_name=DB, default_schema=SCHEMA, create_db=True)
basic_data = con.pull(symbols=[SYMBOL], schema=SCHEMA,start="2024-08-01", end="2024-08-08", tz_convert='America/New_York')
#endregion

#region DISCOVERY
#get parameters of method
vbt.phelp(vbt.indicator("talib:MOM").run)

vbt.IF.list_indicators("*sma")

#endregion

#region RESAMPLING
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

t1data = basic_data[['open', 'high', 'low', 'close', 'volume','vwap','buyvolume','trades','sellvolume']].resample("1T")
t1data = t1data.transform(lambda df: df.between_time('09:30', '16:00').dropna())

#realign closing
resampler_s = vbt.Resampler(t1data.index, s1data.index, source_freq="1T", target_freq="1s")
t1close_realigned = t1data.data["BAC"].close.vbt.realign_closing(resampler_s)



#endregion

#region ENTRIES/EXITS
#doc from_signal http://5.161.179.223:8000/vbt-doc/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals
- StopExitPrice (Which price to use when exiting a position upon a stop signal?)
- StopEntryPrice (Which price to use as an initial stop price?)
                  
price = close.vbt.wrapper.fill()
price[entries] = entry_price
price[exits] = exit_price


# window open/close 


#END OF DAY EXITS
# end_of_day_dates = index.to_series().resample("1d").last().values
# exit_signals.loc[end_of_day_dates] = True
end_of_day_dates = open_hours_index.to_series().resample("1d").last()
df['exit'][df['exit'].index.isin(end_of_day_dates)] = True
# This index should be probably open_hours_index
# But also check that end_of_day_dates doesn't have nans (NaT), and if it has, you need to filter them out (edited)

#endregion

#region STOPLOSS/TAKEPROFIT

#doc StopOrders http://5.161.179.223:8000/vbt-doc/documentation/portfolio/from-signals/index.html#stop-orders

#SL - ATR based
atr = data.run("atr").atr
pf = vbt.Portfolio.from_signals(
    data,
    entries=entries,
    sl_stop=atr / sub_data.close
)

#EXIT after time http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#from-signals
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

#CALLBACKS - 
"""
 - a signal function (signal_func_nb)
    - can dynamically generate signals (True, True, False,False)
    - runs at beginning of bar
 - an adjustment function (adjust_func_nb) 
    - runs only if signal function above was not provided, but entry,exit arrays
    - runs before default signal function ls_signal_func_nb http://5.161.179.223:8000/vbt-doc/api/portfolio/nb/from_signals/index.html#vectorbtpro.portfolio.nb.from_signals.ls_signal_func_nb
        - can change pending limit orders etc.
 - a post-signal function (post_signal_func_nb)
 - post-segment function (post_segment_func_nb)

all of them are accessing SignalContext (c) as named tuple http://5.161.179.223:8000/vbt-doc/api/portfolio/enums/index.html#vectorbtpro.portfolio.enums.SignalContext
SignalContaxt (contains various metrics)
   - last_limit_info - 1D with latest limit order per column
    - order_counts
    - last_return ...
    
"""

#MEMORY http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks
#save an information piece at one timestamp and re-use at a later timestamp


#MULTIPLE LIMIT ORDERS at TIME http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks

#IGNORE ENTRIES number of DAYS after losing trade - signal function http://5.161.179.223:8000/vbt-doc/cookbook/portfolio/index.html#callbacks

#adjust_func_nb http://5.161.179.223:8000/vbt-doc/documentation/portfolio/from-signals/#adjustment


#endregion


#region INDICATORS
#anchored VWAP
t1vwap_h = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="H")
t1vwap_d = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="D")
t1vwap_t = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="T")

t1vwap_h_real = t1vwap_h.vwap.vbt.realign_closing(resampler_s)
t1vwap_d_real = t1vwap_d.vwap.vbt.realign_closing(resampler_s)
t1vwap_t_real = t1vwap_t.vwap.vbt.realign_closing(resampler_s)

#BBANDS = vbt.indicator("pandas_ta:BBANDS")
mom_anch_d = AnchoredIndicator("talib:MOM", anchor='30min').run(t1data.data["BAC"].close, timeperiod=10)
mom = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close, timeperiod=10, skipna=True)
#macd = vbt.indicator("talib:MACD").run(t1data.data["BAC"].close) #, timeframe=["1T"]) #, 
t1data.ohlcv.data["BAC"].lw.plot(auto_scale=[mom_anch_d, mom])

#SMA - note for TALIB use skipna=True
mom_multi_beztf = vbt.indicator("talib:MOM").run(t1data.close, timeperiod=5, skipna=True)

#TALIB INDICATORS can do realing closing : timeframe=["1T"]
mom_multi = vbt.indicator("talib:MOM").run(t1data.close, timeperiod=5, timeframe=["1T","5T"], skipna=True) #returned 5T can be directly compared with 1T

#ANCHORED indciators vbt.indicator("pandas_ta:BBANDS") is called AnchoredIndicator("pandas_ta:BBANDS")
from ttools import AnchoredIndicator

#BBANDS = vbt.indicator("pandas_ta:BBANDS")
mom_anch_d = AnchoredIndicator("talib:MOM", anchor='30min').run(t1data.data["BAC"].close, timeperiod=10)
mom = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close, timeperiod=10, skipna=True)
#macd = vbt.indicator("talib:MACD").run(t1data.data["BAC"].close) #, timeframe=["1T"]) #, 
t1data.ohlcv.data["BAC"].lw.plot(auto_scale=[mom_anch_d, mom])

#REGISTER CUSTOM INDICATOR
vbt.IndicatorFactory.register_custom_indicator(
    SupportResistance,
    name="SUPPRES",
    location=None,
    if_exists='raise'
)

#endregion

#region GROUPING - SPLITTING

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


#endregion

#region CHARTING

#LW df/sr accessor
t1data.ohlcv.data["BAC"].lw.plot(left=[(mom_multi, "mom_multi")]) #OHLCV with indicators on top


t5data.ohlcv.data["BAC"].lw.plot(
    left=[(mom_multi.real, "mom"),(mom_multi_beztf, "mom_beztf"), (mom_5t_orig, "mom_5t_orig"), (mom_5t_orig_realigned, "mom_5t_orig_realigned")],
    right=[(t1data.data["BAC"].close, "t1 close"),(t5data.data["BAC"].close, "t5 close")],
    size="s") #.loc[:,(20,"1T","BAC")]

#PANEL
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

ch = chart([pane1], size="s")



#endregion

#region MULTIACCOUNT
#simultaneous LONG and short (hedging)
#VBT: One position requires one column of data, so hedging is possible by using two columns representing the same asset but different directions,
# then stack both portfolio together (http://5.161.179.223:8000/vbt-doc/features/productivity/#column-stacking)
pf_join = vbt.PF.column_stack((pf1, pf2), group_by=True)


#endregion

#region CUSTOM SIMULATION



#endregion


#region ANALYSIS


#endregion