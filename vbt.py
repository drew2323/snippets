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

# window open/close 



#END OF DAY EXITS
# end_of_day_dates = index.to_series().resample("1d").last().values
# exit_signals.loc[end_of_day_dates] = True
end_of_day_dates = open_hours_index.to_series().resample("1d").last()
df['exit'][df['exit'].index.isin(end_of_day_dates)] = True
# This index should be probably open_hours_index
# But also check that end_of_day_dates doesn't have nans (NaT), and if it has, you need to filter them out (edited)

#endregion


#region INDICATORS
#anchored VWAP
t1vwap_h = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="H")
t1vwap_d = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="D")
t1vwap_t = vbt.VWAP.run(t1data.high, t1data.low, t1data.close, t1data.volume, anchor="T")

t1vwap_h_real = t1vwap_h.vwap.vbt.realign_closing(resampler_s)
t1vwap_d_real = t1vwap_d.vwap.vbt.realign_closing(resampler_s)
t1vwap_t_real = t1vwap_t.vwap.vbt.realign_closing(resampler_s)

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