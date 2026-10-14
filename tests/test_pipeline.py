import pandas as pd
from cyclone.pipeline import make_examples

def test_six_hour_target_and_causal_change():
    frame = pd.DataFrame({
        "sid":["A"]*4,
        "time":pd.to_datetime(["2020-01-01 00:00","2020-01-01 06:00",
                               "2020-01-01 12:00","2020-01-02 00:00"],utc=True),
        "basin":["NA"]*4,"lat":[10]*4,"lon":[-60]*4,
        "wind_kt":[40,50,60,70],"pressure_mb":[1000]*4,"storm_speed":[8]*4
    })
    out = make_examples(frame)
    assert len(out)==2
    assert out.iloc[0]["target_wind_6h"]==50
    assert pd.isna(out.iloc[0]["wind_change_6h"])
    assert out.iloc[1]["wind_change_6h"]==10
    assert out.iloc[1]["target_wind_6h"]==60

def test_storms_not_mixed_in_lags():
    frame = pd.DataFrame({
        "sid":["A","B","A","B"],
        "time":pd.to_datetime(["2020-01-01 00:00","2020-01-01 00:00",
                               "2020-01-01 06:00","2020-01-01 06:00"],utc=True),
        "basin":["NA"]*4,"lat":[10]*4,"lon":[-60]*4,
        "wind_kt":[40,90,50,95],"pressure_mb":[1000]*4,"storm_speed":[8]*4
    })
    out = make_examples(frame)
    assert set(out["target_wind_6h"])=={50,95}
