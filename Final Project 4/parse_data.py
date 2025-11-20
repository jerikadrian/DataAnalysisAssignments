import pandas as pd

def parse_data():
    df0 = pd.read_csv("energydata_complete.csv")
    df0['date'] = pd.to_datetime(df0['date'])
    df0['day'] = df0['date'].dt.date

    indoor_temp_cols = ["T1","T2","T3","T4","T5","T7","T8","T9"]
    indoor_humid_cols = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_7","RH_8","RH_9"]

    df0["Indoor_T"]  = df0[indoor_temp_cols].mean(axis=1)
    df0["Indoor_RH"] = df0[indoor_humid_cols].mean(axis=1)

    df0["Energy"] = df0["Appliances"] + df0["lights"]

    daily = df0.groupby("day").agg({
    "Energy":     "sum",
    "Indoor_T":   "mean",
    "Indoor_RH":  "mean",
    "T_out":      "mean",
    "RH_out":     "mean",
    "Windspeed":  "mean",
    "Visibility": "mean"
    }).reset_index()

    daily = daily.rename(columns={
    "day":        "date",
    "Energy":     "total_energy",
    "Indoor_T":   "t_in",
    "Indoor_RH":  "h_in",
    "T_out":      "t_out",
    "RH_out":     "h_out",
    "Windspeed":  "windspeed",
    "Visibility": "visibility"
    })

    df = daily.drop(columns=["date"])

    return df