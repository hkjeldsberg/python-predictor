import pandas as pd

df = pd.read_csv("data/owid-covid-data.csv", index_col="location")
df_norway = df.loc["Norway"]
coi = ["date", "new_cases", "new_deaths"]
df_new = df_norway[coi]
df_new.to_csv("data/covid_no.csv")
