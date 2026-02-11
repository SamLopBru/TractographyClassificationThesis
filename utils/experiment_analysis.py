import pandas as pd
import json

df = pd.read_csv("experiments.csv")

def filter_results(df, metric, ascending=False, top_n=10):
    return df.sort_values(by=metric, ascending=ascending).head(top_n)

# Filter for best results
df_best = filter_results(df, "best_val_f1", ascending=False, top_n=10)
print(df_best)