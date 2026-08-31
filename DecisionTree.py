import pandas as pd
import numpy as np

def ent(column):
    entropy = 0

    for value in column.unique():
        p = (column == value).sum() / len(column)
        entropy += p * np.log2(p)

    return -entropy

def weighted_ent(df, column):
    weighted_ent = 0
    for value in column.unique():
            subset = df.loc[column == value, "Play Tennis"]
            weight = len(subset) / len(df)
            weighted_ent += weight * ent(subset)  
    return weighted_ent

def split_decesion(df , column , features_list):
     main_ent = ent(column)
     weighted_ent_stored = {}
     information_gain = {}
     for feature in features_list:
      weightedent_value = weighted_ent(df , feature)
      weighted_ent_stored[feature.name] = weightedent_value

     for feature in features_list:
      information_gain[feature.name] = main_ent - weighted_ent_stored[feature.name]
     return main_ent , information_gain

def print_ents(main_ent, info_gain, node_name):
    print(f"Node: {node_name}")
    print(f"Main Entropy: {main_ent:.3f}")

    print("Information Gain for Each Feature:")
    for feature, gain in info_gain.items():
        print(f"{feature:<12}: {gain:.3f}")

    print()

data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
                "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast",
                "Overcast", "Rain"],

    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
                    "Cool", "Mild", "Cool", "Mild", "Mild", "Mild",
                    "Hot", "Mild"],

    "H": ["High", "High", "High", "High", "Normal", "Normal",
          "Normal", "High", "Normal", "Normal", "Normal", "High",
          "Normal", "High"],

    "W": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong",
          "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
          "Weak", "Strong"],

    "Play Tennis": ["N", "N", "Y", "Y", "Y", "N", "Y",
                    "N", "Y", "Y", "Y", "Y", "Y", "N"]
}

df = pd.DataFrame(data)

main_ent , info_gain = split_decesion(
      df,
      df["Play Tennis"],
      [df["Outlook"], df["Temperature"], df["H"], df["W"]]
)
print_ents(main_ent, info_gain , "Root")

sunny_df = df[df["Outlook"] == "Sunny"]
main_ent_sunny, info_gain_sunny = split_decesion(
    sunny_df,
    sunny_df["Play Tennis"],
    [
        sunny_df["Temperature"],
        sunny_df["H"],
        sunny_df["W"]
    ]
)
print_ents(main_ent_sunny, info_gain_sunny , "Sunny")

rain_df = df[df["Outlook"] == "Rain"]
main_ent_rain, info_gain_rain = split_decesion(
    rain_df,
    rain_df["Play Tennis"],
    [
        rain_df["Temperature"],
        rain_df["H"],
        rain_df["W"]
    ]
)
print_ents(main_ent_rain, info_gain_rain , "Rain")

overcast_df= df[df["Outlook"] == "Overcast"]
main_ent_overcast, info_gain_overcast = split_decesion(
    overcast_df,
    overcast_df["Play Tennis"],
    [
        overcast_df["Temperature"],
        overcast_df["H"],
        overcast_df["W"]
    ]
)

print_ents(main_ent_overcast, info_gain_overcast , "Overcast")

