import sys
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding for tree symbols on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from tree_gui import show_tree
except ImportError:
    show_tree = None




def ent(column):
    entropy = 0

    for value in column.unique():
        p = (column == value).sum() / len(column)
        entropy += p * np.log2(p)

    return -entropy

def weighted_ent(df, column, target_columname):
    weighted_ent = 0
    for value in column.unique():
            subset = df.loc[column == value, target_columname]
            weight = len(subset) / len(df)
            weighted_ent += weight * ent(subset)  
    return weighted_ent

def split_decesion(df , features_list, target ):
     main_ent = ent(target)
     weighted_ent_stored = {}
     for feature in features_list:
      weightedent_value = weighted_ent(df , feature , target.name)
      weighted_ent_stored[feature.name] = weightedent_value

     return main_ent , weighted_ent_stored

def build_tree(df, features , target ):
    main_ent , weighted_ents = split_decesion(df , features , target)
    if np.isclose(main_ent , 0) :
        return target.iloc[0]
    
    best_feature = min(weighted_ents, key=weighted_ents.get)
    tree = {best_feature: {}}

    for value in df[best_feature].unique():

        subset = df[df[best_feature] == value]
        remaining_features = [
        subset[col]
        for col in subset.columns
        if col != best_feature and col != target.name
        ]

        tree[best_feature][value] = build_tree(
                    subset,
                    features =remaining_features,
                    target = subset[target.name]
                )
    return tree

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print("→", tree)
        return

    feature = next(iter(tree))
    branches = tree[feature]

    print(indent + feature)

    items = list(branches.items())

    for i, (value, child) in enumerate(items):

        if i == len(items) - 1:
            branch = "`-- "
            next_indent = indent + "    "
        else:
            branch = "|-- "
            next_indent = indent + "|   "

        if isinstance(child, dict):
            print(indent + branch + str(value))
            print_tree(child, next_indent)
        else:
            print(indent + branch + str(value) + " -> " + str(child))

#Create Data List
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

#Load data into a data frame
df = pd.DataFrame(data)

#Set Target
target = df["Play Tennis"]

#Define features
features = [
    df[col]
    for col in df.columns
    if col != "Play Tennis"
]
#Build The Tree With entropy
tree = build_tree(df , features, target)

#Print the tree
print_tree(tree)




