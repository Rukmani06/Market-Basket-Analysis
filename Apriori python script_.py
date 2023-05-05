import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
data = pd.read_excel("groceries_Full_data.xlsx", header=None)
transactions = []
for i in range(data.shape[0]):
    transaction = data.loc[i].dropna().tolist()
    transactions.append(transaction)
one_hot_encoded_data = pd.get_dummies(pd.DataFrame(transactions).stack(dropna=True)).groupby(level=0).sum()
min_support = 0.01
frequent_itemsets = apriori(one_hot_encoded_data, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
rules_three_items = rules[rules['antecedents'].apply(lambda x: len(x)) == 2]
rules_three_items = rules_three_items.sort_values(by="lift", ascending=False)
print("Frequent itemsets:")
print(frequent_itemsets)
print("\nAssociation rules (3 items purchased together):")
print(rules_three_items)