import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel('groceries_Full_data.xlsx', header=None)
print(data.head())
print(data.info())
unique_items = pd.Series(data.stack().unique())
print(f"Number of unique items: {len(unique_items)}")
item_frequency = data.stack().value_counts()
print(item_frequency)
top_items = item_frequency[:20]
plt.figure(figsize=(12, 6))
sns.barplot(x=top_items.index, y=top_items.values)
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.title('Top 20 Items Frequency')
plt.xticks(rotation=90)
plt.show()
from mlxtend.preprocessing import TransactionEncoder
transactions = []
for i in range(len(data)):
    transactions.append([str(data.values[i, j]) for j in range(data.shape[1]) if str(data.values[i, j]) != 'nan'])
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
three_item_rules = rules[rules['antecedents'].apply(lambda x: len(x) == 2)]
print(three_item_rules.head())
three_item_rules_sorted = three_item_rules.sort_values(by=['confidence', 'lift'], ascending=False)
top_rule = three_item_rules_sorted.iloc[0]
antecedents = list(top_rule['antecedents'])
consequents = list(top_rule['consequents'])
top_three_items = antecedents + consequents
print("Three items most often purchased together:", top_three_items)
