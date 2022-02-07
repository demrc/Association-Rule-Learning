!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#Part 1
df_ = pd.read_excel("recommender_systems/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011") #Step 1
df = df_.copy()
df.head()
df = df[~df["StockCode"].str.contains("POST",na=False)] #Step 2
df.dropna(inplace=True) #Step 3
df.isnull().sum()
df = df[~df["Invoice"].str.contains("C",na=False)] #Step 4
df = df[df["Price"]>0] #Step 5
df = df[df["Quantity"]>0]
df.describe([0.99]).T #Step 6
#Price ve Quantity sütunlarında %99 olan değerlerine
#bakıldığına ciddi miktarda büyük fark gözükmektedir. Yani aykırı değer vardır.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#lets try to replace values
replace_with_thresholds(df,"Quantity")
replace_with_thresholds(df,"Price")
df.describe([0.99]).T
#Part 2
df_ger = df[df["Country"]=="Germany"] #Step 1
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_ger)
ger_inv_pro_df.head()
def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules ##Step 2
rules = create_rules(df)
#Part 3
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_ger, 22423) #Step 1
def arl_recommender(rules_df, product_id, rec_count=1): #Step 2
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules,21987,2)
arl_recommender(rules,23235,2)
arl_recommender(rules,22747,2)
#---------Step 3----------
check_id(df_ger,22745) #product
check_id(df_ger,22746) #product
check_id(df_ger,23243) #product
check_id(df_ger,23244) #product
check_id(df_ger,21989) #product
check_id(df_ger,21988) #product
