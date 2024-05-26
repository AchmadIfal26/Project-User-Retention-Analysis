## User Retention Analysis

User retention is a key metric for understanding how well a business retains its customers over time. The analysis presented here examines the monthly retention of customers who made their first purchase in a given month (cohort). 

### Data Cleansing
The data was cleaned and processed with the following steps:
1. **Convert `order_date` to datetime format**:
    ```python
    df_clean['order_date'] = df_clean['order_date'].astype('datetime64[ns]')
    ```

2. **Create a `year_month` column**:
    ```python
    df_clean['year_month'] = df_clean['order_date'].dt.to_period('M')
    ```

3. **Remove rows without `customer_id` or `product_name`**:
    ```python
    df_clean = df_clean[~df_clean['customer_id'].isna()]
    df_clean = df_clean[~df_clean['product_name'].isna()]
    ```

4. **Standardize `product_name` to lowercase**:
    ```python
    df_clean['product_name'] = df_clean['product_name'].str.lower()
    ```

5. **Filter out test orders**:
    ```python
    df_clean = df_clean[(~df_clean['product_code'].str.lower().str.contains('test')) | (~df_clean['product_name'].str.contains('test '))]
    ```

6. **Assign `order_status` based on `order_id`**:
    ```python
    df_clean['order_status'] = np.where(df_clean['order_id'].str[:1]=='C', 'cancelled', 'delivered')
    ```

7. **Convert negative `quantity` to positive**:
    ```python
    df_clean['quantity'] = df_clean['quantity'].abs()
    ```

8. **Remove rows with negative `price`**:
    ```python
    df_clean = df_clean[df_clean['price']>0]
    ```

9. **Calculate `amount` as the product of `quantity` and `price`**:
    ```python
    df_clean['amount'] = df_clean['quantity'] * df_clean['price']
    ```

10. **Replace `product_name` with the most frequent name for each `product_code`**:
    ```python
    most_freq_product_name = df_clean.groupby(['product_code','product_name'], as_index=False).agg(order_cnt=('order_id','nunique')).sort_values(['product_code','order_cnt'], ascending=[True,False])
    most_freq_product_name['rank'] = most_freq_product_name.groupby('product_code')['order_cnt'].rank(method='first', ascending=False)
    most_freq_product_name = most_freq_product_name[most_freq_product_name['rank']==1].drop(columns=['order_cnt','rank'])
    df_clean = df_clean.merge(most_freq_product_name.rename(columns={'product_name':'most_freq_product_name'}), how='left', on='product_code')
    df_clean['product_name'] = df_clean['most_freq_product_name']
    df_clean = df_clean.drop(columns='most_freq_product_name')
    ```

11. **Convert `customer_id` to string**:
    ```python
    df_clean['customer_id'] = df_clean['customer_id'].astype(str)
    ```

12. **Remove outliers**:
    ```python
    from scipy import stats
    df_clean = df_clean[(np.abs(stats.zscore(df_clean[['quantity','amount']]))<3).all(axis=1)]
    ```

### Cohort Analysis

1. **Aggregate monthly transactions per user**:
    ```python
    df_user_monthly = df_clean.groupby(['customer_id', 'year_month'], as_index=False).agg(order_cnt=('order_id', 'nunique'))
    ```

2. **Determine cohort month (first transaction month for each user)**:
    ```python
    df_user_monthly['cohort'] = df_user_monthly.groupby('customer_id')['year_month'].transform('min')
    ```

3. **Calculate the period number**:
    ```python
    df_user_monthly['period_num'] = (df_user_monthly['year_month'] - df_user_monthly['cohort']).apply(attrgetter('n')) + 1
    ```

4. **Create a pivot table with cohort and period number**:
    ```python
    df_cohort_pivot = pd.pivot_table(df_user_monthly, index='cohort', columns='period_num', values='customer_id', aggfunc=pd.Series.nunique)
    ```

5. **Calculate retention rate**:
    ```python
    cohort_size = df_cohort_pivot.iloc[:, 0]
    df_retention_cohort = df_cohort_pivot.divide(cohort_size, axis=0)
    ```

### Visualizing Retention Rate
A heatmap can be used to visualize the retention rate for each cohort over time.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors

with sns.axes_style('white'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True, gridspec_kw={'width_ratios': [1, 11]})

    # User retention cohort
    sns.heatmap(df_retention_cohort, annot=True, fmt='.0%', cmap='RdYlGn', ax=ax[1])
    ax[1].set_title('User Retention Cohort')
    ax[1].set(xlabel='Month Number', ylabel='')

    # Cohort size
    df_cohort_size = pd.DataFrame(cohort_size)
    white_camp = mcolors.ListedColormap(['White'])
    sns.heatmap(df_cohort_size, annot=True, cbar=False, fmt='g', cmap=white_camp, ax=ax[0])
    ax[0].tick_params(bottom=False)
    ax[0].set(xlabel='Cohort Size', ylabel='First Order Month', xticklabels=[])

fig.tight_layout()
plt.show()
```

### Insights
From the retention heatmap, it can be observed that:
- **Initial retention** is typically higher, especially in the first month.
- **Retention tends to decline** over subsequent months, indicating a common challenge in maintaining customer engagement over time.

Analyzing this data helps businesses identify the effectiveness of their customer retention strategies and adapt them to improve long-term customer loyalty.
