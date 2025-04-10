
import pandas as pd


df = pd.read_csv("universe.csv", parse_dates=['constituentDate'], dayfirst=True)


filtered_df = df[
    (df['bondType'] == 'Corporates') &
    (df['constituentDate'] == pd.Timestamp('2022-09-30'))
]


sorted_df = filtered_df.sort_values(by=['sector', 'ytm'], ascending=[True, False])


sorted_df.to_csv("filtered_sorted_universe_20220930.csv", index=False)

print("Filtered and sorted data saved to 'filtered_sorted_universe_20220930.csv'")



import pandas as pd


df = pd.read_csv("universe.csv", parse_dates=['constituentDate'], dayfirst=True)


filtered_df = df[
    (df['bondType'] == 'Corporates') &
    (df['constituentDate'] == pd.Timestamp('2022-10-31'))
]


sorted_df = filtered_df.sort_values(by=['sector', 'ytm'], ascending=[True, False])


sorted_df.to_csv("filtered_sorted_universe_20221031.csv", index=False)

print("Second parent index saved to 'filtered_sorted_universe_20221031.csv'")



import pandas as pd


df = pd.read_csv("universe.csv", parse_dates=['constituentDate'], dayfirst=True)


filtered_df = df[
    (df['bondType'] == 'Corporates') &
    (df['constituentDate'] == pd.Timestamp('2022-11-30'))
]


sorted_df = filtered_df.sort_values(by=['sector', 'ytm'], ascending=[True, False])


sorted_df.to_csv("filtered_sorted_universe_20221130.csv", index=False)

print("Third parent index saved to 'filtered_sorted_universe_20221130.csv'")



import pandas as pd


df = pd.read_csv("universe.csv", parse_dates=['constituentDate'], dayfirst=True)


filtered_df = df[
    (df['bondType'] == 'Corporates') &
    (df['constituentDate'] == pd.Timestamp('2022-12-31'))
]


sorted_df = filtered_df.sort_values(by=['sector', 'ytm'], ascending=[True, False])


sorted_df.to_csv("filtered_sorted_universe_20221231.csv", index=False)

print("Fourth parent index saved to 'filtered_sorted_universe_20221231.csv'")



import pandas as pd


df = pd.read_csv("filtered_sorted_universe_20220930.csv")


selected_bonds = []


grouped = df.groupby('sector')


for sector, group in grouped:

    group_sorted = group.sort_values(by='ytm', ascending=False)


    total_sector_weight = group_sorted['weight'].sum()


    cumulative_weight = 0
    for _, row in group_sorted.iterrows():
        selected_bonds.append(row)
        cumulative_weight += row['weight']
        if cumulative_weight >= 0.5 * total_sector_weight:
            break


yield_enhanced_index_df = pd.DataFrame(selected_bonds)


yield_enhanced_index_df.to_csv("yield_enhanced_index_20220930.csv", index=False)

print("First rebalance Yield-Enhanced Index saved to 'yield_enhanced_index_20220930.csv'")



import pandas as pd


parent_df = pd.read_csv("filtered_sorted_universe_20221031.csv")
buffer_df = pd.read_csv("yield_enhanced_index_20220930.csv")


buffer_bonds = set(buffer_df['bondIdentifier'])


selected_bonds = []


grouped = parent_df.groupby('sector')


for sector, group in grouped:

    group_sorted = group.sort_values(by='ytm', ascending=False)


    total_sector_weight = group_sorted['weight'].sum()


    sector_buffer_bonds = group_sorted[group_sorted['bondIdentifier'].isin(buffer_bonds)]
    cumulative_weight = sector_buffer_bonds['weight'].sum()

    selected_sector_bonds = sector_buffer_bonds.copy()


    for _, row in group_sorted.iterrows():
        if row['bondIdentifier'] in buffer_bonds:
            continue
        if cumulative_weight >= 0.5 * total_sector_weight:
            break
        selected_sector_bonds = pd.concat([selected_sector_bonds, pd.DataFrame([row])])
        cumulative_weight += row['weight']


    selected_bonds.append(selected_sector_bonds)


yield_enhanced_df = pd.concat(selected_bonds)


yield_enhanced_df.to_csv("yield_enhanced_index_20221031.csv", index=False)

print("Second Yield-Enhanced Index saved to 'yield_enhanced_index_20221031.csv'")



import pandas as pd


parent_df = pd.read_csv("filtered_sorted_universe_20221130.csv")


buffer_df = pd.read_csv("yield_enhanced_index_20221031.csv")


buffer_bonds = set(buffer_df['bondIdentifier'])


selected_bonds = []


grouped = parent_df.groupby('sector')


for sector, group in grouped:
    group_sorted = group.sort_values(by='ytm', ascending=False)

    total_sector_weight = group_sorted['weight'].sum()


    sector_buffer = group_sorted[group_sorted['bondIdentifier'].isin(buffer_bonds)]
    cumulative_weight = sector_buffer['weight'].sum()

    selected_sector = sector_buffer.copy()


    for _, row in group_sorted.iterrows():
        if row['bondIdentifier'] in buffer_bonds:
            continue
        if cumulative_weight >= 0.5 * total_sector_weight:
            break
        selected_sector = pd.concat([selected_sector, pd.DataFrame([row])])
        cumulative_weight += row['weight']


    selected_bonds.append(selected_sector)


yield_enhanced_df = pd.concat(selected_bonds)


yield_enhanced_df.to_csv("yield_enhanced_index_20221130.csv", index=False)

print("Third Yield-Enhanced Index saved to 'yield_enhanced_index_20221130.csv'")



import pandas as pd


parent_df = pd.read_csv("filtered_sorted_universe_20221231.csv")


buffer_df = pd.read_csv("yield_enhanced_index_20221130.csv")
buffer_bonds = set(buffer_df['bondIdentifier'])


selected_bonds = []


grouped = parent_df.groupby('sector')


for sector, group in grouped:
    group_sorted = group.sort_values(by='ytm', ascending=False)
    total_sector_weight = group_sorted['weight'].sum()


    sector_buffer = group_sorted[group_sorted['bondIdentifier'].isin(buffer_bonds)]
    cumulative_weight = sector_buffer['weight'].sum()

    selected_sector = sector_buffer.copy()


    for _, row in group_sorted.iterrows():
        if row['bondIdentifier'] in buffer_bonds:
            continue
        if cumulative_weight >= 0.5 * total_sector_weight:
            break
        selected_sector = pd.concat([selected_sector, pd.DataFrame([row])])
        cumulative_weight += row['weight']

    selected_bonds.append(selected_sector)


yield_enhanced_df = pd.concat(selected_bonds)


yield_enhanced_df.to_csv("yield_enhanced_index_20221231.csv", index=False)

print("Fourth Yield-Enhanced Index saved to 'yield_enhanced_index_20221231.csv'")



import pandas as pd
import cvxpy as cp
import numpy as np


parent_df = pd.read_csv("filtered_sorted_universe_20220930.csv")
enhanced_df = pd.read_csv("yield_enhanced_index_20220930.csv")


parent_df['parent_weight'] = parent_df['marketValue'] / parent_df['marketValue'].sum()


enhanced_df = enhanced_df.merge(
    parent_df[['bondIdentifier', 'sector', 'parent_weight', 'duration']],
    on='bondIdentifier',
    how='left',
    suffixes=('', '_parent')
)


n = len(enhanced_df)
w = cp.Variable(n)


w_ref = enhanced_df['parent_weight'].values
durations = enhanced_df['duration'].values
sectors = enhanced_df['sector']
min_weight = 0.0005
sector_list = sectors.unique()


parent_duration = np.sum(enhanced_df['duration'] * enhanced_df['parent_weight'])


duration_constraint = cp.abs(cp.sum(cp.multiply(w, durations)) - parent_duration) <= 0.25


sector_constraints = []
for sector in sector_list:
    mask = (sectors == sector).values.astype(float)
    sector_weight = np.sum(enhanced_df[enhanced_df['sector'] == sector]['parent_weight'])
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) <= sector_weight + 0.05)
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) >= sector_weight - 0.05)


min_weight_constraint = [w >= min_weight]


sum_constraint = [cp.sum(w) == 1]


objective = cp.Minimize(cp.sum_squares(w - w_ref))


constraints = [duration_constraint] + sector_constraints + min_weight_constraint + sum_constraint


problem = cp.Problem(objective, constraints)
problem.solve()


enhanced_df['optimized_weight'] = w.value


enhanced_df.to_csv("yield_enhanced_index_20220930_weighted.csv", index=False)

print(" Optimized weights saved to 'yield_enhanced_index_20220930_weighted.csv'")



import pandas as pd
import cvxpy as cp
import numpy as np


oct_df = pd.read_csv("yield_enhanced_index_20221031.csv")


sep_weights_df = pd.read_csv("yield_enhanced_index_20220930_weighted.csv")
ref_weights = sep_weights_df[['bondIdentifier', 'optimized_weight']].rename(columns={'optimized_weight': 'ref_weight'})


parent_df = pd.read_csv("filtered_sorted_universe_20221031.csv")
parent_df['parent_weight'] = parent_df['marketValue'] / parent_df['marketValue'].sum()


oct_df = oct_df.merge(ref_weights, on='bondIdentifier', how='left')


oct_df = oct_df.merge(
    parent_df[['bondIdentifier', 'sector', 'duration', 'parent_weight']],
    on='bondIdentifier',
    how='left',
    suffixes=('', '_parent')
)


oct_df['ref_weight'] = oct_df['ref_weight'].fillna(oct_df['parent_weight'])


n = len(oct_df)
w = cp.Variable(n)
w_ref = oct_df['ref_weight'].values
durations = oct_df['duration_parent'].values
sectors = oct_df['sector']
sector_list = sectors.unique()
min_weight = 0.0005


parent_duration = np.sum(oct_df['duration_parent'] * oct_df['parent_weight'])


duration_constraint = cp.abs(cp.sum(cp.multiply(w, durations)) - parent_duration) <= 0.25

sector_constraints = []
for sector in sector_list:
    mask = (sectors == sector).values.astype(float)
    sector_weight = np.sum(oct_df[oct_df['sector'] == sector]['parent_weight'])
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) <= sector_weight + 0.05)
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) >= sector_weight - 0.05)

min_weight_constraint = [w >= min_weight]
sum_constraint = [cp.sum(w) == 1]


objective = cp.Minimize(cp.sum_squares(w - w_ref))
constraints = [duration_constraint] + sector_constraints + min_weight_constraint + sum_constraint


problem = cp.Problem(objective, constraints)
problem.solve()


oct_df['optimized_weight'] = w.value
oct_df.to_csv("yield_enhanced_index_20221031_weighted.csv", index=False)

print(" October weights optimized and saved to 'yield_enhanced_index_20221031_weighted.csv'")



import pandas as pd
import cvxpy as cp
import numpy as np


nov_df = pd.read_csv("yield_enhanced_index_20221130.csv")


oct_weights_df = pd.read_csv("yield_enhanced_index_20221031_weighted.csv")
ref_weights = oct_weights_df[['bondIdentifier', 'optimized_weight']].rename(columns={'optimized_weight': 'ref_weight'})


parent_df = pd.read_csv("filtered_sorted_universe_20221130.csv")
parent_df['parent_weight'] = parent_df['marketValue'] / parent_df['marketValue'].sum()


nov_df = nov_df.merge(ref_weights, on='bondIdentifier', how='left')


nov_df = nov_df.merge(
    parent_df[['bondIdentifier', 'sector', 'duration', 'parent_weight']],
    on='bondIdentifier',
    how='left',
    suffixes=('', '_parent')
)


nov_df['ref_weight'] = nov_df['ref_weight'].fillna(nov_df['parent_weight'])


n = len(nov_df)
w = cp.Variable(n)
w_ref = nov_df['ref_weight'].values
durations = nov_df['duration_parent'].values
sectors = nov_df['sector']
sector_list = sectors.unique()
min_weight = 0.0005


parent_duration = np.sum(nov_df['duration_parent'] * nov_df['parent_weight'])


duration_constraint = cp.abs(cp.sum(cp.multiply(w, durations)) - parent_duration) <= 0.25

sector_constraints = []
for sector in sector_list:
    mask = (sectors == sector).values.astype(float)
    sector_weight = np.sum(nov_df[nov_df['sector'] == sector]['parent_weight'])
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) <= sector_weight + 0.05)
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) >= sector_weight - 0.05)

min_weight_constraint = [w >= min_weight]
sum_constraint = [cp.sum(w) == 1]


objective = cp.Minimize(cp.sum_squares(w - w_ref))
constraints = [duration_constraint] + sector_constraints + min_weight_constraint + sum_constraint


problem = cp.Problem(objective, constraints)
problem.solve()


nov_df['optimized_weight'] = w.value
nov_df.to_csv("yield_enhanced_index_20221130_weighted.csv", index=False)

print(" November weights optimized and saved to 'yield_enhanced_index_20221130_weighted.csv'")



import pandas as pd
import cvxpy as cp
import numpy as np


dec_df = pd.read_csv("yield_enhanced_index_20221231.csv")


nov_weights_df = pd.read_csv("yield_enhanced_index_20221130_weighted.csv")
ref_weights = nov_weights_df[['bondIdentifier', 'optimized_weight']].rename(columns={'optimized_weight': 'ref_weight'})


parent_df = pd.read_csv("filtered_sorted_universe_20221231.csv")
parent_df['parent_weight'] = parent_df['marketValue'] / parent_df['marketValue'].sum()


dec_df = dec_df.merge(ref_weights, on='bondIdentifier', how='left')


dec_df = dec_df.merge(
    parent_df[['bondIdentifier', 'sector', 'duration', 'parent_weight']],
    on='bondIdentifier',
    how='left',
    suffixes=('', '_parent')
)


dec_df['ref_weight'] = dec_df['ref_weight'].fillna(dec_df['parent_weight'])


n = len(dec_df)
w = cp.Variable(n)
w_ref = dec_df['ref_weight'].values
durations = dec_df['duration_parent'].values
sectors = dec_df['sector']
sector_list = sectors.unique()
min_weight = 0.0005


parent_duration = np.sum(dec_df['duration_parent'] * dec_df['parent_weight'])


duration_constraint = cp.abs(cp.sum(cp.multiply(w, durations)) - parent_duration) <= 0.25

sector_constraints = []
for sector in sector_list:
    mask = (sectors == sector).values.astype(float)
    sector_weight = np.sum(dec_df[dec_df['sector'] == sector]['parent_weight'])
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) <= sector_weight + 0.05)
    sector_constraints.append(cp.sum(cp.multiply(mask, w)) >= sector_weight - 0.05)

min_weight_constraint = [w >= min_weight]
sum_constraint = [cp.sum(w) == 1]


objective = cp.Minimize(cp.sum_squares(w - w_ref))
constraints = [duration_constraint] + sector_constraints + min_weight_constraint + sum_constraint


problem = cp.Problem(objective, constraints)
problem.solve()


dec_df['optimized_weight'] = w.value
dec_df.to_csv("yield_enhanced_index_20221231_weighted.csv", index=False)

print(" December weights optimized and saved to 'yield_enhanced_index_20221231_weighted.csv'")



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


files = {
    '2022-09-30': 'yield_enhanced_index_20220930_weighted.csv',
    '2022-10-31': 'yield_enhanced_index_20221031_weighted.csv',
    '2022-11-30': 'yield_enhanced_index_20221130_weighted.csv',
    '2022-12-31': 'yield_enhanced_index_20221231_weighted.csv'
}


sector_data = []

for date, file in files.items():
    df = pd.read_csv(file)
    sector_weights = df.groupby('sector')['optimized_weight'].sum()
    sector_weights.name = date
    sector_data.append(sector_weights)


sector_df = pd.concat(sector_data, axis=1).fillna(0).T
sector_df.index = list(files.keys())


plt.figure(figsize=(14, 8))
sector_df.plot(kind='bar', stacked=True)
plt.title('Sector Composition Over Time (Yield-Enhanced Index)')
plt.ylabel('Weight (%)')
plt.xlabel('Rebalance Date')
plt.xticks(rotation=0)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


files = [
    'yield_enhanced_index_20220930_weighted.csv',
    'yield_enhanced_index_20221031_weighted.csv',
    'yield_enhanced_index_20221130_weighted.csv',
    'yield_enhanced_index_20221231_weighted.csv'
]

dates = ['2022-09-30', '2022-10-31', '2022-11-30', '2022-12-31']
turnover_rates = [0]


for i in range(1, len(files)):
    prev = pd.read_csv(files[i - 1])
    curr = pd.read_csv(files[i])

    prev_ids = set(prev['bondIdentifier'])
    curr_ids = set(curr['bondIdentifier'])

    overlap = len(prev_ids & curr_ids)
    turnover = 1 - (overlap / len(curr_ids))

    turnover_rates.append(turnover)


plt.figure(figsize=(8, 5))
plt.plot(dates, turnover_rates, marker='o', linestyle='-', color='teal')
plt.title("Turnover Rate Over Time (Yield-Enhanced Index)")
plt.xlabel("Rebalance Date")
plt.ylabel("Turnover Rate")
plt.ylim(0, 0.03)
plt.grid(True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))


for i, rate in enumerate(turnover_rates):
    plt.text(dates[i], rate + 0.0005, f"{rate:.2%}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt


yield_files = {
    '2022-09-30': 'yield_enhanced_index_20220930_weighted.csv',
    '2022-10-31': 'yield_enhanced_index_20221031_weighted.csv',
    '2022-11-30': 'yield_enhanced_index_20221130_weighted.csv',
    '2022-12-31': 'yield_enhanced_index_20221231_weighted.csv'
}

parent_files = {
    '2022-09-30': 'filtered_sorted_universe_20220930.csv',
    '2022-10-31': 'filtered_sorted_universe_20221031.csv',
    '2022-11-30': 'filtered_sorted_universe_20221130.csv',
    '2022-12-31': 'filtered_sorted_universe_20221231.csv'
}

dates = list(yield_files.keys())
yield_durations = []
parent_durations = []


for date in dates:
    y_df = pd.read_csv(yield_files[date])
    yd = (y_df['duration'] * y_df['optimized_weight']).sum()
    yield_durations.append(yd)

    p_df = pd.read_csv(parent_files[date])
    p_df['parent_weight'] = p_df['marketValue'] / p_df['marketValue'].sum()
    pdur = (p_df['duration'] * p_df['parent_weight']).sum()
    parent_durations.append(pdur)


all_durations = yield_durations + parent_durations
y_min = min(all_durations) - 0.5
y_max = max(all_durations) + 0.5


plt.figure(figsize=(10, 6))
plt.plot(dates, parent_durations, label='Parent Index', marker='o')
plt.plot(dates, yield_durations, label='Yield-Enhanced Index', marker='o')
plt.title("Weighted Average Duration Over Time")
plt.xlabel("Rebalance Date")
plt.ylabel("Duration (Years)")
plt.ylim(y_min, y_max)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


df = pd.read_csv('yield_enhanced_index_20221231_weighted.csv')


if 'market_value_weight' not in df.columns:
    df['market_value_weight'] = df['marketValue'] / df['marketValue'].sum()


plt.figure(figsize=(7, 7))
plt.scatter(df['market_value_weight'], df['optimized_weight'], alpha=0.7, edgecolors='k')


plt.plot([0, 0.01], [0, 0.01], 'r--', label='No Change')


plt.xlim(0, 0.01)
plt.ylim(0, df['optimized_weight'].max() * 1.1)


plt.title("Market Value Weight vs Optimized Weight (Dec 2022)")
plt.xlabel("Market Value Weight (Pre-Optimization)")
plt.ylabel("Optimized Weight (Post-Optimization)")
plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


files = {
    '2022-09-30': 'yield_enhanced_index_20220930_weighted.csv',
    '2022-10-31': 'yield_enhanced_index_20221031_weighted.csv',
    '2022-11-30': 'yield_enhanced_index_20221130_weighted.csv',
    '2022-12-31': 'yield_enhanced_index_20221231_weighted.csv'
}

combined_top_movers = []

for date, path in files.items():
    df = pd.read_csv(path)

    if 'market_value_weight' not in df.columns:
        df['market_value_weight'] = df['marketValue'] / df['marketValue'].sum()

    df['weight_diff'] = (df['optimized_weight'] - df['market_value_weight']).abs()
    df['date'] = date

    top_n = 5
    top_movers = df.nlargest(top_n, 'weight_diff')[
        ['bondIdentifier', 'market_value_weight', 'optimized_weight', 'weight_diff', 'date']]
    combined_top_movers.append(top_movers)


all_top = pd.concat(combined_top_movers)


plt.figure(figsize=(12, 6))
for bond in all_top['bondIdentifier'].unique():
    bond_data = all_top[all_top['bondIdentifier'] == bond]
    plt.plot(bond_data['date'], bond_data['weight_diff'], marker='o', label=bond)

plt.title("Top Movers Across Rebalances (by Absolute Weight Change)")
plt.xlabel("Rebalance Date")
plt.ylabel("Absolute Weight Change")
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.legend(title="Bond Identifier", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


