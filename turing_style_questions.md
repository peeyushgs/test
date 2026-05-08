# Turing-Style EDA Questions & Answers

Modeled on the format of the reference notebook
([Jayron003/Exploratory-Data-Analysis-of-Health-Dataset](https://github.com/Jayron003/Exploratory-Data-Analysis-of-Health-Dataset/blob/main/Health_Data_Analysis.ipynb)).

Each question is **specific**, **single-answer**, and accompanied by full pandas code, the expected output style, and the gotcha to watch for.

---

## Setup (run once)

```python
import io, requests, numpy as np, pandas as pd, scipy.stats as stats
import matplotlib.pyplot as plt, seaborn as sns
from zipfile import ZipFile

DATA_FILE = 'https://drive.usercontent.google.com/uc?id=1TMOK9yFDm6nJ2PAC-unBbrG-tR5THJ90&export=download'
zf = ZipFile(io.BytesIO(requests.get(DATA_FILE).content))
contents = {n: pd.read_csv(zf.open(n), sep='[,;]', engine='python') for n in zf.namelist()}

cardio = contents['cardio_base.csv'].copy()
alco   = contents['cardio_alco.csv'].copy()
covid  = contents['covid_data.csv'].copy()

# Convert age from DAYS to YEARS (rounded down — matches the reference notebook)
cardio['age'] = (cardio['age'] / 365.25).astype(int)

# Drop biologically impossible BP outliers (same thresholds the reference uses)
cardio = cardio[(cardio['ap_hi'].between(70, 200)) &
                (cardio['ap_lo'].between(40, 140))].copy()

covid['date'] = pd.to_datetime(covid['date'])
print(cardio.shape, alco.shape, covid.shape)
```

---

# Cardio_base questions

## Question 1
**How much heavier is the age group with the highest average weight than the age group with the lowest average weight?**

```python
avg_weight_by_age = cardio.groupby('age')['weight'].mean()
hi_age, hi_w = avg_weight_by_age.idxmax(), avg_weight_by_age.max()
lo_age, lo_w = avg_weight_by_age.idxmin(), avg_weight_by_age.min()
diff = hi_w - lo_w

print(f'Heaviest age group: {hi_age} yrs at {hi_w:.2f} kg')
print(f'Lightest age group: {lo_age} yrs at {lo_w:.2f} kg')
print(f'Difference: {diff:.2f} kg')
```
**Gotcha:** `groupby('age')` requires the age-in-years conversion above. Without it you'd be grouping by days.

---

## Question 2
**Are men more likely to be smokers than women, and by what factor?**

```python
men_pct   = cardio.loc[cardio['gender'] == 2, 'smoke'].mean() * 100
women_pct = cardio.loc[cardio['gender'] == 1, 'smoke'].mean() * 100
ratio = men_pct / women_pct

print(f'Men smokers:   {men_pct:.2f}%')
print(f'Women smokers: {women_pct:.2f}%')
print(f'Men are {ratio:.1f}x more likely to be smokers.')
```
**Gotcha:** gender encoding is `1 = female`, `2 = male` — verify by checking which group has higher mean height.

---

## Question 3
**How tall are the tallest 1% of people (in cm)?**

```python
top_1_threshold = np.percentile(cardio['height'], 99)
print(f'The tallest 1% are taller than {top_1_threshold:.0f} cm.')
```
**Gotcha:** `np.percentile(x, 99)` returns the cutoff above which the top 1% sits — not the height of the single tallest person.

---

## Question 4
**What is the absolute difference in mean systolic blood pressure between people above 50 and people 50 or below?**

```python
above = cardio.loc[cardio['age'] > 50, 'ap_hi'].mean()
below = cardio.loc[cardio['age'] <= 50, 'ap_hi'].mean()
print(f'>50 mean ap_hi:  {above:.2f}')
print(f'<=50 mean ap_hi: {below:.2f}')
print(f'Difference:      {above - below:.2f} mmHg')
```
**Gotcha:** the result depends on whether you cleaned outliers first — the un-cleaned mean ap_hi can be inflated to the thousands by data-entry typos like `ap_hi = 14000`.

---

## Question 5
**Which gender has higher mean cholesterol level, and by what margin?**

```python
g_chol = cardio.groupby('gender')['cholesterol'].mean()
print(g_chol)
print(f'Difference: {abs(g_chol[2] - g_chol[1]):.4f}')
```
**Gotcha:** cholesterol is **ordinal** (1, 2, 3) not continuous. The "mean" is interpretable as a tendency but not a true average — and a t-test on it is technically inappropriate (use Mann-Whitney U or chi-square instead).

---

## Question 6
**What percentage of the dataset is overweight (BMI > 25)?**

```python
bmi = cardio['weight'] / (cardio['height'] / 100) ** 2
pct_overweight = (bmi > 25).mean() * 100
print(f'{pct_overweight:.2f}% of the dataset is overweight.')
```
**Gotcha:** height is in **cm** — divide by 100 to get meters before squaring. Forgetting this gives BMIs in the 0.001 range.

---

## Question 7
**Among people with hypertension (ap_hi > 130 OR ap_lo > 90), what percentage are smokers?**

```python
hyper = cardio[(cardio['ap_hi'] > 130) | (cardio['ap_lo'] > 90)]
pct = hyper['smoke'].mean() * 100
print(f'Smoker rate among hypertensives: {pct:.2f}%')
```
**Gotcha:** the condition uses **OR**, not AND — clinical hypertension is diagnosed if *either* reading is elevated.

---

## Question 8
**Which single age contains the largest number of hypertensive patients?**

```python
hyper = cardio[(cardio['ap_hi'] > 130) | (cardio['ap_lo'] > 90)]
top_age = hyper['age'].value_counts().idxmax()
top_n   = hyper['age'].value_counts().max()
print(f'Age {top_age} has the most hypertensive patients ({top_n}).')
```
**Gotcha:** count, not rate. The most-represented age is partly an artifact of how the source dataset is sampled.

---

## Question 9
**What is the Pearson correlation between weight and systolic blood pressure?**

```python
r = cardio['weight'].corr(cardio['ap_hi'])
print(f'Pearson r(weight, ap_hi) = {r:.4f}')
```
**Gotcha:** the value moves substantially before vs after outlier cleaning. Always state which version you used.

---

## Question 10
**Which two features in cardio have the strongest positive correlation, and what is the value?**

```python
corr = cardio.corr().abs()
np.fill_diagonal(corr.values, 0)          # ignore self-correlations
flat = corr.unstack().sort_values(ascending=False).drop_duplicates()
print(flat.head(3))
```
Expected: `ap_hi` and `ap_lo` ≈ 0.71 (matches the reference notebook).

---

# Joined cardio + alco questions

## Question 11
**After inner-joining cardio_base with cardio_alco on `id`, how many rows match?**

```python
m = cardio.merge(alco, on='id', how='inner')
print(f'Joined rows: {len(m)}')
print(f'IDs in alco but missing from cardio: {(~alco["id"].isin(cardio["id"])).sum()}')
```
**Gotcha:** the difference between the two file sizes tells you how much each side is missing. If `cardio_alco` has duplicate IDs, an inner join inflates aggregate counts.

---

## Question 12
**Of people above 50, what percentage consume alcohol?**

```python
m = cardio.merge(alco, on='id', how='inner')
over_50 = m[m['age'] > 50]
pct = over_50['alco'].mean() * 100
print(f'{pct:.2f}% of people above 50 consume alcohol.')
```
**Gotcha:** `.mean()` of a 0/1 column gives you the proportion directly — no need for `.sum() / len()`.

---

## Question 13
**Which age group consumes alcohol the most (in absolute count)?**

```python
m = cardio.merge(alco, on='id', how='inner')
drinkers = m[m['alco'] == 1]
top = drinkers['age'].value_counts().head(5)
print('Top 5 alcohol-consuming ages:')
print(top)
```
**Gotcha:** "age group" in the reference means a single integer age. If asked for grouped buckets (e.g. 50–60), use `pd.cut`.

---

## Question 14
**Are alcohol consumers more likely to have hypertension than non-consumers? Test at 95% confidence.**

```python
m = cardio.merge(alco, on='id', how='inner')
m['hyper'] = ((m['ap_hi'] > 130) | (m['ap_lo'] > 90)).astype(int)

drinker_rate    = m.loc[m['alco'] == 1, 'hyper'].mean()
nondrinker_rate = m.loc[m['alco'] == 0, 'hyper'].mean()

# Chi-square test of independence
table = pd.crosstab(m['alco'], m['hyper'])
chi2, p, _, _ = stats.chi2_contingency(table)

print(f'Hypertension rate, drinkers:    {drinker_rate*100:.2f}%')
print(f'Hypertension rate, nondrinkers: {nondrinker_rate*100:.2f}%')
print(f'Chi-square p-value: {p:.4f}')
print('Significant at 95%' if p < 0.05 else 'Not significant at 95%')
```
**Gotcha:** for two binary variables use chi-square, not a t-test (which assumes continuous outcomes).

---

## Question 15
**With 95% confidence, do smokers have higher systolic BP than non-smokers?**

```python
smokers   = cardio.loc[cardio['smoke'] == 1, 'ap_hi']
nonsmokers = cardio.loc[cardio['smoke'] == 0, 'ap_hi']

t, p = stats.ttest_ind(smokers, nonsmokers, equal_var=False)   # Welch
print(f't = {t:.3f}, p = {p:.4g}')
print(f'Smoker mean: {smokers.mean():.2f}, non-smoker mean: {nonsmokers.mean():.2f}')
print('Smokers significantly higher' if (p < 0.05 and t > 0) else 'Not significantly higher')
```
**Gotcha:** Welch's t-test (`equal_var=False`) is safer when group sizes / variances differ — this is the case here because there are far fewer smokers.

---

# Covid_data questions

## Question 16
**On which date did the world reach its peak in daily new cases?**

```python
world = covid[covid['location'] == 'World']
peak_date = world.loc[world['new_cases'].idxmax(), 'date']
peak_val  = world['new_cases'].max()
print(f'Global peak: {peak_date.date()} with {peak_val:,.0f} new cases')

# Fallback if no 'World' row exists in the file
if world.empty:
    by_date = covid.groupby('date')['new_cases'].sum()
    print('Computed by summing countries:', by_date.idxmax().date(), by_date.max())
```
**Gotcha:** OWID files have continent and World rollups as rows where `continent` is NaN — including these in a country-level groupby double-counts.

---

## Question 17
**Among countries with population > 10 million, which had the highest deaths-per-million as of the latest date?**

```python
latest = covid['date'].max()
snap = covid[(covid['date'] == latest) & covid['continent'].notna()]
snap = snap[snap['population'] > 10_000_000].copy()
snap['deaths_per_m'] = snap['total_deaths'] / snap['population'] * 1e6

top = snap.nlargest(1, 'deaths_per_m')[['location','total_deaths','population','deaths_per_m']]
print(top)
```
**Gotcha:** filter `continent.notna()` to drop "Europe", "World", "International" rollup rows.

---

## Question 18
**What is the 7-day rolling average of daily new cases in Italy on 2020-04-15?**

```python
italy = covid[covid['location'] == 'Italy'].sort_values('date')
italy['ma7'] = italy['new_cases'].rolling(7).mean()
val = italy.loc[italy['date'] == '2020-04-15', 'ma7']
print(f'Italy 7-day MA on 2020-04-15: {float(val):,.1f}')
```
**Gotcha:** `.rolling(7)` produces NaN for the first 6 rows. If the target date is in the first week of the country's series, the answer is NaN.

---

## Question 19
**Across all countries, what is the median number of days between the first reported case and the 1,000th case?**

```python
def days_to_1000(g):
    g = g.sort_values('date')
    first = g.loc[g['total_cases'] > 0, 'date'].min()
    thou  = g.loc[g['total_cases'] >= 1000, 'date'].min()
    if pd.isna(first) or pd.isna(thou): return np.nan
    return (thou - first).days

per_country = covid[covid['continent'].notna()].groupby('location').apply(days_to_1000)
print(f'Median days from 1st case to 1000th: {per_country.median():.0f}')
print(f'Countries that never reached 1000: {per_country.isna().sum()}')
```
**Gotcha:** countries that never crossed 1,000 cases produce NaN — exclude them from the median, but report the count.

---

## Question 20
**Which continent has the highest case-fatality rate (total_deaths / total_cases) on the latest date?**

```python
latest = covid['date'].max()
snap = covid[(covid['date'] == latest) & covid['continent'].notna()]
g = snap.groupby('continent')[['total_cases','total_deaths']].sum()
g['cfr'] = g['total_deaths'] / g['total_cases']
print(g.sort_values('cfr', ascending=False))
```
**Gotcha:** sum by continent from country-level rows (with `continent.notna()` filter); using OWID's continent-rollup rows can give a different number due to differing inclusion rules for territories.

---

## How to use this for the actual MCQ

1. Run the **Setup** block once.
2. For each question, run the snippet — the printed numeric answer maps to one of the multiple-choice options.
3. If your numeric answer doesn't match any option, the most likely cause is one of these (in order):
   - Forgot to convert age from days → years.
   - Forgot the height-cm → meters conversion in BMI.
   - Didn't clean ap_hi / ap_lo outliers (or cleaned them when the question wants raw).
   - Used `inner` join when the question implies `left`, or vice versa.
   - Counted continent / World rollup rows in a country-level COVID query.
4. Re-read the question for the exact phrasing of "above" vs "above or equal to" — these flip whether the boundary value is included.
