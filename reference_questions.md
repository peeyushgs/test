# Turing Reference Questions — Full Answers with Code

The actual reference questions for the Turing Data Analysis MCQ. Two sections:
1. **Cardiovascular Data Analysis** (10 questions)
2. **COVID-19 Data Analysis** (11 questions — Q4–Q9 reuse the cardio data)

Each answer includes runnable pandas code, the gotcha to watch for, and the expected output style.

---

## Setup (run once)

```python
import io, requests, numpy as np, pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
from zipfile import ZipFile

DATA_FILE = 'https://drive.usercontent.google.com/uc?id=1TMOK9yFDm6nJ2PAC-unBbrG-tR5THJ90&export=download'
zf = ZipFile(io.BytesIO(requests.get(DATA_FILE).content))
contents = {n: pd.read_csv(zf.open(n), sep='[,;]', engine='python') for n in zf.namelist()}

cardio = contents['cardio_base.csv'].copy()
alco   = contents['cardio_alco.csv'].copy()
covid  = contents['covid_data.csv'].copy()

# Cardio: convert age days → years (rounded down)
cardio['age'] = (cardio['age'] / 365.25).astype(int)

# Drop biologically impossible BP values
cardio = cardio[(cardio['ap_hi'].between(70, 200)) &
                (cardio['ap_lo'].between(40, 140))].copy()

# Covid: parse dates
covid['date'] = pd.to_datetime(covid['date'])

print('cardio:', cardio.shape, '| alco:', alco.shape, '| covid:', covid.shape)
```

---

# Cardiovascular Data Analysis

## Q1 — How much heavier is the age group with the highest average weight than the age group with the lowest weight?

```python
avg_weight_by_age = cardio.groupby('age')['weight'].mean()
hi_age, hi_w = avg_weight_by_age.idxmax(), avg_weight_by_age.max()
lo_age, lo_w = avg_weight_by_age.idxmin(), avg_weight_by_age.min()
diff = hi_w - lo_w

print(f'Heaviest group:  age {hi_age}, {hi_w:.2f} kg')
print(f'Lightest group:  age {lo_age}, {lo_w:.2f} kg')
print(f'Difference:      {diff:.2f} kg')
```
**Gotcha:** `groupby('age')` only makes sense after the days→years conversion. Without it you'd be grouping by ~19,000 distinct integers.

---

## Q2 — Who smokes more, men or women?

```python
men_pct   = cardio.loc[cardio['gender'] == 2, 'smoke'].mean() * 100
women_pct = cardio.loc[cardio['gender'] == 1, 'smoke'].mean() * 100

print(f'Men smokers:   {men_pct:.2f}%')
print(f'Women smokers: {women_pct:.2f}%')
print(f'Men are {men_pct / women_pct:.1f}x more likely to smoke.')
```
**Gotcha:** gender encoding is `1 = female`, `2 = male` (verify by checking which group has higher mean height — the taller group is male).

---

## Q3 — How tall are the tallest 1% of people?

```python
top_1_threshold = np.percentile(cardio['height'], 99)
print(f'The tallest 1% are taller than {top_1_threshold:.0f} cm.')
```
**Gotcha:** `np.percentile(x, 99)` gives the cutoff above which the top 1% sits — not the height of the single tallest person.

---

## Q4 — Which two features have the highest Spearman rank correlation?

```python
spearman = cardio.corr(method='spearman').abs()
np.fill_diagonal(spearman.values, 0)                          # ignore self-pairs

flat = spearman.unstack().sort_values(ascending=False)
flat = flat[flat.index.get_level_values(0) < flat.index.get_level_values(1)]   # dedupe (a,b)/(b,a)
print(flat.head(5))
```
**Gotcha:** **Spearman**, not Pearson — Spearman is rank-based and robust to non-linear monotonic relationships. The top pair is almost always `(ap_hi, ap_lo)`.

---

## Q5 — What percentage of people are more than 2 standard deviations away from the average height?

```python
mu, sd = cardio['height'].mean(), cardio['height'].std()
mask = (cardio['height'] - mu).abs() > 2 * sd
pct = mask.mean() * 100
print(f'{pct:.2f}% of people are more than 2σ from mean height.')
```
**Gotcha:** "**more than 2 SD**" means **|x − μ| > 2σ**, both tails — not just the upper tail. Under a true normal distribution this is ~4.6%; the actual percentage will differ because the height distribution is bimodal (men + women).

---

## Q6 — How many people over 50 years old consume alcohol?

```python
m = cardio.merge(alco, on='id', how='inner')
n = m[(m['age'] > 50) & (m['alco'] == 1)].shape[0]
print(f'People over 50 who consume alcohol: {n}')
```
**Gotcha:** "over 50" = **strictly greater than 50** (`> 50`), not `>= 50`. This single decision can change the answer by hundreds of rows.

---

## Q7 — What percentage of people over 50 years old drink alcohol?

```python
m = cardio.merge(alco, on='id', how='inner')
over_50 = m[m['age'] > 50]
pct = over_50['alco'].mean() * 100
print(f'{pct:.2f}% of people over 50 drink alcohol.')
```
**Gotcha:** denominator must be **only people over 50 in the joined dataframe**, not the whole population.

---

## Q8 — What is the percentage difference in cholesterol levels between people over 50 and people under 50?

```python
over  = cardio.loc[cardio['age'] > 50, 'cholesterol'].mean()
under = cardio.loc[cardio['age'] < 50, 'cholesterol'].mean()    # strictly under
pct_diff = (over - under) / under * 100

print(f'Mean cholesterol — over 50:  {over:.4f}')
print(f'Mean cholesterol — under 50: {under:.4f}')
print(f'Percentage difference:        {pct_diff:.2f}% higher in over-50s')
```
**Gotcha:** the rows where `age == 50` belong in neither bucket given the wording. Either include them in one group consistently, or exclude — but be explicit.

---

## Q9 — Do smokers or non-smokers have higher cholesterol levels?

```python
smk_chol  = cardio.loc[cardio['smoke'] == 1, 'cholesterol']
nsmk_chol = cardio.loc[cardio['smoke'] == 0, 'cholesterol']

print(f'Smoker mean cholesterol:     {smk_chol.mean():.4f}')
print(f'Non-smoker mean cholesterol: {nsmk_chol.mean():.4f}')

# Mann-Whitney U test (cholesterol is ordinal, not continuous → t-test is technically wrong)
u, p = stats.mannwhitneyu(smk_chol, nsmk_chol, alternative='two-sided')
print(f'Mann-Whitney U p-value: {p:.4g}')
```
**Gotcha:** cholesterol is ordinal (1, 2, 3). Mann-Whitney U is the correct non-parametric test; a t-test will give a similar answer in practice but is technically inappropriate.

---

## Q10 — Which of the following statements is true with 95% confidence?

The Turing version asks you to evaluate a list of comparisons — typically:
1. Smokers have higher BP than non-smokers
2. Smokers have higher cholesterol than non-smokers
3. People over 50 have higher cholesterol than younger people
4. Men are taller than women

Run all of them at α = 0.05 and pick the ones with `p < 0.05` **and** the direction matching the claim:

```python
def claim(name, sample_a, sample_b, expectation='greater'):
    """expectation = 'greater' means we test mean(a) > mean(b)"""
    t, p_two = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    p = p_one if expectation == 'greater' else (1 - p_one)
    direction_ok = (t > 0) if expectation == 'greater' else (t < 0)
    sig = (p < 0.05) and direction_ok
    print(f'{name:55s} t={t:+.2f}  p={p:.4g}  → {"TRUE" if sig else "false"}')

# Smokers vs non-smokers — BP
claim('Smokers have higher ap_hi than non-smokers',
      cardio.loc[cardio.smoke==1, 'ap_hi'], cardio.loc[cardio.smoke==0, 'ap_hi'])

# Smokers vs non-smokers — cholesterol
claim('Smokers have higher cholesterol than non-smokers',
      cardio.loc[cardio.smoke==1, 'cholesterol'], cardio.loc[cardio.smoke==0, 'cholesterol'])

# Over-50 vs under-50 — cholesterol
claim('Over 50 have higher cholesterol than under 50',
      cardio.loc[cardio.age>50, 'cholesterol'], cardio.loc[cardio.age<50, 'cholesterol'])

# Men vs women — height
claim('Men are taller than women',
      cardio.loc[cardio.gender==2, 'height'], cardio.loc[cardio.gender==1, 'height'])
```
**Gotcha:** a two-sided p < 0.05 with the wrong direction is **not** evidence for the claim. Always check the sign of `t`. Welch's t-test (`equal_var=False`) is safer when group sizes differ.

---

# COVID-19 Data Analysis

## Q1 — Which country has the 3rd highest cumulative deaths-per-million?

```python
latest = covid['date'].max()
snap = covid[(covid['date'] == latest) & covid['continent'].notna()].copy()
snap['deaths_per_m'] = snap['total_deaths'] / snap['population'] * 1e6

ranked = snap.dropna(subset=['deaths_per_m']).nlargest(10, 'deaths_per_m')[
    ['location', 'total_deaths', 'population', 'deaths_per_m']
]
print(ranked)
print(f'\n3rd highest: {ranked.iloc[2]["location"]} '
      f'({ranked.iloc[2]["deaths_per_m"]:.1f} per million)')
```
**Gotcha:** filter `continent.notna()` to drop "World", "Europe", "North America" rollup rows. Tiny territories with reporting quirks (e.g. San Marino, Vatican) often dominate the top of the list — confirm whether the question wants them included.

---

## Q2 — For Italy, days since 2020-02-28 for each entry in [2020-02-28, 2020-03-20]?

```python
italy = covid[covid['location'] == 'Italy'].copy()
mask = italy['date'].between('2020-02-28', '2020-03-20')
italy_window = italy[mask].sort_values('date').reset_index(drop=True)
italy_window['days_since_feb28'] = (italy_window['date'] - pd.Timestamp('2020-02-28')).dt.days

print(italy_window[['date', 'days_since_feb28', 'total_cases']].to_string(index=False))
```
**Gotcha:** `.dt.days` requires the column to be `datetime64`. If the date column is a string, the subtraction silently fails — always run `pd.to_datetime` first (already done in setup).

---

## Q3 / Q11 — Difference between exponential curve and real cases on 2020-03-20?

The standard interpretation: fit an exponential to Italy's early case curve (using data up to and including some training date), extrapolate to 2020-03-20, then compute predicted − actual.

```python
italy = covid[covid['location'] == 'Italy'].copy().sort_values('date').reset_index(drop=True)
italy = italy[italy['date'] >= '2020-02-28'].reset_index(drop=True)
italy['days'] = (italy['date'] - pd.Timestamp('2020-02-28')).dt.days

# Fit y = a * exp(b * t) on data up to 2020-03-20
train = italy[italy['date'] <= '2020-03-20']
def exp_model(t, a, b):
    return a * np.exp(b * t)
popt, _ = curve_fit(exp_model, train['days'], train['total_cases'], p0=[1, 0.3], maxfev=10000)
a, b = popt
print(f'Fit: total_cases ≈ {a:.2f} * exp({b:.4f} * days)')

# Predicted vs actual on 2020-03-20
target_days = (pd.Timestamp('2020-03-20') - pd.Timestamp('2020-02-28')).days
predicted = exp_model(target_days, a, b)
actual = italy.loc[italy['date'] == '2020-03-20', 'total_cases'].iloc[0]

print(f'Predicted on 2020-03-20: {predicted:,.0f}')
print(f'Actual on 2020-03-20:    {actual:,.0f}')
print(f'Difference (pred-actual): {predicted - actual:,.0f}')
```
**Gotcha:** fit on **log-scale** if `curve_fit` struggles to converge:
```python
log_fit = np.polyfit(train['days'], np.log(train['total_cases'].clip(lower=1)), 1)
b, log_a = log_fit
predicted = np.exp(log_a) * np.exp(b * target_days)
```
The two methods can give different numerical answers depending on weighting — the question's expected answer likely uses one specific convention.

---

## Q4 — How much heavier is the age group with the highest average weight than the age group with the lowest weight?
**Same as cardio Q1** — see above.

---

## Q5 — Do people over 50 have higher cholesterol levels than the rest?

```python
over  = cardio.loc[cardio['age'] > 50, 'cholesterol']
rest  = cardio.loc[cardio['age'] <= 50, 'cholesterol']

print(f'Over 50 mean:     {over.mean():.4f}')
print(f'50 or younger:    {rest.mean():.4f}')

u, p = stats.mannwhitneyu(over, rest, alternative='greater')
print(f'Mann-Whitney p (one-sided, over > rest): {p:.4g}')
print('YES — over-50s have significantly higher cholesterol' if p < 0.05
      else 'No significant difference')
```
**Gotcha:** "the rest" includes age == 50; this differs from cardio Q8's "under 50".

---

## Q6 — Are men more likely to be smokers than women?
**Same as cardio Q2** — see above.

---

## Q7 — How tall are the tallest 1% of people?
**Same as cardio Q3** — see above.

---

## Q8 — Which two features have the highest Spearman rank correlation?
**Same as cardio Q4** — see above.

---

## Q9 — What percentage of people are more than 2 SD from the average height?
**Same as cardio Q5** — see above.

---

## Q10 — When did the difference in confirmed cases between Italy and Germany first exceed 10,000?

```python
italy = covid.loc[covid['location'] == 'Italy', ['date', 'total_cases']].rename(
    columns={'total_cases': 'italy'})
germany = covid.loc[covid['location'] == 'Germany', ['date', 'total_cases']].rename(
    columns={'total_cases': 'germany'})

merged = italy.merge(germany, on='date').sort_values('date')
merged['diff'] = (merged['italy'] - merged['germany']).abs()

first_over = merged[merged['diff'] > 10_000].head(1)
print(first_over[['date', 'italy', 'germany', 'diff']].to_string(index=False))
```
**Gotcha:** use **absolute** difference unless the question specifies "Italy − Germany" direction. The earliest crossover date typically falls in early March 2020 because Italy's curve led Germany's by ~10 days.

---

## How to use this for the actual MCQ

1. Run **Setup** once.
2. For each question, run its snippet and read the printed answer.
3. If your answer doesn't match any option, recheck:
   - Days → years on `age` (one-time conversion, must apply before any age-based query).
   - Inclusive vs exclusive bounds (`> 50` vs `>= 50`).
   - Height in cm (no conversion needed for percentile, but needed for BMI).
   - Outlier filtering on `ap_hi` / `ap_lo` — apply or skip consistently.
   - For COVID country queries, drop continent-rollup rows (`continent.notna()`).
   - Spearman vs Pearson — the question explicitly asks for one.
4. Trust the test direction: a two-sided significant result with the wrong sign is not evidence for a directional claim.
