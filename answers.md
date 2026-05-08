# Detailed Answers — Turing Data Analysis MCQ Prep

Companion to `potential_questions.md`. For data-specific questions (Q1–Q53), the exact numerical answer depends on the loaded CSVs — each entry gives the **method**, the **pandas code** to run, and the **gotcha** to watch for. For conceptual questions (Q54–Q67) full written answers are provided.

> **Critical preprocessing for all `cardio_base` questions:**
> ```python
> import pandas as pd
> import numpy as np
>
> df   = contents['cardio_base.csv'].copy()
> alco = contents['cardio_alco.csv'].copy()
> covid = contents['covid_data.csv'].copy()
>
> # cardio_base: age stored in DAYS
> df['age_years'] = (df['age'] / 365.25).round(1)
>
> # Optional clean filter — apply only when the question asks for "realistic" values
> clean = df[(df['ap_hi'] > df['ap_lo'])
>            & df['height'].between(120, 220)
>            & df['weight'].between(30, 200)
>            & df['ap_hi'].between(70, 240)
>            & df['ap_lo'].between(40, 160)]
>
> # covid_data: parse dates once
> covid['date'] = pd.to_datetime(covid['date'])
>
> # Quick health check
> for name, d in [('cardio_base', df), ('cardio_alco', alco), ('covid', covid)]:
>     print(name, d.shape, 'dups:', d.duplicated().sum(), 'nulls:', d.isna().sum().sum())
> ```

---

## 1. `cardio_base.csv` — Descriptive Statistics

**Q1. Average weight of people aged 50+?**
```python
df[df['age_years'] >= 50]['weight'].mean()
```
Gotcha: convert age from days first.

**Q2. Median height across the full dataset?**
```python
df['height'].median()
```
Gotcha: outliers (e.g. 50 cm, 250 cm) skew the mean but not the median — that's why the question asks for median.

**Q3. % with cholesterol > 1?**
```python
(df['cholesterol'] > 1).mean() * 100
```
Note: cholesterol is ordinal: 1 = normal, 2 = above normal, 3 = well above normal.

**Q4. How many smokers?**
```python
(df['smoke'] == 1).sum()
```

**Q5. Std deviation of `ap_hi`?**
```python
df['ap_hi'].std()
```
Gotcha: with bad outliers (e.g. ap_hi = 14000) the std will be huge. Decide if the question wants raw or cleaned.

**Q6. Average BMI?**
```python
bmi = df['weight'] / (df['height']/100)**2
bmi.mean()
```
Gotcha: height is in **cm**, must divide by 100.

**Q7. Which gender has higher mean weight?**
```python
df.groupby('gender')['weight'].mean()
```
In this dataset, `gender = 1` is typically female, `2` is male — verify with average height per group (taller group = male).

**Q8. % with height > mean + 1·std?**
```python
threshold = df['height'].mean() + df['height'].std()
(df['height'] > threshold).mean() * 100
```
Expected ≈ 16% under normality, but outliers will pull mean and std and change the answer.

**Q9. 90th percentile of weight?**
```python
df['weight'].quantile(0.90)
```

**Q10. Distinct ages in years?**
```python
df['age_years'].astype(int).nunique()
```

---

## 2. Conditional / Comparative

**Q11. Of people with `ap_hi` above the avg, % over 50?**
```python
high_bp = df[df['ap_hi'] > df['ap_hi'].mean()]
(high_bp['age_years'] > 50).mean() * 100
```

**Q12. Men vs women — who smokes more?**
```python
df.groupby('gender')['smoke'].agg(['sum', 'mean'])
```
`sum` = absolute count of smokers, `mean` = smoking rate.

**Q13. Smokers vs non-smokers — avg cholesterol?**
```python
df.groupby('smoke')['cholesterol'].mean()
```

**Q14. Avg weight: cardio vs no cardio?**
```python
df.groupby('cardio')['weight'].mean()
```

**Q15. Above vs below median age — which has higher cholesterol?**
```python
median_age = df['age_years'].median()
df.assign(older=df['age_years'] > median_age).groupby('older')['cholesterol'].mean()
```

**Q16. Of people > 55 yrs, % with `cholesterol == 3`?**
```python
older = df[df['age_years'] > 55]
(older['cholesterol'] == 3).mean() * 100
```

**Q17. Of `ap_hi >= 140` (hypertensive), what fraction smoke?**
```python
hyper = df[df['ap_hi'] >= 140]
(hyper['smoke'] == 1).mean()
```

**Q18. P(cardio = 1 | age > 60)?**
```python
older = df[df['age_years'] > 60]
older['cardio'].mean()
```

---

## 3. Correlation

**Q19. Strongest correlate of `cardio`?**
```python
df[['age_years','weight','cholesterol','ap_hi','cardio']].corr()['cardio'].abs().sort_values(ascending=False)
```
Typically `ap_hi` and `age` lead.

**Q20. Pearson r(height, weight)?**
```python
df['height'].corr(df['weight'])
```

**Q21. Age vs systolic or diastolic — which is more correlated?**
```python
df['age_years'].corr(df['ap_hi']), df['age_years'].corr(df['ap_lo'])
```

**Q22. Strongest negative correlation among numerics?**
```python
df.select_dtypes('number').corr().unstack().sort_values().head(10)
```
Filter out self-correlations (value = 1.0) before reading.

**Q23. Does corr(weight, ap_hi) change after outlier removal?**
```python
df['weight'].corr(df['ap_hi'])
clean['weight'].corr(clean['ap_hi'])
```
Almost always increases in magnitude after cleaning — extreme bad rows dilute true signal.

---

## 4. Data Quality Traps

**Q24. Mean age in years (after day→year conversion)?**
```python
df['age_years'].mean()
```
Expected ≈ 53. If you get ≈ 19,000, you forgot to divide by 365.25.

**Q25. Rows with `ap_lo > ap_hi`?**
```python
(df['ap_lo'] > df['ap_hi']).sum()
```

**Q26. Rows with height outside 100–220 cm?**
```python
(~df['height'].between(100, 220)).sum()
```

**Q27. Rows with weight outside 30–200 kg?**
```python
(~df['weight'].between(30, 200)).sum()
```

**Q28. Duplicate `id` values?**
```python
df['id'].duplicated().sum()
```

**Q29. Mean `ap_hi` after cleaning?**
```python
clean['ap_hi'].mean()
```
Expect a much more sensible number (≈ 126) vs the raw mean which can be in the thousands due to typos like `14000`.

**Q30. % excluded by trimming top/bottom 1% on `ap_hi` and `ap_lo`?**
```python
lo_h, hi_h = df['ap_hi'].quantile([0.01, 0.99])
lo_l, hi_l = df['ap_lo'].quantile([0.01, 0.99])
mask = df['ap_hi'].between(lo_h, hi_h) & df['ap_lo'].between(lo_l, hi_l)
(1 - mask.mean()) * 100
```

---

## 5. `cardio_alco.csv` + Joins

**Q31. % of alcohol consumers with cardio disease?**
```python
alco = contents['cardio_alco.csv']
joined = df.merge(alco, on='id', how='inner')
drinkers = joined[joined['alco'] == 1]
drinkers['cardio'].mean() * 100
```

**Q32. Inner-join match count?**
```python
df.merge(alco, on='id', how='inner').shape[0]
```

**Q33. IDs in `cardio_alco` not in `cardio_base`?**
```python
~alco['id'].isin(df['id']).sum()  # missing
alco.loc[~alco['id'].isin(df['id']), 'id'].nunique()
```

**Q34. Avg age (years) of alcohol consumers?**
```python
joined.loc[joined['alco'] == 1, 'age_years'].mean()
```

**Q35. Cardio rate: drinkers vs non-drinkers?**
```python
joined.groupby('alco')['cardio'].mean()
```

**Q36. Inflation factor from duplicate IDs in `cardio_alco`?**
If `alco` has `k` rows for an ID, an inner join on that ID produces `k` rows in the result — the count is inflated by factor `k`. Diagnose with:
```python
alco['id'].value_counts().head()
```
Fix: dedupe before joining (`alco.drop_duplicates('id')`).

**Q37. Right join strategy?**
- **Inner**: keeps only matched rows — drops base rows without alcohol info.
- **Left** (base ← alco): keeps all base rows; alcohol fields become NaN where missing → flag with `alco.isna()`.
- **Outer**: keeps everything; useful for auditing both-side mismatches.
For "keep all base, flag missing alcohol info" the answer is **left join**.

---

## 6. `covid_data.csv` — Time-Series & Aggregation

Assumes columns like `iso_code, location, continent, date, total_cases, new_cases, total_deaths, new_deaths, population` (Our World In Data schema). Run `covid.columns` first to confirm.

**Q38. Date of global peak in daily new cases?**
```python
covid = contents['covid_data.csv']
covid['date'] = pd.to_datetime(covid['date'])
world = covid[covid['location'] == 'World']  # OWID-style
world.loc[world['new_cases'].idxmax(), 'date']
# OR sum by date if no World row exists:
covid.groupby('date')['new_cases'].sum().idxmax()
```

**Q39. Country with highest total deaths on latest date?**
```python
latest = covid['date'].max()
snap = covid[(covid['date'] == latest) & covid['continent'].notna()]  # exclude continent rollups
snap.nlargest(1, 'total_deaths')[['location', 'total_deaths']]
```
Gotcha: rows where `continent` is NaN are typically continent/world rollups — exclude them when ranking countries.

**Q40. Highest deaths per million among countries with > 1M pop?**
```python
snap['deaths_per_m'] = snap['total_deaths'] / snap['population'] * 1e6
snap[snap['population'] > 1_000_000].nlargest(1, 'deaths_per_m')
```

**Q41. 7-day moving avg of new cases, country X on date Y?**
```python
c = covid[covid['location'] == 'X'].sort_values('date')
c['ma7'] = c['new_cases'].rolling(7).mean()
c.loc[c['date'] == 'Y', 'ma7']
```

**Q42. Continent with highest case-fatality rate?**
```python
g = snap.groupby('continent')[['total_cases','total_deaths']].sum()
(g['total_deaths'] / g['total_cases']).sort_values(ascending=False)
```

**Q43. Countries with zero reported cases on first date?**
```python
first = covid['date'].min()
covid[(covid['date'] == first) & (covid['total_cases'].fillna(0) == 0)]['location'].nunique()
```

**Q44. Doubling time of cases in country X between D1 and D2?**
Doubling time `T = (D2 - D1) * ln(2) / ln(C2/C1)`.
```python
import numpy as np
c = covid[covid['location'] == 'X'].set_index('date')
C1, C2 = c.loc['D1', 'total_cases'], c.loc['D2', 'total_cases']
days = (pd.Timestamp('D2') - pd.Timestamp('D1')).days
T = days * np.log(2) / np.log(C2 / C1)
```

**Q45. Top 5 countries by total cases per capita?**
```python
snap['cases_pc'] = snap['total_cases'] / snap['population']
snap.nlargest(5, 'cases_pc')[['location','cases_pc']]
```

**Q46. First date country X had > 1000 daily new cases?**
```python
c = covid[(covid['location'] == 'X') & (covid['new_cases'] > 1000)].sort_values('date')
c['date'].min()
```

**Q47. Avg days between first reported case and 100th case?**
```python
def gap(g):
    g = g.sort_values('date')
    first = g.loc[g['total_cases'] > 0, 'date'].min()
    hundred = g.loc[g['total_cases'] >= 100, 'date'].min()
    return (hundred - first).days if pd.notna(first) and pd.notna(hundred) else None

gaps = covid[covid['continent'].notna()].groupby('location').apply(gap)
gaps.dropna().mean()
```

---

## 7. COVID Data Quality

**Q48. Cumulative or daily — how to check?**
- If `total_cases` is **monotonically non-decreasing** per country → cumulative.
- If it spikes and falls → daily (or has corrections).
```python
chk = covid.sort_values(['location','date']).groupby('location')['total_cases'].apply(lambda s: (s.diff().fillna(0) < 0).sum())
chk[chk > 0].head()  # countries with non-monotonic series = corrections happened
```

**Q49. Countries with `new_cases < 0`?**
```python
neg = covid[covid['new_cases'] < 0]
neg['location'].unique()
```
These are **retroactive corrections** when prior counts were over-reported.

**Q50. Missing (country, date) pairs?**
```python
expected = pd.MultiIndex.from_product([covid['location'].unique(),
                                       pd.date_range(covid['date'].min(), covid['date'].max())])
present = covid.set_index(['location','date']).index
expected.difference(present).size
```

**Q51. Do continent rollups equal sum of members?**
Often **no** — OWID's continent rows can include territories that don't appear as separate country rows, or use slightly different reporting cut-offs. Verify:
```python
by_continent_from_countries = covid[covid['continent'].notna()].groupby(['continent','date'])['total_cases'].sum()
official_continent = covid[covid['continent'].isna() & covid['location'].isin(['Europe','Asia',...])].set_index(['location','date'])['total_cases']
```
Differences are normal — call it out as a data-quality note when reporting.

**Q52. Why does `total_cases` sometimes decrease day-over-day?**
Three causes: (a) retroactive correction of prior over-counts, (b) re-classification (e.g. probable → confirmed reversed), (c) data-pipeline error / source change. Decreases in a "cumulative" column are a red flag, not a feature.

**Q53. Two dashboards disagreeing on country case totals — three plausible reasons?**
1. **Different source cut-off times** (one snapshots 00:00 UTC, the other 06:00).
2. **Different inclusion rules** for territories (e.g. France with vs without overseas territories).
3. **Different metric definitions** — confirmed cases vs confirmed + probable, or excluding/including reinfections.

---

## 8. SQL / BigQuery Concept Questions

**Q54. `ROWS BETWEEN 6 PRECEDING` vs `RANGE INTERVAL 7 DAY PRECEDING` — when does the difference matter?**
- `ROWS` counts physical rows. A 7-row window over daily data with one missing day gives you a window spanning 8 calendar days but only 7 rows — your "weekly" average is actually wider in time.
- `RANGE INTERVAL` is calendar-aware: it includes every row within 7 days regardless of how many rows that is.
- They diverge whenever data has **gaps** or **duplicates per day**. Use `RANGE` for true rolling windows over time; use `ROWS` only when rows are guaranteed dense and unique per period.

```sql
-- ROWS — physical 7 rows
SELECT
  date,
  AVG(new_cases) OVER (
    PARTITION BY country
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS ma7_rows
FROM covid;

-- RANGE — calendar 7 days (BigQuery / Postgres)
SELECT
  date,
  AVG(new_cases) OVER (
    PARTITION BY country
    ORDER BY UNIX_DATE(date)
    RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS ma7_range
FROM covid;
```

**Q55. BigQuery cost: `SELECT *` vs `SELECT col1, col2` on a partitioned table?**
BigQuery is columnar — billing scans only the columns you select. `SELECT *` reads every column on the partition; selecting two columns reads only those two. Partitioning prunes which **rows** (partitions) get scanned; column selection prunes which **columns**. They're orthogonal. For a wide table, `SELECT *` can be 10–50× more expensive than selecting a couple of columns.

```sql
-- BAD: scans every column for each partition, even though only 2 are used
SELECT *
FROM `proj.ds.events`
WHERE event_date = '2026-05-07';

-- GOOD: scans only the 2 columns + prunes to one partition
SELECT user_id, event_name
FROM `proj.ds.events`
WHERE event_date = '2026-05-07';

-- Estimate cost without running:
--   bq query --dry_run --use_legacy_sql=false 'SELECT ... FROM ...'
-- The bytesProcessed delta between the two queries shows the savings.
```

**Q56. `COUNT(DISTINCT)` vs `APPROX_COUNT_DISTINCT` — tradeoff?**
- `COUNT(DISTINCT x)` is exact but requires hashing every value in memory; on huge cardinality (billions of users) it's slow / can OOM the slot.
- `APPROX_COUNT_DISTINCT(x)` uses HyperLogLog++; ~1–2% relative error, near-constant memory, parallel-friendly.
Use the approximate version for dashboards / monitoring; use exact only when audit-required (financial reporting, regulatory).

```sql
-- Exact (slow, expensive, audit-grade)
SELECT event_date, COUNT(DISTINCT user_id) AS dau_exact
FROM events
GROUP BY event_date;

-- Approximate (fast, dashboard-grade)
SELECT event_date, APPROX_COUNT_DISTINCT(user_id) AS dau_approx
FROM events
GROUP BY event_date;

-- Reusable HLL sketch — merge across days without re-reading rows
WITH daily AS (
  SELECT event_date, HLL_COUNT.INIT(user_id) AS sketch
  FROM events
  GROUP BY event_date
)
SELECT HLL_COUNT.MERGE(sketch) AS wau
FROM daily
WHERE event_date BETWEEN '2026-05-01' AND '2026-05-07';
```

**Q57. Correct grain for daily-active-users from `events` × `users`?**
The metric grain is `(date, user_id)`. The query is:
```sql
-- Plain DAU — no join needed
SELECT event_date, COUNT(DISTINCT user_id) AS dau
FROM events
GROUP BY event_date;

-- DAU segmented by a user attribute — only here do you need the join
SELECT e.event_date, u.country, COUNT(DISTINCT e.user_id) AS dau
FROM events e
JOIN users u USING (user_id)
GROUP BY e.event_date, u.country;
```
You typically don't need to join `users` at all — DAU only requires the user_id column on the fact. Join `users` only if you need to filter or segment by a user attribute.

**Q58. `LEFT JOIN + GROUP BY` vs `JOIN + GROUP BY` — different counts?**
- `INNER JOIN` drops left-side rows with no match; if you `COUNT(*)` you'll undercount the left set.
- `LEFT JOIN` keeps unmatched left rows with NULLs on the right; `COUNT(*)` includes them but `COUNT(right.col)` does not (NULLs don't count).
The classic bug: `COUNT(*)` after a left join overcounts a left row that has multiple right matches (one row becomes N rows). Fix: `COUNT(DISTINCT left.id)`.

```sql
-- Demonstrates all three numbers diverging:
--   users:   id=1, id=2, id=3
--   orders:  user=1 (2 rows), user=2 (1 row), user=4 (1 row)

SELECT COUNT(*)                    AS rows_inner   -- 3 (only matched orders)
FROM users u JOIN orders o ON u.id = o.user_id;

SELECT COUNT(*)                    AS rows_left    -- 4 (3 matched + 1 unmatched user)
FROM users u LEFT JOIN orders o ON u.id = o.user_id;

SELECT COUNT(DISTINCT u.id)        AS users_left   -- 3 (correct user count)
FROM users u LEFT JOIN orders o ON u.id = o.user_id;

SELECT COUNT(o.id)                 AS orders_left  -- 3 (NULL right side excluded)
FROM users u LEFT JOIN orders o ON u.id = o.user_id;
```

**Q59. Deduplicating client-retry events?**
Use a stable client-side `event_id` (UUID) on every emit, then in SQL:
```sql
-- Preferred: dedupe on stable event_id
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY received_at) AS rn
  FROM events
)
SELECT * EXCEPT(rn) FROM ranked WHERE rn = 1;

-- Fallback when event_id is missing (lossy — risks collapsing legitimate fast-fire events)
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY user_id, event_name, TIMESTAMP_TRUNC(event_timestamp, SECOND)
      ORDER BY received_at
    ) AS rn
  FROM events
)
SELECT * EXCEPT(rn) FROM ranked WHERE rn = 1;
```
If no `event_id`, dedupe on `(user_id, event_name, event_timestamp)` rounded to the second — but this is lossy and risks dropping legitimate fast-fire events.

**Q60. `country` on fact or dimension table?**
- If country **describes the user** and rarely changes → dimension table (`dim_user.country`), join on `user_id`.
- If country is **about the event itself** (e.g. country of the IP at event time, which can differ from the user's home country) → fact table (`fact_event.country_at_event`).
Star-schema rule: facts hold measurements + foreign keys; dimensions hold descriptive attributes that change slowly. When attributes change over time and you need history, use Slowly Changing Dimensions (SCD Type 2).

```sql
-- DDL sketch
CREATE TABLE dim_user (
  user_id     STRING NOT NULL,
  country     STRING,            -- where the user lives (slowly-changing)
  signup_date DATE,
  PRIMARY KEY (user_id) NOT ENFORCED
);

CREATE TABLE fact_event (
  event_id          STRING NOT NULL,
  user_id           STRING,             -- FK → dim_user
  event_timestamp   TIMESTAMP,
  event_name        STRING,
  country_at_event  STRING,             -- IP-derived, varies per event
  amount_usd        NUMERIC
)
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id;

-- SCD Type 2 if you need historical country-of-residence
CREATE TABLE dim_user_scd2 (
  user_id      STRING,
  country      STRING,
  valid_from   TIMESTAMP,
  valid_to     TIMESTAMP,        -- NULL means current row
  is_current   BOOL
);

-- Joining a fact to the SCD2 dimension as-of event time
SELECT f.event_id, d.country
FROM fact_event f
JOIN dim_user_scd2 d
  ON f.user_id = d.user_id
 AND f.event_timestamp >= d.valid_from
 AND (f.event_timestamp < d.valid_to OR d.valid_to IS NULL);
```

---

## 9. Product Event Data / Event Taxonomy

**Q61. Likely DQ issues for `product_viewed` `{user_id, product_id, timestamp, source}` in production?**
- **Anonymous user_id collisions** — pre-login traffic uses an anonymous_id that resets on browser clear.
- **Duplicate emits** — SPA route changes firing the same event on every re-render.
- **Bot traffic** — headless crawlers inflating views; need a `is_bot` filter.
- **Schema drift** — `source` enum gains new values silently (`web`, `web-mobile`, `Web` casing).
- **Late-arriving events** — mobile clients firing offline; events show up days later with old timestamps.

```sql
-- DQ probes you'd run daily in monitoring

-- 1. Schema drift on `source`
SELECT source, COUNT(*) AS n
FROM product_viewed
WHERE event_date = CURRENT_DATE() - 1
GROUP BY source
ORDER BY n DESC;

-- 2. Duplicate-emit detection (same user + product within 1 second)
SELECT user_id, product_id, COUNT(*) AS dup_emits
FROM product_viewed
WHERE event_date = CURRENT_DATE() - 1
GROUP BY user_id, product_id, TIMESTAMP_TRUNC(timestamp, SECOND)
HAVING COUNT(*) > 1;

-- 3. Late-arriving events (received >24h after event timestamp)
SELECT COUNT(*) AS late_events
FROM product_viewed
WHERE TIMESTAMP_DIFF(received_at, timestamp, HOUR) > 24;

-- 4. Anonymous-id churn — new anon_ids per real user per day
SELECT user_id, COUNT(DISTINCT anonymous_id) AS distinct_anons
FROM product_viewed
WHERE user_id IS NOT NULL
GROUP BY user_id
HAVING COUNT(DISTINCT anonymous_id) > 5;
```

**Q62. Avoid double-counting when `signup_completed` fires both client and server?**
Best practice: **server-side is source of truth** for state-change events. Treat client emit as analytics-only (UX funnel) and tag it with `source='client'`. In dashboards, count signups from server events only, OR dedupe across both on a shared correlation id (e.g. `signup_id` set server-side and echoed back to the client).

```sql
-- Option A: server-only count (simplest)
SELECT DATE(timestamp) AS d, COUNT(*) AS signups
FROM signup_completed
WHERE source = 'server'
GROUP BY d;

-- Option B: dedupe across both sources on shared signup_id
WITH unioned AS (
  SELECT signup_id, timestamp, source FROM signup_completed
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY signup_id
           ORDER BY (source = 'server') DESC, timestamp  -- prefer server row
         ) AS rn
  FROM unioned
)
SELECT DATE(timestamp) AS d, COUNT(*) AS signups
FROM ranked WHERE rn = 1
GROUP BY d;
```

**Q63. Event vs user property in Amplitude/Mixpanel — why it matters for retention?**
- **Event property**: snapshot at event time. `plan='pro'` on a `feature_used` event means the user was Pro **when that event fired**.
- **User property**: current value, overwritten on each set call. Pulling `plan='pro'` retention today returns users who are Pro **now**, even if they were Free during the retention window.
Wrong choice = wrong cohort. For "did Pro users come back", use the event property. For "are current Pro users active", use the user property.

```sql
-- "Users who were Pro at time of activity AND came back the next day"
-- (uses event property — historically accurate)
WITH pro_today AS (
  SELECT DISTINCT user_id, DATE(timestamp) AS d
  FROM events
  WHERE event_properties.plan = 'pro'
    AND DATE(timestamp) = '2026-05-07'
),
back_tomorrow AS (
  SELECT DISTINCT user_id
  FROM events
  WHERE DATE(timestamp) = '2026-05-08'
)
SELECT COUNT(*) AS retained
FROM pro_today p JOIN back_tomorrow b USING (user_id);

-- "Users who are Pro now AND were active yesterday"
-- (uses user property — current state, may misclassify history)
SELECT COUNT(DISTINCT e.user_id) AS active_pros
FROM events e
JOIN dim_user u USING (user_id)
WHERE u.current_plan = 'pro'
  AND DATE(e.timestamp) = '2026-05-07';
```

**Q64. Why might SQL DAU disagree with Amplitude DAU?**
- **Timezone**: Amplitude project tz vs UTC-based SQL.
- **Identity resolution**: Amplitude merges anonymous → identified user; raw SQL on `user_id` does not.
- **Sampling**: free/lower Amplitude tiers sample at scale.
- **Filters**: Amplitude may drop events without certain required properties; raw SQL keeps them.
- **Retroactive backfills** in the warehouse vs Amplitude's frozen ingestion.

```sql
-- Reconciliation query: aligned to Amplitude's project timezone (e.g. America/Los_Angeles)
-- and using a merged identity column to mimic Amplitude's identity resolution.
SELECT
  DATE(timestamp, 'America/Los_Angeles') AS d_local,
  COUNT(DISTINCT COALESCE(amplitude_id, user_id, anonymous_id)) AS dau_aligned
FROM events
GROUP BY d_local;

-- Diff against Amplitude's DAU export
SELECT a.d_local,
       a.dau_aligned    AS sql_dau,
       b.dau            AS amp_dau,
       a.dau_aligned - b.dau AS delta
FROM sql_dau a
LEFT JOIN amplitude_dau b USING (d_local)
ORDER BY ABS(delta) DESC;
```

---

## 10. Metric Definition / Dashboard Discrepancy

**Q65. Two dashboards, different revenue for yesterday — three plausible root causes?**
1. **Timezone** — one uses UTC `event_date`, the other uses local store time.
2. **Late-arriving data** — one refreshed at 06:00, the other at 12:00; an extra 6 hours of orders is included in the later one.
3. **Definition** — gross (with tax + shipping) vs net (excluding refunds, discounts).
Other common causes: filter on order status (paid vs created), inclusion of test merchants, currency conversion timestamps.

```sql
-- Decompose the gap between two revenue numbers for the same date

WITH dash_a AS (   -- e.g. gross, UTC, includes pending
  SELECT DATE(created_at) AS d,
         SUM(amount + tax + shipping) AS rev
  FROM orders
  WHERE DATE(created_at) = '2026-05-07'
  GROUP BY d
),
dash_b AS (        -- e.g. net, local tz, only paid
  SELECT DATE(created_at, 'America/New_York') AS d,
         SUM(amount - discount - refund) AS rev
  FROM orders
  WHERE DATE(created_at, 'America/New_York') = '2026-05-07'
    AND status = 'paid'
    AND is_test = FALSE
  GROUP BY d
)
SELECT a.rev AS dash_a_rev,
       b.rev AS dash_b_rev,
       a.rev - b.rev AS gap
FROM dash_a a CROSS JOIN dash_b b;

-- Then peel back one filter at a time to attribute the gap.
```

**Q66. Rolling 7-day WAU vs calendar-week WAU — how does it change?**
- **Rolling 7-day**: every day, count distinct users in last 7 days. Smooth, lag-free, but yesterday's number changes tomorrow as the window rolls.
- **Calendar week** (Mon–Sun): one number per ISO week. No daily smoothing; numbers stable once the week closes but cannot be read mid-week without partial-week caveats.
Same underlying user behavior, very different chart shape and reporting cadence. Always specify which definition is used.

```sql
-- Rolling 7-day WAU: one row per day, recomputed nightly
WITH daily_users AS (
  SELECT DATE(timestamp) AS d, user_id
  FROM events
  GROUP BY d, user_id
)
SELECT d,
       COUNT(DISTINCT user_id) OVER (
         ORDER BY UNIX_DATE(d)
         RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS wau_rolling
FROM daily_users
ORDER BY d;

-- Calendar-week WAU: one row per ISO week
SELECT DATE_TRUNC(DATE(timestamp), WEEK(MONDAY)) AS week_start,
       COUNT(DISTINCT user_id) AS wau_calendar
FROM events
GROUP BY week_start
ORDER BY week_start;
```

**Q67. Systematically detect "gross vs net" naming clashes across an org?**
- **Metric layer / semantic layer** (LookML, dbt metrics, Cube) — define each metric exactly once with a canonical SQL expression and let dashboards reference it by name.
- **Linter on dashboard SQL** — extract metric names and warn when two dashboards use `revenue` with different aggregation expressions.
- **Metric catalog** with mandatory `definition`, `owner`, `sql_expression` fields, surfaced in BI tool tooltips so consumers see the formula at view time.
- **Cross-dashboard test** in CI — compute the same metric from two sources nightly and alert on drift > X%.
Root fix: stop hand-rolling metrics in each dashboard. Centralize the definition.

```yaml
# dbt semantic-model snippet — single source of truth for "revenue"
metrics:
  - name: net_revenue
    label: Net Revenue
    description: Gross sales minus discounts and refunds. USD.
    type: simple
    type_params:
      measure: net_revenue_usd
    owner: finance-data@company.com

  - name: gross_revenue
    label: Gross Revenue
    description: Sum of order amount + tax + shipping, before refunds.
    type: simple
    type_params:
      measure: gross_revenue_usd
    owner: finance-data@company.com
```

```sql
-- Drift-detection job: compare two pipelines computing the "same" metric
WITH a AS (SELECT d, SUM(rev) v FROM dash_a_revenue GROUP BY d),
     b AS (SELECT d, SUM(rev) v FROM dash_b_revenue GROUP BY d)
SELECT a.d,
       a.v AS rev_a,
       b.v AS rev_b,
       SAFE_DIVIDE(ABS(a.v - b.v), a.v) AS pct_drift
FROM a JOIN b USING (d)
WHERE SAFE_DIVIDE(ABS(a.v - b.v), a.v) > 0.01   -- alert at 1% drift
ORDER BY pct_drift DESC;
```
