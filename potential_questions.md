# Potential MCQ Questions — Turing Data Analysis Assessment

Based on the three datasets loaded in the notebook (`cardio_base.csv`, `cardio_alco.csv`, `covid_data.csv`) and the Data Analyst / Data Scientist job description (SQL, dimensional modeling, product event data, BigQuery, data quality).

---

## 1. `cardio_base.csv` — Descriptive Statistics

1. What is the average weight of people aged 50 and above (in years)?
2. What is the median height across the full dataset?
3. What percentage of the population has cholesterol level > 1?
4. How many people in the dataset are smokers?
5. What is the standard deviation of systolic blood pressure (`ap_hi`)?
6. What is the average BMI (weight / height²) across all rows?
7. Which gender has the higher mean weight?
8. What percentage of records have a height greater than 1 standard deviation above the mean?
9. What is the 90th percentile of weight?
10. How many distinct ages (in years) appear in the dataset?

---

## 2. `cardio_base.csv` — Conditional / Comparative Questions

11. Among people whose systolic blood pressure (`ap_hi`) is above the population average, what percentage are over 50 years old?
12. Do men or women smoke more, in absolute and percentage terms?
13. Among smokers, what is the average cholesterol level vs non-smokers?
14. What is the average weight difference between people with cardiovascular disease and those without?
15. Which group has higher mean cholesterol — those above or below the median age?
16. Among people above age 55, what percentage have `cholesterol = 3` (well above normal)?
17. Of people with `ap_hi >= 140` (hypertensive), what fraction are smokers?
18. What is the probability that a randomly selected person above age 60 has cardiovascular disease?

---

## 3. `cardio_base.csv` — Correlation & Relationships

19. Which variable is most strongly correlated with `cardio` (the disease flag) — age, weight, cholesterol, or `ap_hi`?
20. What is the Pearson correlation between height and weight?
21. Is age more correlated with systolic or diastolic blood pressure?
22. Which two numeric variables have the strongest negative correlation?
23. Does the correlation between weight and `ap_hi` change after removing outliers?

---

## 4. `cardio_base.csv` — Data Quality Traps

24. The `age` column is stored in **days**, not years — what is the average age once converted?
25. How many rows have `ap_lo > ap_hi` (impossible blood pressure)?
26. How many rows have height < 100 cm or > 220 cm (likely data-entry errors)?
27. How many rows have weight < 30 kg or > 200 kg?
28. How many duplicate `id` values exist?
29. After cleaning the impossible ranges, what is the new mean systolic blood pressure?
30. What percentage of records would be excluded if you trimmed the top and bottom 1% of `ap_hi` and `ap_lo`?

---

## 5. `cardio_alco.csv` + Joins

31. What percentage of people who consume alcohol also have cardiovascular disease? *(requires join)*
32. After joining `cardio_base` and `cardio_alco` on `id`, how many rows are matched?
33. Are there `id` values in `cardio_alco` that don't exist in `cardio_base`?
34. Among alcohol consumers, what is the average age (in years)?
35. Is there a higher cardiovascular-disease rate among alcohol consumers vs non-consumers?
36. If you do a naive inner join and `cardio_alco` has duplicate IDs, by what factor are your aggregate counts inflated?
37. What's the right join strategy — inner, left, or outer — to keep all base records and flag missing alcohol info?

---

## 6. `covid_data.csv` — Time-Series & Aggregation

38. On which date did the world reach its peak daily new cases?
39. Which country had the highest total deaths as of the latest date in the dataset?
40. Which country had the highest **deaths per million** among countries with > 1M population?
41. What is the 7-day moving average of new cases for country X on date Y?
42. Which continent has the highest case-fatality rate (deaths / total_cases)?
43. How many countries had zero reported cases on the first date in the dataset?
44. What is the doubling time of cases in country X between dates D1 and D2?
45. Rank the top 5 countries by total cases per capita.
46. On what date did country X first record more than 1,000 daily new cases?
47. Across all countries, what's the average days between first reported case and 100th case?

---

## 7. `covid_data.csv` — Data Quality & Methodology

48. Are `total_cases` cumulative or daily? How can you verify?
49. Are there countries with `new_cases < 0` (corrections to prior data)?
50. How many (country, date) pairs are missing from the expected grid?
51. Do continent-level rollups in the dataset equal the sum of their member countries? If not, why might that be?
52. Are there countries whose `total_cases` decreases day-over-day? What does that indicate?
53. If two dashboards show different total case counts for the same country, what are three plausible reasons?

---

## 8. SQL / BigQuery Concept Questions (likely framed as MCQ)

54. Which SQL construct correctly computes a 7-day rolling average — `OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` or `RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW`? When does the difference matter?
55. In BigQuery, what's the cost difference between `SELECT *` and `SELECT col1, col2` on a partitioned table?
56. Which is more efficient — `COUNT(DISTINCT user_id)` or `APPROX_COUNT_DISTINCT(user_id)` — and what's the tradeoff?
57. Given a fact table `events` and dimension `users`, what's the correct grain for a daily-active-users metric?
58. Why might `LEFT JOIN` + `GROUP BY` produce different counts than `JOIN` + `GROUP BY`?
59. What is the right way to deduplicate event data when the same event can fire twice from a client retry?
60. In dimensional modeling, should `country` live on the fact table or a dimension table? Why?

---

## 9. Product Event Data / Event Taxonomy (per JD)

61. Given a Segment-style event `product_viewed` with properties `{user_id, product_id, timestamp, source}`, what's a likely data quality issue you'd see in production?
62. If `signup_completed` fires both client-side and server-side, how do you avoid double-counting?
63. What's the difference between an Amplitude/Mixpanel-style **event** vs a **user property**, and why does it matter for retention queries?
64. Why might a daily active users (DAU) number from a SQL query disagree with the number shown in Amplitude?

---

## 10. Metric Definition / Dashboard Discrepancy (per JD)

65. Two dashboards show different revenue numbers for yesterday — list three plausible root causes (timezone, late-arriving data, definition of "revenue", refresh schedule, filters).
66. If "weekly active users" is defined as "users with ≥1 event in the last 7 days", how does the result change between a rolling window and a calendar week?
67. A metric is `gross_revenue` on one dashboard and `net_revenue` on another — how would you systematically detect this kind of discrepancy across an organization?

---

## Tips for Tackling These

- Always inspect: `df.info()`, `df.describe()`, `df.isna().sum()`, `df.duplicated().sum()` first.
- Convert `cardio_base.age` from days → years before any age-based question.
- Filter out impossible ranges (`ap_lo > ap_hi`, height < 100, weight < 30) before aggregating.
- Confirm the grain of `covid_data` (one row per country-date) before summing.
- For "which is correlated most strongly" questions, use `df.corr()` and read the absolute value, not the signed one.
