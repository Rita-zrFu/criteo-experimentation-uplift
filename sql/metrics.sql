-- Core experiment aggregation. The Python runner creates experiment_data as a
-- view over the processed Parquet file before executing this query.
SELECT
    treatment,
    COUNT(*) AS users,
    SUM(conversion) AS conversions,
    AVG(conversion) AS conversion_rate,
    SUM(visit) AS visits,
    AVG(visit) AS visit_rate,
    SUM(exposure) AS exposures,
    AVG(exposure) AS exposure_rate
FROM experiment_data
GROUP BY treatment
ORDER BY treatment;

