# Methodology and limitations

This is a 6-hour-ahead historical best-track wind baseline, not an operational forecast.

IBTrACS v04r01 combines different agencies. USA_WIND and WMO_WIND can represent different wind averaging conventions, so fallback across sources may introduce inconsistent targets. For rigorous basin comparisons, harmonize wind averaging periods or restrict to a consistent source.

Observations are matched at exactly 6-hour intervals. Prior-wind changes are calculated only when the preceding observation is exactly six hours earlier. The held-out test split is by storm SID, not by observation, preventing direct within-storm leakage. A stricter deployment study should also use a chronological holdout by season.

The persistence baseline assumes the current wind remains unchanged for six hours. Model skill should be compared against this baseline rather than presenting Random Forest metrics alone.

Feature importance is impurity-based and may be biased by correlated predictors. Permutation importance and per-basin error audits are recommended future improvements.

Do not use these predictions for evacuation, navigation, or safety-critical decisions.
