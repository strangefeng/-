# final_recommendation

1. Whether a new model exceeded N2
- No. No completed round14 model exceeded the N2 anchor on standard validation.

2. Whether HN150/HN300 are better than HN50
- Not concluded yet. HN150/HN300 were blocked because a curated train-only negative pool was not available.

3. Main bottleneck
- Label quality and train-side annotation consistency remain a major bottleneck.

4. Recommended paper main model
- Keep `N2` as the main paper model unless a later clean-train or hard-negative expansion run clearly exceeds it.

5. Recommended digital-twin deployment model
- Prefer the strongest hard-negative-suppressed model that keeps mAP close to N2, because deployment visuals are sensitive to false positives.

6. Continue experiments or stop
- Continue with a manual clean-train branch; that is currently the highest-value next step.
