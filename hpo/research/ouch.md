# HPO Errors ("Ouch!")

Purpose: Painful lessons, systematic mistakes, biases, and failure modes discovered during HPO research. The goal is learning, not blame.

| Nr | Error | Topics | Status |
|---|---|---|---|
| [[#E1 Selection Bias In HP Robustness\|E1]] | Selection Bias In HP Robustness | HP, Evaluation | open |

Topics: `HP` = Hyperparameters.

## E1 Selection Bias In HP Robustness

**Error:** The original best trial score is displayed together with fresh HP-robustness scores, although that trial was selected as the winner from many stochastic trials.

**Why it matters:** This double-counts luck. The winner was selected on an unusually good observed score, so including that same score in a supposedly neutral follow-up evaluation overestimates the HP quality.

**Fix:** Treat the original winner score as a reference only. Judge HP robustness from fresh retrainings/seeds that were not used for selecting the candidate.

**Keywords:** selection bias, winner's curse, regression to the mean, double dipping.

**When:** 2026-07-02.
