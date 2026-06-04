# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] — 2026-06-02

- Hotfix: decline-curve module crashed on import under numpy ≥ 2.3 (where `np.trapz`
  was removed), taking the whole app down. Resolve `np.trapezoid` without eagerly
  touching the removed `np.trapz` alias.

## [0.3.0] — 2026-06-02

- True type-curve benchmark (early-window fit + cumulative deferred bbl/$)
- Monte-Carlo intervention economics (P10/P50/P90 + tornado sensitivity)
- Eval dashboard (20-case agreement, confusion breakdown, $/review) + CI regression gate
- Structured, validated diagnosis export for AFE-Copilot chaining
- Fixed payout off-by-one; replaced mislabeled "rate of return" with discounted profitability index

## [0.2.0]

- Initial public demo
</content>
</invoke>
