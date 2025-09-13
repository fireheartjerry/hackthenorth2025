# modes.py
MODES = {
  "unicorn_hunting": {
    "filters": {
      "lob_in": ["COMMERCIAL PROPERTY"],
      "new_business_only": True,
      "loss_ratio_max": 0.40,
      "tiv_max": 100_000_000,
      "premium_range": [150_000, 1_800_000],
      "min_winnability": 0.60,
      "min_year": 2010,
      "good_construction_only": True
    },
    "weights": {
      "w_prem": 0.35, "w_win": 0.35, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.05, "w_fresh": 0.05,
      "premium_lo": 150_000, "premium_mid_lo": 300_000, "premium_mid_hi": 1_000_000, "premium_hi": 1_800_000,
      "tiv_hi": 100_000_000
    }
  },
  "balanced_growth": {
    "filters": {
      "lob_in": ["COMMERCIAL PROPERTY", "HABITATIONAL"],
      "new_business_only": True,
      "loss_ratio_max": 0.65
    },
    "weights": {
      "w_prem": 0.25, "w_win": 0.25, "w_year": 0.15, "w_con": 0.15, "w_tiv": 0.10, "w_fresh": 0.10,
      "premium_lo": 100_000, "premium_mid_lo": 250_000, "premium_mid_hi": 900_000, "premium_hi": 1_500_000,
      "tiv_hi": 150_000_000
    }
  },
  "loose_fits": {
    "filters": {
      "loss_ratio_max": 0.85
    },
    "weights": {
      "w_prem": 0.20, "w_win": 0.30, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.20,
      "premium_lo": 50_000, "premium_mid_lo": 150_000, "premium_mid_hi": 600_000, "premium_hi": 1_500_000,
      "tiv_hi": 150_000_000
    }
  },
  "turnaround_bets": {
    "filters": {
      "loss_ratio_max": 1.0,
      "premium_range": [50_000, 300_000],
      "tiv_max": 75_000_000
    },
    "weights": {
      "w_prem": 0.15, "w_win": 0.25, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.30,
      "premium_lo": 50_000, "premium_mid_lo": 120_000, "premium_mid_hi": 250_000, "premium_hi": 300_000,
      "tiv_hi": 75_000_000
    }
  }
}
