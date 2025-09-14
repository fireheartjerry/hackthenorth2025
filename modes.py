# modes.py

MODES = {
  # Ultra-selective, high-value, new-business focus (tightened ~65%)
  "unicorn_hunting": {
    "filters": {
      "lob_in": ["COMMERCIAL PROPERTY"],
      "new_business_only": True,
      "loss_ratio_max": 0.14,                  # much more conservative LR
      "tiv_max": 35_000_000,                   # 65% tighter
      "premium_range": [52_500, 630_000],      # 65% tighter
      "min_winnability": 0.79,                 # 65% closer to 1
      "min_year": 2017,                        # newer buildings
      "good_construction_only": True
    },
    "weights": {
      "w_prem": 0.35, "w_win": 0.35, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.05, "w_fresh": 0.05,
      "premium_lo": 52_500, "premium_mid_lo": 105_000, "premium_mid_hi": 350_000, "premium_hi": 630_000,
      "tiv_hi": 35_000_000
    }
  },

  # Realistic, portfolio-friendly balance of value and risk (tightened ~65%)
  "balanced_growth": {
    "filters": {
      "lob_in": [
        "COMMERCIAL PROPERTY", "HABITATIONAL", "GENERAL LIABILITY", "UMBRELLA", "INLAND MARINE", "AUTO", "CYBER", "WORKERS COMPENSATION"
      ],
      "new_business_only": False,
      "loss_ratio_max": 0.525,                 # 65% tighter
      "tiv_max": 350_000_000,                  # 65% tighter
      "premium_range": [0, 35_000_000],        # 65% tighter
      "min_winnability": 0.35,                 # 65% closer to 1
      "min_year": 1965,                        # 65% newer
      "good_construction_only": False
    },
    "weights": {
      "w_prem": 0.15, "w_win": 0.15, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.40,
      "premium_lo": 0, "premium_mid_lo": 35_000, "premium_mid_hi": 350_000, "premium_hi": 35_000_000,
      "tiv_hi": 350_000_000
    }
  },

  # Broad discovery; keep doors open but still rank with sensible priorities (tightened ~65%)
  "loose_fits": {
    "filters": {
      "lob_in": [
        "COMMERCIAL PROPERTY", "HABITATIONAL", "GENERAL LIABILITY", "UMBRELLA", "INLAND MARINE", "AUTO", "CYBER", "WORKERS COMPENSATION"
      ],
      "loss_ratio_max": 0.7,                   # 65% tighter
      "tiv_max": 700_000_000,                  # 65% tighter
      "premium_range": [0, 350_000_000],       # 65% tighter
      "min_winnability": 0.0,
      "min_year": 1940,                        # 65% newer
      "good_construction_only": False
    },
    "weights": {
      "w_prem": 0.10, "w_win": 0.10, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.50,
      "premium_lo": 0, "premium_mid_lo": 35_000, "premium_mid_hi": 350_000, "premium_hi": 350_000_000,
      "tiv_hi": 700_000_000
    }
  },

  # Turnaround bets (tightened ~65%)
  "turnaround_bets": {
    "filters": {
      "lob_in": [
        "COMMERCIAL PROPERTY", "HABITATIONAL", "GENERAL LIABILITY", "UMBRELLA", "INLAND MARINE", "AUTO", "CYBER", "WORKERS COMPENSATION"
      ],
      "loss_ratio_max": 0.7,                   # 65% tighter
      "tiv_max": 700_000_000,                  # 65% tighter
      "premium_range": [0, 350_000_000],       # 65% tighter
      "min_winnability": 0.0,
      "min_year": 1940,                        # 65% newer
      "good_construction_only": False
    },
    "weights": {
      "w_prem": 0.10, "w_win": 0.10, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.50,
      "premium_lo": 0, "premium_mid_lo": 35_000, "premium_mid_hi": 350_000, "premium_hi": 350_000_000,
      "tiv_hi": 700_000_000
    }
  },
}
