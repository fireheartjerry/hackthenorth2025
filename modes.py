# modes.py

MODES = {
  # Ultra-selective, high-value, new-business focus
  "unicorn_hunting": {
    "filters": {
      "lob_in": ["COMMERCIAL PROPERTY"],
      "new_business_only": True,
      "loss_ratio_max": 0.40,                  # conservative LR
      "tiv_max": 100_000_000,
      "premium_range": [150_000, 1_800_000],   # larger tickets
      "min_winnability": 0.60,
      "min_year": 2010,                        # newer buildings preferred
      "good_construction_only": True
    },
    "weights": {
      # core weights (sum renormalized by scorer if you do that elsewhere)
      "w_prem": 0.35, "w_win": 0.35, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.05, "w_fresh": 0.05,
      # premium sweet-spot params
      "premium_lo": 150_000, "premium_mid_lo": 300_000, "premium_mid_hi": 1_000_000, "premium_hi": 1_800_000,
      # TIV reference for z-scaling
      "tiv_hi": 100_000_000
    }
  },

  # Realistic, portfolio-friendly balance of value and risk (now extremely inclusive)
  "balanced_growth": {
    "filters": {
      # Allow all LOBs
      "lob_in": [
        "COMMERCIAL PROPERTY", "HABITATIONAL", "GENERAL LIABILITY", "UMBRELLA", "INLAND MARINE", "AUTO", "CYBER", "WORKERS COMPENSATION"
      ],
      "new_business_only": False,
      "loss_ratio_max": 1.5,     # Extremely loose
      "tiv_max": 1_000_000_000,  # Extremely loose
      "premium_range": [0, 100_000_000],  # Extremely loose
      "min_winnability": 0.0,    # Allow all
      "min_year": 1900,          # Allow all
      "good_construction_only": False
    },
    "weights": {
      "w_prem": 0.15, "w_win": 0.15, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.40,
      "premium_lo": 0, "premium_mid_lo": 100_000, "premium_mid_hi": 1_000_000, "premium_hi": 100_000_000,
      "tiv_hi": 1_000_000_000
    }
  },

  # Broad discovery; keep doors open but still rank with sensible priorities (now most loose)
  "loose_fits": {
    "filters": {
      "lob_in": [
        "COMMERCIAL PROPERTY", "HABITATIONAL", "GENERAL LIABILITY", "UMBRELLA", "INLAND MARINE", "AUTO", "CYBER", "WORKERS COMPENSATION"
      ],
      "loss_ratio_max": 2.0,
      "tiv_max": 2_000_000_000,
      "premium_range": [0, 1_000_000_000],
      "min_winnability": 0.0,
      "min_year": 1800,
      "good_construction_only": False
    },
    "weights": {
      "w_prem": 0.10, "w_win": 0.10, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.50,
      "premium_lo": 0, "premium_mid_lo": 100_000, "premium_mid_hi": 1_000_000, "premium_hi": 1_000_000_000,
      "tiv_hi": 2_000_000_000
    }
  },

  # Turnaround bets (now most loose)
  "turnaround_bets": {
    "filters": {
      "lob_in": [
        "COMMERCIAL PROPERTY", "HABITATIONAL", "GENERAL LIABILITY", "UMBRELLA", "INLAND MARINE", "AUTO", "CYBER", "WORKERS COMPENSATION"
      ],
      "loss_ratio_max": 2.0,
      "tiv_max": 2_000_000_000,
      "premium_range": [0, 1_000_000_000],
      "min_winnability": 0.0,
      "min_year": 1800,
      "good_construction_only": False
    },
    "weights": {
      "w_prem": 0.10, "w_win": 0.10, "w_year": 0.10, "w_con": 0.10, "w_tiv": 0.10, "w_fresh": 0.50,
      "premium_lo": 0, "premium_mid_lo": 100_000, "premium_mid_hi": 1_000_000, "premium_hi": 1_000_000_000,
      "tiv_hi": 2_000_000_000
    }
  },
}
