# Appetizer-IQ Priority Scoring System Documentation

## Overview

The Appetizer-IQ priority scoring system evaluates insurance submissions across multiple dimensions to provide consistent appetite classification and priority ranking throughout the platform.

## Scoring Components

### 1. Appetite Status Classification

**TARGET** - Premium submissions meeting optimal criteria:
- Commercial Property line of business
- Target states (OH, PA, MD, CO, CA, FL)
- Appetite score ≥ 7.0
- Priority score ≥ 7.0
- No severe issues identified

**IN** - Acceptable submissions meeting core requirements:
- Commercial Property line of business
- Acceptable or target states
- Appetite score ≥ 3.0
- Fewer than 3 severe issues

**OUT** - Submissions failing critical requirements:
- Non-Commercial Property LOB (automatic OUT)
- Appetite score < 3.0
- 3+ severe issues identified

### 2. Priority Score Calculation (0.0 - 10.0)

Priority score combines multiple factors:
- **Appetite Alignment (40%)**: How well submission fits underwriting guidelines
- **Winnability (30%)**: Probability of winning the business
- **Premium Size (20%)**: Revenue potential and policy size
- **Freshness (10%)**: Recency and urgency factors

## Scoring Factors Detail

### States
- **Target States** (+2.0 points, 1.2x multiplier): OH, PA, MD, CO, CA, FL
- **Acceptable States** (+1.0 points): NC, SC, GA, VA, UT  
- **Non-preferred States** (-1.5 points): All others

### Total Insured Value (TIV)
- **Target Range** (50M-100M): +2.0 points, 1.15x multiplier
- **Acceptable Range** (≤150M): +1.0 points
- **Over Limit** (>150M): -2.0 points
- **Missing Data**: -0.5 points

### Premium
- **Target Range** (75K-100K): +2.0 points, 1.3x multiplier
- **Acceptable Range** (50K-175K): +1.0 points, 1.1x multiplier
- **High Value** (≥1M): +1.5 points, additional 1.2x multiplier
- **Substantial** (≥500K): +0.5 points
- **Below Minimum** (<50K): -1.0 points
- **Above Preferred** (>175K): -1.0 points

### Building Age
- **Modern** (Post-2010): +2.0 points, 1.1x multiplier
- **Acceptable** (1990-2010): +1.0 points
- **Older** (Pre-1990): -1.5 points
- **Unknown**: -0.5 points

### Construction Type
- **Preferred Types**: +1.0 points
  - Joisted Masonry, Non-Combustible, Masonry Non-Combustible, Fire Resistive
- **Other Types**: -0.5 points

### Loss History
- **Excellent** (<25K): +1.5 points, 1.1x multiplier
- **Acceptable** (<100K): +0.5 points
- **Concerning** (≥100K): -2.0 points
- **Unknown**: Neutral (0 points)

### Winnability
- **Very High** (≥80%): +1.5 points, 1.2x multiplier
- **High** (≥60%): +1.0 points
- **Moderate** (≥40%): +0.5 points
- **Low** (<40%): -0.5 points
- **Unknown**: Assumed 50%

### Business Type
- **New Business**: +1.0 points (preferred)
- **Renewal**: -2.0 points (allowed but penalized)

## Critical Requirements

Some criteria result in automatic OUT status:
- Non-Commercial Property line of business
- Multiple critical failures in core requirements

## Severe Issues

Factors that count as "severe issues" for status determination:
- TIV exceeds limits
- Premium below minimum or above preferred maximum
- Concerning loss history
- Non-preferred state location
- Multiple deficiencies

## Usage Across Platform

The scoring system is used consistently across:
- **Dashboard Home**: Priority worklist ranking
- **Submissions Page**: Table sorting and filtering
- **Priority Accounts**: High-priority submission identification
- **Detail Pages**: Individual submission explanations
- **API Endpoints**: Consistent data delivery

## Implementation Notes

- Scores are calculated in real-time during data processing
- All submissions receive both appetite_score and priority_score
- Missing data is handled gracefully with reasonable defaults
- Multipliers are applied after base scoring for enhanced differentiation
- Final scores are bounded to prevent extreme outliers

## Debugging

Debug information is logged to `scores_debug.txt` including:
- Submission ID
- Final status determination  
- Appetite score
- Priority score
- Applied reasons/factors

This ensures transparency and allows for scoring system refinement based on actual data patterns.
