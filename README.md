# Week 2 Task: Advanced Data Visualization and Storytelling with Python

## Overview
This repository contains my Week 2 submission for the Data Science internship. The task builds on Week 1's cleaned Titanic dataset to create a complete **visual narrative** aimed at a non-technical audience — explaining not just *what* happened aboard the Titanic, but *who* was most affected and *why*, through a sequence of six connected visualizations.

## Dataset
**Titanic Passenger Dataset** (same cleaned dataset from Week 1 — 780 records, 0 missing values), with additional derived fields for this task: age groups, family size, and human-readable labels for class and embarkation port.

## Tools & Libraries
- Python 3
- Pandas — data preparation and derived-feature engineering
- Seaborn — statistical visualizations (violin plot, heatmap)
- Matplotlib — custom chart composition, annotation, and styling

## The Data Story
The visualizations are sequenced deliberately, each building on the last:

1. **Overall Survival (Donut Chart)** — the headline number: only 41.3% of passengers survived.
2. **Gender & Class Divide (Diverging Bar Chart)** — survival rate by sex within each class; reveals a 6x gap between 1st-class women (96.8%) and 3rd-class men (15.9%).
3. **Age Profiles (Split Violin Plot)** — age distribution by class and outcome, showing younger 3rd-class passengers and a survivor skew toward youth.
4. **Fare vs. Age (Bubble Scatter)** — fare, age, and family size together; survivors' median fare ($26.25) was double that of non-survivors ($13.00).
5. **Embarkation Port (Stacked Bar Chart)** — Cherbourg passengers survived at 58% vs. 34–37% elsewhere, reflecting a wealthier passenger mix rather than the port itself.
6. **Age × Class (Heatmap)** — the full picture: 2nd-class children survived at 100%, while 3rd-class adults (36–60) survived at just 8.8%.

## Key Insights
- Survival was **not random** — it was driven predictably by sex, age, and socioeconomic status (class/fare).
- The largest single gap in the dataset is between 2nd-class children (100% survival) and 3rd-class adults aged 36–60 (8.8% survival).
- Even after accounting for the "women and children first" policy, class-based access to lifeboats explains much of the remaining gap.

## Real-World Implications
- **Business (Risk Modelling/Insurance):** the same segmentation approach used here mirrors how insurers build actuarial models to identify disproportionate risk across customer segments.
- **Scientific/Policy (Emergency Evacuation Design):** the class-based survival gap is still referenced in maritime and public-safety research when designing evacuation procedures that guarantee equitable access regardless of a passenger's location aboard a vessel.

## Files in this Repository
| File | Description |
|---|---|
| `story_analysis.py` | Full Python script: data prep and all 6 visualizations |
| `titanic_story_data.csv` | Dataset with derived storytelling fields |
| `story1_overall_survival_donut.png` | Overall survival donut chart |
| `story2_gender_class_survival.png` | Survival by sex and class |
| `story3_age_violin.png` | Age distribution by class and outcome |
| `story4_fare_age_bubble.png` | Fare vs. age bubble chart |
| `story5_embarkation_stacked.png` | Survival by embarkation port |
| `story6_age_class_heatmap.png` | Survival by age group and class heatmap |

## How to Run
```bash
pip install pandas seaborn matplotlib
python3 story_analysis.py
```
This regenerates the derived dataset and all six visualizations.
