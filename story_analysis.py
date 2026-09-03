"""
Week 2 Task: Advanced Data Visualization and Storytelling with Python
Dataset: Titanic passenger dataset (continued from Week 1's cleaned data)
Goal: Build a visual narrative for a NON-technical audience explaining
who survived the Titanic disaster and why.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Survived": "#2E86AB", "Did not survive": "#C1440E"}
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13

# ---------------------------------------------------------------
# LOAD CLEANED DATA (reuse Week 1's cleaning logic)
# ---------------------------------------------------------------
df = sns.load_dataset("titanic")
df = df.drop(columns=["deck", "embark_town", "alive"]).drop_duplicates()
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["survived_label"] = df["survived"].map({1: "Survived", 0: "Did not survive"})
df["class_label"] = df["pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
df["port_label"] = df["embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["age_group"] = pd.cut(
    df["age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["Child (0-12)", "Teen (13-18)", "Young Adult (19-35)", "Adult (36-60)", "Senior (60+)"]
)

df.to_csv("titanic_story_data.csv", index=False)
print("Data ready:", df.shape)

# =================================================================
# VISUAL 1: The Big Picture — Overall survival (donut chart)
# =================================================================
fig, ax = plt.subplots(figsize=(6, 6))
counts = df["survived_label"].value_counts()
colors = [PALETTE[c] for c in counts.index]
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
    colors=colors, pctdistance=0.8, wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 12}
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
ax.set_title("Only 4 in 10 Passengers Survived the Titanic Disaster", fontsize=14, pad=20)
ax.text(0, 0, f"{len(df)}\npassengers", ha="center", va="center", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("story1_overall_survival_donut.png", bbox_inches="tight")
plt.close()

# =================================================================
# VISUAL 2: The Gender Divide (diverging bar chart)
# =================================================================
gender_class = df.groupby(["class_label", "sex"])["survived"].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(7, 4.5))
y_pos = np.arange(len(gender_class.index))
ax.barh(y_pos, gender_class["female"], color="#E07A5F", label="Female", height=0.35, align="edge")
ax.barh(y_pos - 0.35, gender_class["male"], color="#3D5A80", label="Male", height=0.35, align="edge")
ax.set_yticks(y_pos - 0.175)
ax.set_yticklabels(gender_class.index)
ax.set_xlabel("Survival Rate (%)")
ax.set_title('"Women and Children First": Survival Rate by Sex and Class', fontsize=13)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="lower right", frameon=True)
for i, cls in enumerate(gender_class.index):
    ax.text(gender_class.loc[cls, "female"] + 1, i, f"{gender_class.loc[cls,'female']:.0f}%", va="center", fontsize=9)
    ax.text(gender_class.loc[cls, "male"] + 1, i - 0.35, f"{gender_class.loc[cls,'male']:.0f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("story2_gender_class_survival.png", bbox_inches="tight")
plt.close()

# =================================================================
# VISUAL 3: Who Was Aboard — Age distribution by outcome (violin)
# =================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.violinplot(
    data=df, x="class_label", y="age", hue="survived_label", split=True,
    palette=PALETTE, ax=ax, inner="quartile", order=["1st Class", "2nd Class", "3rd Class"]
)
ax.set_title("Age Profiles Tell a Story: Younger 3rd-Class Passengers, Wider Spread in 1st", fontsize=12.5)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age (years)")
ax.legend(title="Outcome", loc="upper right")
plt.tight_layout()
plt.savefig("story3_age_violin.png", bbox_inches="tight")
plt.close()

# =================================================================
# VISUAL 4: Money Mattered — Fare vs Age scatter, sized by family, colored by outcome
# =================================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
for label, color in PALETTE.items():
    sub = df[df["survived_label"] == label]
    ax.scatter(
        sub["age"], sub["fare"], s=sub["family_size"] * 25, c=color, alpha=0.55,
        edgecolors="white", linewidth=0.5, label=label
    )
ax.set_yscale("symlog")
ax.set_title("Higher Fares Bought Better Odds — Bubble Size Shows Family Size", fontsize=13)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Fare Paid (log scale)")
legend1 = ax.legend(title="Outcome", loc="upper right")
ax.add_artist(legend1)
plt.tight_layout()
plt.savefig("story4_fare_age_bubble.png", bbox_inches="tight")
plt.close()

# =================================================================
# VISUAL 5: The Journey's Start — Survival by embarkation port (stacked bar)
# =================================================================
port_ct = pd.crosstab(df["port_label"], df["survived_label"], normalize="index") * 100
port_counts = df["port_label"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4.5))
port_ct = port_ct.loc[port_counts.index]
bottom = np.zeros(len(port_ct))
for label in ["Did not survive", "Survived"]:
    ax.bar(port_ct.index, port_ct[label], bottom=bottom, color=PALETTE[label], label=label, width=0.55)
    for i, v in enumerate(port_ct[label]):
        if v > 5:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    bottom += port_ct[label].values
for i, port in enumerate(port_ct.index):
    ax.text(i, 103, f"n={port_counts[port]}", ha="center", fontsize=9, color="#555555")
ax.set_ylim(0, 112)
ax.set_title("Cherbourg Passengers Fared Better — Likely Reflecting Higher-Class Mix", fontsize=12.5)
ax.set_ylabel("Share of Passengers (%)")
ax.set_xlabel("Port of Embarkation")
ax.legend(loc="lower right", bbox_to_anchor=(1, -0.02))
plt.tight_layout()
plt.savefig("story5_embarkation_stacked.png", bbox_inches="tight")
plt.close()

# =================================================================
# VISUAL 6: Putting It Together — Heatmap of survival by class x age group
# =================================================================
pivot = df.pivot_table(index="age_group", columns="class_label", values="survived", aggfunc="mean", observed=False) * 100
pivot = pivot[["1st Class", "2nd Class", "3rd Class"]]
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlBu", center=50, ax=ax,
            cbar_kws={"label": "Survival Rate (%)"}, linewidths=1, linecolor="white")
ax.set_title("The Full Picture: Survival Rate by Age Group and Class", fontsize=13)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age Group")
plt.tight_layout()
plt.savefig("story6_age_class_heatmap.png", bbox_inches="tight")
plt.close()

print("All 6 storytelling visualizations saved.")

# Print key numbers referenced in the narrative
print("\nOverall survival:", (df["survived"].mean() * 100).round(1))
print("\nSurvival by sex & class:\n", gender_class.round(1))
print("\nSurvival by port:\n", df.groupby("port_label")["survived"].mean().round(3))
print("\nSurvival by age group & class:\n", pivot.round(1))
print("\nMedian fare survived vs not:\n", df.groupby("survived_label")["fare"].median())
