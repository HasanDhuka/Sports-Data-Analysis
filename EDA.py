import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from math import pi

# Load the dataset
df = pd.read_csv("cricket_eda_fake_data.csv")

# Set Seaborn style
sns.set_style("whitegrid")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Top 5 Players Radar Chart
top_players = df.groupby("Player")[["Runs_Scored", "Strike_Rate", "Wickets_Taken", "Economy_Rate"]].mean().nlargest(5, "Runs_Scored")
labels = top_players.index.tolist()
categories = ["Runs_Scored", "Strike_Rate", "Wickets_Taken", "Economy_Rate"]
n_vars = len(categories)
angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
angles += angles[:1]
ax = plt.subplot(2, 3, 1, polar=True)
for player in labels:
    values = top_players.loc[player].tolist()
    values += values[:1]
    ax.plot(angles, values, label=player)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title("Top 5 Players Performance")
ax.legend(loc="upper right")

# Plot 2: Game Predictions Heatmap
prediction_accuracy = pd.crosstab(df["Predicted_Winner"], df["Actual_Winner"], normalize="index")
sns.heatmap(prediction_accuracy, annot=True, cmap="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title("Game Predictions Accuracy")

# Plot 3: Fan Engagement Network Graph
G = nx.Graph()
for platform in df["Social_Platform"].unique():
    G.add_node(platform)
    for metric in ["Likes", "Shares", "Comments"]:
        G.add_node(metric)
        G.add_edge(platform, metric, weight=df[metric].mean())
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, edge_color="gray", ax=axes[0, 2])
axes[0, 2].set_title("Fan Engagement Network")

# Plot 4: Sentiment Trend Over Matches
rolling_sentiment = df[["Sentiment_Score"]].rolling(window=10).mean()
axes[1, 0].plot(rolling_sentiment, color="purple")
axes[1, 0].set_title("Sentiment Score Trend")
axes[1, 0].set_ylabel("Sentiment Score")

# Plot 5: Violin Plot for Runs Scored
sns.violinplot(x=df["Match_Type"], y=df["Runs_Scored"], palette="muted", ax=axes[1, 1])
axes[1, 1].set_title("Runs Distribution by Match Type")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()