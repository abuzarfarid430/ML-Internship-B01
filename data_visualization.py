import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("visualizations", exist_ok=True)
# Load cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")

sns.set(style="whitegrid")

# 1. Line Plot - Age Distribution
plt.figure()
df['Age'].sort_values().reset_index(drop=True).plot()
plt.title("Age Distribution (Line Plot)")
plt.xlabel("Passenger Index")
plt.ylabel("Age")
plt.savefig("visualizations/line_age.png")
plt.close()

# 2. Scatter Plot - Age vs Fare
plt.figure()
plt.scatter(df['Age'], df['Fare'])
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.savefig("visualizations/scatter_age_fare.png")
plt.close()

# 3. Histogram - Passenger Class
plt.figure()
plt.hist(df['Pclass'], bins=3)
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.savefig("visualizations/histogram_pclass.png")
plt.close()

# 4. Bar Chart - Survival Rates
plt.figure()
df.groupby('Survived')['Survived'].count().plot(kind='bar')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.savefig("visualizations/bar_survival.png")
plt.close()

# 5. Box Plot - Fare by Passenger Class
plt.figure()
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare Distribution by Class")
plt.savefig("visualizations/box_fare_class.png")
plt.close()

# 6. Violin Plot - Age by Gender
plt.figure()
sns.violinplot(x='Sex', y='Age', data=df)
plt.title("Age Distribution by Gender")
plt.savefig("visualizations/violin_age_gender.png")
plt.close()

# 7. Heatmap - Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("visualizations/heatmap_corr.png")
plt.close()

# 8. Pair Plot - Numerical Features
pairplot = sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']])
pairplot.savefig("visualizations/pairplot.png")
plt.close()

print("All visualizations created and saved successfully!")
