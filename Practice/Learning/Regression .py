import pandas as pd
import matplotlib.pyplot as plt

hamza = pd.read_csv("No. of Cases of TCC from 2003-2023.txt")

ax = plt.gca()

# Plot the data
hamza.plot(kind='line', x='Year', y='Cases', color='green', ax=ax)

# Set x-axis ticks to show every year
ax.set_xticks(hamza['Year'])
ax.set_xticklabels(hamza['Year'], rotation=45, ha='right')  # Rotate labels for better readability

# Annotate each data point with a circle around the label
for index, row in hamza.iterrows():
    label = str(row['Cases'])
    ax.annotate(label, (row['Year'], row['Cases']), textcoords="offset points", xytext=(1.5, 8), ha='center')


plt.title("Number of cases per year")
plt.xlabel("Year")
plt.ylabel("Number of Cases")

plt.show()
