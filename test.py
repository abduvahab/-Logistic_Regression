import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target

# Map target values to species names
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
df['Species'] = df['Species'].map(species_map)

# Create a pair plot
sns.pairplot(df, hue="Species", diag_kind="kde")
plt.show()