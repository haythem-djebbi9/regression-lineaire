import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importer le fichier CSV et créer le dataframe
df = pd.read_csv('Regression-AreaPrice.csv')

# Afficher les 5 premières lignes du dataframe
print(df.head())

# Créer un vecteur X contenant la première colonne (entrée du modèle)
X = df['Surface'].values.reshape(-1, 1)

# Créer un vecteur Y contenant la deuxième colonne (sortie du modèle)
y = df['Price'].values

# Tracer le nuage de points du prix en fonction de la surface
plt.scatter(X, y)
plt.title('Scatter Plot of Price vs Surface')
plt.xlabel('Surface')
plt.ylabel('Price')
plt.show()

# Calculer le coefficient de corrélation
correlation = df.corr()
print(correlation)

# Créer un modèle de régression linéaire et lancer l’apprentissage
model = LinearRegression()
model.fit(X, y)

# Afficher la taille de X
print("Taille de X avant reshape:", X.shape)

# Modifier la taille de X en appelant la fonction reshape (-1,1)
X = X.reshape(-1, 1)

# Afficher la taille de X après reshape
print("Taille de X après reshape:", X.shape)

# Lancer la prédiction pour une surface 400
prediction_400 = model.predict([[400]])
print("Prédiction pour Surface 400:", prediction_400)

# Lancer la prédiction pour tout le vecteur X
predictions = model.predict(X)

# Calculer le score R2
r2 = r2_score(y, predictions)
print("Score R2:", r2)

# Analyse des résidus
residuals = y - predictions
sns.residplot(predictions, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
