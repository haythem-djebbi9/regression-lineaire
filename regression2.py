import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Charger le fichier CSV dans votre drive
df = pd.read_csv('chemin/vers/votre/fichier/carprice.csv')

# Affiche le nom des attributs (features ou clés)
print("Nom des attributs:", df.columns)

# Affiche les 5 premières lignes du fichier
print("Les 5 premières lignes du fichier:\n", df.head())

# Vérifier s’il y a des attributs sans valeurs (ayant la valeur null)
print("Attributs avec des valeurs nulles:\n", df.isnull().sum())

# Afficher le heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap de corrélation')
plt.show()

# Créer le vecteur Y qui contient la colonne « selling_price »
Y = df['selling_price'].values.reshape(-1, 1)

# Créer la matrice qui contient les colonnes 1-5
X = df.iloc[:, 1:6].values
X = X.reshape(-1, 5)

# Diviser X et Y en 70% pour l’apprentissage et 30% pour le test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Créer le modèle de régression linéaire et lancer l’apprentissage
model = LinearRegression()
model.fit(x_train, y_train)

# Afficher les coefficients du modèle
print("Coefficients du modèle:", model.coef_)

# Calculer R2 du modèle appliqué aux données d’apprentissage
y_train_pred = model.predict(x_train)
r2_train = r2_score(y_train, y_train_pred)
print("R2 du modèle (apprentissage):", r2_train)

# Calculer R2 du modèle appliqué aux données de test
y_test_pred = model.predict(x_test)
r2_test = r2_score(y_test, y_test_pred)
print("R2 du modèle (test):", r2_test)

# Lancer la prédiction du vecteur x_test
predictions = model.predict(x_test)

# Effectuer l’analyse des résidus
residuals = y_test - predictions
sns.residplot(predictions.flatten(), residuals.flatten())
plt.title('Analyse des résidus')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.show()

# Conclure quant au choix du modèle de régression linéaire
# Vous pouvez interpréter les résultats en fonction des valeurs R2 et de l'analyse des résidus.
