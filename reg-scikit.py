import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Charger le dataset
house_data = pd.read_csv('house.csv')

# Afficher le nuage de points
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.xlabel('Surface')
plt.ylabel('Loyer')

# Définir les variables indépendantes (X) et la variable cible (y)
X = house_data[['surface']]
y = house_data['loyer']

# Créer un modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle
model.fit(X, y)

# Obtenir les coefficients du modèle
theta0 = model.intercept_
theta1 = model.coef_[0]

print("Coefficient theta0 (intercept) :", theta0)
print("Coefficient theta1 (pente) :", theta1)

# Tracer la droite de régression
plt.plot([0, 250], [theta0, theta0 + 250 * theta1], linestyle='--', c='#000000')

plt.show()
