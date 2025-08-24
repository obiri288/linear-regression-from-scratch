import numpy as np
import matplotlib.pyplot as plt


X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17], dtype=np.float32)


plt.figure(figsize=(8, 6))
plt.scatter(X, y)
plt.title("Unsere Beispieldaten")
plt.xlabel("Wohnfläche")
plt.ylabel("Preis")
plt.show()

# Initialisierung der Parameter
# m -> weight
# b -> bias
weight = 0.0
bias = 0.0

# Hyperparameter definieren
learning_rate = 0.01
n_iterations = 1000 # Anzahl der Trainingsdurchläufe

n_samples = len(X)

# Der Trainings-Loop
for epoch in range(n_iterations):
    # 1. Vorhersage berechnen
    y_predicted = weight * X + bias

    # 2. Fehler (Loss) berechnen (optional, aber gut zur Kontrolle)
    loss = np.mean((y_predicted - y)**2)

    # 3. Gradienten berechnen
    dw = (1/n_samples) * np.sum(2 * X * (y_predicted - y)) # Ableitung nach weight
    db = (1/n_samples) * np.sum(2 * (y_predicted - y))     # Ableitung nach bias

    # 4. Parameter aktualisieren
    weight -= learning_rate * dw
    bias -= learning_rate * db

    # Alle 100 Epochen den Fortschritt ausgeben
    if (epoch + 1) % 100 == 0:
        print(f'Epoche {epoch+1}: Gewicht = {weight:.3f}, Bias = {bias:.3f}, Fehler = {loss:.4f}')

print(f'\nTraining abgeschlossen! Finale Parameter:')
print(f'Gewicht (m): {weight:.3f}')
print(f'Bias (b): {bias:.3f}')

n_samples = len(X)

# Der Trainings-Loop
for epoch in range(n_iterations):
    # 1. Vorhersage berechnen
    y_predicted = weight * X + bias

    # 2. Fehler (Loss) berechnen (optional, aber gut zur Kontrolle)
    loss = np.mean((y_predicted - y)**2)

    # 3. Gradienten berechnen
    dw = (1/n_samples) * np.sum(2 * X * (y_predicted - y)) # Ableitung nach weight
    db = (1/n_samples) * np.sum(2 * (y_predicted - y))     # Ableitung nach bias

    # 4. Parameter aktualisieren
    weight -= learning_rate * dw
    bias -= learning_rate * db

    # Alle 100 Epochen den Fortschritt ausgeben
    if (epoch + 1) % 100 == 0:
        print(f'Epoche {epoch+1}: Gewicht = {weight:.3f}, Bias = {bias:.3f}, Fehler = {loss:.4f}')

print(f'\nTraining abgeschlossen! Finale Parameter:')
print(f'Gewicht (m): {weight:.3f}')
print(f'Bias (b): {bias:.3f}')

# Die finale Regressionslinie berechnen
predicted_line = weight * X + bias

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Originaldaten')
plt.plot(X, predicted_line, color='red', label='Regressionslinie')
plt.title("Lineare Regression von Grund auf")
plt.xlabel("Wohnfläche")
plt.ylabel("Preis")
plt.legend()
plt.show()

# Testvorhersage für eine neue Wohnfläche, z.B. 900 qm (X=9)
neue_wohnflaeche = 9
vorhergesagter_preis = weight * neue_wohnflaeche + bias
print(f'\nFür eine Wohnfläche von {neue_wohnflaeche} wird ein Preis von {vorhergesagter_preis:.3f} vorhergesagt.')

# Initialisierung der Parameter
weight = 0.0
bias = 0.0

# Hyperparameter definieren
learning_rate = 0.01
n_iterations = 1000 # Anzahl der Trainingsdurchläufe
n_samples = len(X)

# Der Trainings-Loop
for epoch in range(n_iterations):
    # 1. Vorhersage mit aktuellen Parametern berechnen
    y_predicted = weight * X + bias

    # 2. Fehler (Loss) berechnen
    loss = np.mean((y_predicted - y)**2)

    # 3. Gradienten berechnen (die Richtung der steilsten "Abfahrt")
    dw = (1/n_samples) * np.sum(2 * X * (y_predicted - y)) # Ableitung nach weight
    db = (1/n_samples) * np.sum(2 * (y_predicted - y))     # Ableitung nach bias

    # 4. Parameter aktualisieren (den Schritt "bergab" machen)
    weight -= learning_rate * dw
    bias -= learning_rate * db

    # Alle 100 Epochen den Fortschritt ausgeben
    if (epoch + 1) % 100 == 0:
        print(f'Epoche {epoch+1}: Gewicht = {weight:.3f}, Bias = {bias:.3f}, Fehler = {loss:.4f}')

print(f'\nTraining abgeschlossen! Finale Parameter:')
print(f'Gewicht (m): {weight:.3f}')
print(f'Bias (b): {bias:.3f}')

# Die finale Regressionslinie mit den gelernten Parametern berechnen
predicted_line = weight * X + bias

# --- Visualisierung ---
# Den ursprünglichen Plot-Code hier einfügen und anpassen

plt.figure(figsize=(8, 6))
# Die Originaldaten als Punkte plotten
plt.scatter(X, y, label='Originaldaten')
# Die gelernte Linie plotten
plt.plot(X, predicted_line, color='red', label='Regressionslinie')
plt.title("Lineare Regression von Grund auf")
plt.xlabel("Wohnfläche")
plt.ylabel("Preis")
plt.legend()
plt.show() # Zeigt das finale Diagramm an