from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# Supongamos que queremos predecir la columna 'toxicity'
target_col = 'toxicity'

# Convertir la columna de destino a numérica
df[target_col] = pd.to_numeric(df[target_col])

# Convertir 'toxicity' en una variable binaria
df['toxicity_binary'] = df[target_col].apply(lambda x: 1 if x >= 0.5 else 0)

# Extraer las características y la variable objetivo
X = df_features
y = df['toxicity_binary']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los modelos a comparar
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

# Evaluar cada modelo usando validación cruzada y luego en el conjunto de prueba
results = {}
for name, model in models.items():
    # Pipeline para escalar y entrenar el modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # with_mean=False por sparse matrix de TF-IDF
        ('classifier', model)
    ])
    
    # Validación cruzada
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"{name} - Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Entrenar y predecir
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{name} - Test Accuracy: {accuracy:.4f}")
    print(f"{name} - ROC AUC: {roc_auc:.4f}")
    
    # Almacenar los resultados
    results[name] = {
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "test_accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

# Mostrar el mejor modelo
best_model = max(results.items(), key=lambda item: item[1]['test_accuracy'])
print(f"\nBest Model: {best_model[0]} with Test Accuracy: {best_model[1]['test_accuracy']:.4f}")

# Generar el informe de clasificación del mejor modelo
best_model_report = classification_report(y_test, models[best_model[0]].predict(X_test))
print("\nBest Model Classification Report:")
print(best_model_report)

# Generar la curva ROC del mejor modelo
fpr, tpr, _ = roc_curve(y_test, models[best_model[0]].predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_model[1]["roc_auc"]:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic - {best_model[0]}')
plt.legend(loc="lower right")
plt.savefig('best_model_roc_curve.png')
plt.show()

# Guardar el reporte y la matriz de confusión en un archivo JSON
report_file = "best_model_report.json"
with open(report_file, "w") as f:
    json.dump(best_model[1], f, indent=4)

print(f"\nEl reporte del mejor modelo se ha guardado en {report_file}")
print(f"La curva ROC del mejor modelo se ha guardado como 'best_model_roc_curve.png'")
