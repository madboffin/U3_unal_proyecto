# Mostrar los resultados detallados de cada modelo
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f} (+/- {result['cv_std_accuracy']:.4f})")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, models[model_name].predict(X_test)))

# Mostrar el mejor modelo
best_model = max(results.items(), key=lambda item: item[1]['test_accuracy'])
print(f"\nBest Model: {best_model[0]} with Test Accuracy: {best_model[1]['test_accuracy']:.4f}")

# Mostrar el informe de clasificación del mejor modelo
print("\nBest Model Classification Report:")
print(classification_report(y_test, models[best_model[0]].predict(X_test)))
