# Load data
df = load_data(download_data=True)

# Preprocess data
to_numeric_cols = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
df = preprocess_df(df, to_numeric_cols)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Convert text to spaCy docs
corpus = df['comment_text'].tolist()
docs = to_spacy(corpus, nlp)

# Preprocess texts
df['processed_text'] = [preprocess_text(doc, lemma=True) for doc in docs]

print("Preprocesamiento completo.")


#Modelamiento
"""Ahora, entrenamos varios modelos de machine learning para establecer una línea base y 
comparar sus rendimientos."""

#1. Extracción de Características
"""Utilizaremos técnicas de preprocesamiento de texto, 
como tokenización, eliminación de stop words, lematización y vectorización utilizando TF-IDF."""
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(corpus: Iterable[str], max_features: int = 5000) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    df_features = pd.DataFrame(X.toarray(), columns=feature_names)
    return df_features

# Extraer características
corpus = df['processed_text'].tolist()
df_features = extract_features(corpus)

# Mostrar las primeras filas del DataFrame de características
print(df_features.head())


#2 Modelamiento
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Test Accuracy: {accuracy:.4f}")
    
    # Almacenar los resultados
    results[name] = {
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "test_accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

# Mostrar el mejor modelo
best_model = max(results.items(), key=lambda item: item[1]['test_accuracy'])
print(f"Best Model: {best_model[0]} with Test Accuracy: {best_model[1]['test_accuracy']:.4f}")

# Mostrar el informe de clasificación del mejor modelo
print(classification_report(y_test, models[best_model[0]].predict(X_test)))
