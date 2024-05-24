# Diccionario de datos

Diccionario de Datos

Tipo: String
Descripción: El texto del comentario en línea.
toxicity

Tipo: Float
Descripción: Probabilidad de que el comentario sea tóxico. Valor continuo entre 0 y 1.
severe_toxicity

Tipo: Float
Descripción: Probabilidad de que el comentario sea severamente tóxico. Valor continuo entre 0 y 1.
obscene

Tipo: Float
Descripción: Probabilidad de que el comentario sea obsceno. Valor continuo entre 0 y 1.
threat

Tipo: Float
Descripción: Probabilidad de que el comentario contenga amenazas. Valor continuo entre 0 y 1.
insult

Tipo: Float
Descripción: Probabilidad de que el comentario contenga insultos. Valor continuo entre 0 y 1.
identity_attack

Tipo: Float
Descripción: Probabilidad de que el comentario sea un ataque a la identidad de una persona o grupo. Valor continuo entre 0 y 1.
sexual_explicit

Tipo: Float
Descripción: Probabilidad de que el comentario sea sexualmente explícito. Valor continuo entre 0 y 1.
Variables de Sesgo de Identidad
Estas variables indican la probabilidad de que un comentario mencione o ataque a un grupo basado en la identidad. Todos son valores continuos entre 0 y 1.

male

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a hombres de forma sesgada.
female

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a mujeres de forma sesgada.
transgender

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas transgénero de forma sesgada.
other_gender

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a otras identidades de género de forma sesgada.
heterosexual

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas heterosexuales de forma sesgada.
homosexual_gay_or_lesbian

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas homosexuales (gays o lesbianas) de forma sesgada.
bisexual

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas bisexuales de forma sesgada.
other_sexual_orientation

Tipo: Float
Descripción: Probabilidad de que el comentario mencione otras orientaciones sexuales de forma sesgada.
christian

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas cristianas de forma sesgada.
jewish

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas judías de forma sesgada.
muslim

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas musulmanas de forma sesgada.
hindu

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas hindúes de forma sesgada.
buddhist

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas budistas de forma sesgada.
atheist

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas ateas de forma sesgada.
other_religion

Tipo: Float
Descripción: Probabilidad de que el comentario mencione otras religiones de forma sesgada.
black

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas negras de forma sesgada.
white

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas blancas de forma sesgada.
asian

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas asiáticas de forma sesgada.
latino

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas latinas de forma sesgada.
other_race_or_ethnicity

Tipo: Float
Descripción: Probabilidad de que el comentario mencione otras razas o etnias de forma sesgada.
physical_disability

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas con discapacidades físicas de forma sesgada.
intellectual_or_learning_disability

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas con discapacidades intelectuales o de aprendizaje de forma sesgada.
psychiatric_or_mental_illness

Tipo: Float
Descripción: Probabilidad de que el comentario mencione a personas con enfermedades psiquiátricas o mentales de forma sesgada.
other_disability

Tipo: Float
Descripción: Probabilidad de que el comentario mencione otras discapacidades de forma sesgada.
Variables Demográficas de los Anotadores
Además de las variables mencionadas, el conjunto de datos también puede contener información demográfica sobre los anotadores que etiquetaron los datos. Esto puede incluir:

anotator_gender

Tipo: String
Descripción: Género del anotador (masculino, femenino, no binario, etc.).
anotator_age

Tipo: Integer
Descripción: Edad del anotador.
anotator_race_or_ethnicity

Tipo: String
Descripción: Raza o etnia del anotador.



