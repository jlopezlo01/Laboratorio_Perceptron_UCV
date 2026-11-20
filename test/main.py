from src.perceptron import Perceptron
from src.dataset import training_data

# Crear el modelo
model = Perceptron()

# Entrenar el modelo
model.train(training_data)

# Realizar una predicción
prediction = model.predict([9, 1])

# Mostrar el resultado
print("Resultado de la predicción:", prediction)

if prediction == 1:
    print("Crédito aprobado")
else:
    print("Crédito rechazado")
