import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

def sigmoid(x): #testowaliśmy też Relu i Tan
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  #pochodna sigmoidu w celu propagacji wstecznej
    return x * (1 - x) 

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
        self.input_hidden_weights_history = []  
        self.hidden_output_weights_history = []  
        self.input_hidden_errors = []  
        self.hidden_output_errors = []  
        self.blad_sredniokwadratowy = []
        self.training_errors = []

    def train(self, data, learning_rate, epochs):
        inputs = data[:, :-1]
        targets = data[:, -1:]
        for epoch in range(epochs):
            hidden_inputs = np.dot(inputs, self.weights_input_hidden)
            hidden_outputs = sigmoid(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
            final_outputs = sigmoid(final_inputs)

            output_error = targets - final_outputs
            hidden_error = np.dot(output_error, self.weights_hidden_output.T)

            output_delta = output_error * sigmoid_derivative(final_outputs)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_outputs)

            self.weights_hidden_output += learning_rate * np.dot(hidden_outputs.T, output_delta)
            self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)

            self.input_hidden_weights_history.append(np.copy(self.weights_input_hidden))
            self.hidden_output_weights_history.append(np.copy(self.weights_hidden_output))

            input_hidden_error = np.mean(np.square(hidden_error))
            hidden_output_error = np.mean(np.square(output_error))
            
            self.input_hidden_errors.append(input_hidden_error)
            self.hidden_output_errors.append(hidden_output_error)
            
            misclassified = np.sum(np.abs(final_outputs - targets))
            self.training_errors.append(misclassified)

        
        

    # Metoda przewidująca wynik na podstawie danych wejściowych
    def predict(self, inputs):
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = sigmoid(final_inputs)
        
        #final_outputs[final_outputs <= 0.5] = -1
        #final_outputs[final_outputs > 0.5] = 1
        return final_outputs

# Przykładowe dane treningowe dla problemu XOR
data = np.array([[0, 0, 0],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]])

perceptron = Perceptron(input_size=2, hidden_size=4, output_size=1) # uruchomienie perceptronu, a ilośc warstw wynosi 3
learning_rate = 0.1 # określa jak szybko model ma sie uczyć, w wypadku naszego kodu możemy tego nie wykorzystywać
epoki = 5000 #ilość wykorzystywanych epok czyli ile razy będzie przechodził przez algorytm uczenia

perceptron.train(data, learning_rate, epoki) # Trenowanie perceptronu

# Wizualizacja
fig = plt.figure(figsize=(10, 8))
#ax1 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Dane treningowe
predictions = perceptron.predict(data[:, :2])  # Przewidywanie wartości dla danych wejściowych
#print(predictions)
for i, point in enumerate(data[:, :2]):
    color = 'green' if predictions[i] > 0.5 else 'red'  # Kolorowanie na zielono dla wartości przewidywanych jako TRUE, na czerwono jako FALSE
    ax1.scatter(point[0], point[1], c=color)


xx, yy = np.meshgrid(np.arange(0, 1, 0.01)- 0.01, np.arange(0, 1, 0.01) + 0.01) #generowanie jedowymiarowej tablicy od 0 do 1 ze skokiem 0,01 do wysłania
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]) #spłaszczenie dwuwymiarowych macierzy na 1 i łączenie ich
#print(Z)
Z = Z.reshape(xx.shape)
#Z = perceptron.predict(np.c_[xx.flatten(), yy.flatten()]).reshape(xx.shape)

ax2.plot_surface(xx, yy, Z, cmap='magma')

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for i in range(perceptron.weights_input_hidden.shape[0]):
    axs[0].plot(range(epoki), [weights[i] for weights in perceptron.input_hidden_weights_history])
    
    axs[0].set_title('Wagi między warstwą wejściową a ukrytą')
    axs[0].set_xlabel('Epoki')
    axs[0].set_ylabel('Waga')
    

for i in range(perceptron.weights_hidden_output.shape[0]):
    axs[1].plot(range(epoki), [weights[i] for weights in perceptron.hidden_output_weights_history])

    axs[1].set_title('Wagi między warstwą ukrytą a wyjściową')
    axs[1].set_xlabel('Epoki')
    axs[1].set_ylabel('Waga')
plt.tight_layout()

# Wizualizacja błędów dla warstw
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

epochs = range(epoki)

axs[0].plot(epochs, perceptron.input_hidden_errors)
axs[0].set_title('Błąd warstwy wejściowej - ukrytej')
axs[0].set_xlabel('Epoki')
axs[0].set_ylabel('Błąd średniokwadratowy')

axs[1].plot(epochs, perceptron.hidden_output_errors)
axs[1].set_title('Błąd warstwy ukrytej - wyjściowej')
axs[1].set_xlabel('Epoki')
axs[1].set_ylabel('Błąd średniokwadratowy')
plt.tight_layout()

fig, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(range(epoki), perceptron.training_errors)
ax4.set_xlabel('Epoki')
ax4.set_ylabel('Błąd klasyfikacji')
ax4.set_title('Liczba źle sklasyfikowanych przykładów w trakcie treningu')


# Ustawienie etykiet i tytułów
ax1.set_xlabel('Wejście 1')
ax1.set_ylabel('Wejśćie 2')
#ax1.set_zlabel('Target')
ax1.set_title('Dane treningowe')

ax2.set_xlabel('Wejście 1')
ax2.set_ylabel('Wejście 2')
ax2.set_zlabel('Przewidywana wartość')
ax2.set_title('Przewidywanie')

plt.show()


#https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7 ta strona nam pomagała
#https://www.geeksforgeeks.org/activation-functions-neural-networks/
#https://ichi.pro/pl/sieci-neuronowe-czesc-2-propagacja-wsteczna-i-sprawdzanie-gradientu-87467333057291
