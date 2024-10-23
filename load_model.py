
import pickle

def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict():
    model = load_model()
    
    Age = float(input('Enter value for Age: '))
Sex = float(input('Enter value for Sex: '))
Chest pain type = float(input('Enter value for Chest pain type: '))
BP = float(input('Enter value for BP: '))
Cholesterol = float(input('Enter value for Cholesterol: '))
FBS over 120 = float(input('Enter value for FBS over 120: '))
EKG results = float(input('Enter value for EKG results: '))
Max HR = float(input('Enter value for Max HR: '))
Exercise angina = float(input('Enter value for Exercise angina: '))
ST depression = float(input('Enter value for ST depression: '))
Slope of ST = float(input('Enter value for Slope of ST: '))
Number of vessels fluro = float(input('Enter value for Number of vessels fluro: '))
Thallium = float(input('Enter value for Thallium: '))
    prediction = model.predict([[str(Age), str(Sex), str(Chest pain type), str(BP), str(Cholesterol), str(FBS over 120), str(EKG results), str(Max HR), str(Exercise angina), str(ST depression), str(Slope of ST), str(Number of vessels fluro), str(Thallium)]])
    
    print("Predicted output:", prediction)

if __name__ == "__main__":
    predict()
