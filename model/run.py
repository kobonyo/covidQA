from models import Models
model = Models()
while True:
    q = input("Q: ")
    a = model.predict(q)
    print("A: ",a)