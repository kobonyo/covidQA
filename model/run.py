from tf_idf import TF_IDF
model_list = [TF_IDF()]
model = model_list[0]
while True:
    q = input("Q: ")
    a = model.predict(q)
    print("A: ",a)