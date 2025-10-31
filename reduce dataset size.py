import pandas as pd
data = pd.read_csv("online payment frauds.csv")

data = data.tail(20000)
print(data.shape)

data.to_csv("Online Payment Fraud tail 20k.csv")

