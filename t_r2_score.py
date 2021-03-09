from sklearn.metrics import r2_score
#r2_score(gercek,tahmin)
gercek = [1,2,3,4,5,6,7]
tahmin = [1.1,2.3,3.5,4.2,7,7.8,8]

print(r2_score(gercek,tahmin))