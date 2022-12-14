import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.trees import RandomForestClassifier

dataFrame = pd.read_csv("data.csv")

ratio = 70
dataDivide = len(dataFrame) * ratio // 100

x1 = dataFrame.iloc[:, 1:].values.tolist()
x2 = sk.utils.validation.column_or_1d(dataFrame.iloc[:, :1].values.tolist())
x1Train, x1Test, x2Train, x2Test = x1[:dataDivide], x1[
    dataDivide:], x2[:dataDivide], x2[dataDivide:]
classifier = RandomForestClassifier(n_estimators=100,
                                    max_depth=5,
                                    random_state=1)
classifier.fit(x1Train, x2Train)
result = sk.metrics.classification_report(x2Test, classifier.predict(x1Test))
print(result)