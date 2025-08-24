import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def select_top_features(X, y, top_n=8, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logreg = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)
    logreg.fit(X_train, y_train)

    coef_abs = np.abs(logreg.coef_[0])
    coef_df = pd.DataFrame({"Feature": X.columns, "Importance": coef_abs})
    coef_df = coef_df.sort_values(by="Importance", ascending=False)

    top_features = coef_df["Feature"].head(top_n).tolist()

    return X_train[top_features], X_test[top_features], y_train, y_test, top_features
