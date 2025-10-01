import statsmodels.api as sm

def variables(train_data):
    y = train_data["Survived"]
    x_with = train_data.drop(columns=["Survived", "PassengerId", "Name","Ticket","Cabin","Embarked"],axis=1)
    x_without = train_data.drop(columns=["Parch","Fare","Survived","PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)

    x_with.describe()
    x_without.describe()

    x1 = sm.add_constant(x_with)
    result1 = sm.Logit(y, x1).fit()
    result1.summary()
#print(f"AIC before reducing our dataset {result1.aic}")

    x2 = sm.add_constant(x_without)
    result2 = sm.Logit(y, x2).fit()
    result2.summary()
#print(f"AIC after reducing our dataset {result2.aic}")

    return x_with if result1.aic < result2.aic else x_without
    

   