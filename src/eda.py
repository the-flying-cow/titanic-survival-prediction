import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def eda_plot(train_data):
    sns.countplot(x="Survived",data= train_data)
    plt.title("Survival")
    plt.show()

    num_cols= []
    num_cols= train_data.select_dtypes(include=['int64','float64']).columns
    num_cols=num_cols.drop("Survived")
    num_cols=num_cols.drop("PassengerId")
    plt.figure(figsize=(20,10))
    for i,col in enumerate(num_cols,start=1):
        plt.subplot(3,3,i)
        sns.boxplot(hue="Survived",y=col,data=train_data)
        plt.title(f"{col} vs Survived")

    plt.tight_layout()
    plt.show()

    num_cols= []
    num_cols= train_data.select_dtypes(include=['int64','float64']).columns
    plt.figure(figsize=(30,20))

    for i,col in enumerate(num_cols,start=1):
        plt.subplot(3,3,i)
        sns.histplot(x=train_data[col].dropna(),data=train_data,bins=30,color="yellow",edgecolor="red")
        plt.ylabel("Frequency")
        plt.title(f"{col}'s Histogram")

    plt.tight_layout()
    plt.show()