#imports

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import VimeoVideo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)



#Function
def wrangle(file path):
    #read csv into df
    df=pd.read_csv(file path)
    #subset to capital federal
    mask_ba = df["place_with_parent_names"].str.contains("capital federal")
    #subset to "appartment"
    mask_apt=df["property_type"] == "apartment"
    #subset to properties where pric epaprox usd <400000
    mask_price = df["price_approx_usd"] <400000


    df=df[mask_ba & mask_apt & mask_price]

    #remove outliers on "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile(0.1,0.9)
    mask_area =df["surface_covered_in_m2"].between(low, high)
    df=df[mask_area]

    return df


#use wrangle function
df = wrangle() 
print("df shape:",df.shape)  
df.head() 


#subset data mask
mask_ba = df["place_with_parent_names"].str.contains("capital federal")
df[mask_ba].head()
#df property type
df["property_type"].unique()
mask_apt=df["property_type"] == "apartment"
df[mask_apt].head()

mask_price = df["price_approx_usd"] <400000
mask_prc.head()

#Area histogram
plt.hist("surface_covered_in_m2")
plt.xlabel("area [square meters]")
plt.title("Distribution of Apartment sizes");


df.describe()["surface_covered_in_m2"]


# quantile 
low, high = df["surface_covered_in_m2"].quantile([0.1,0.9])
mask_area =df["surface_covered_in_m2"].between(low, high)
mask_area.head()

#scatter plot for price 
plt.scatter(x=df["surface_covered_in_m2"],y=df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Buenos Aires price vs Area");

#split
#feature metrics
features = ["surface_covered_in_m2"]
X_train = df[features]
X_train.shape

#Target vector
target = "price_aprox_usd"
y_train = df[target]
y_train.shape

#BUID A MODEL

#Baseline
y_mean = y_train.mean()
y_mean

#
y_pred_baseline = [y_mean] * len(y_train)

#visualize pred baseline
plt.plot(X_train["surface_covered_in_m2"],y_pred_baseline, color = "Orange", label ="Baseline Model")
# can be done as (X_train.values)
plt.scatter(X_train, y_train)
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Buenos Aires: Price vs. Area")
plt.legend();

#performance metric
#baseline mean absolute error
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

#ITERATE
#linear regression
model = LinearRegression()

#fit the model
model.fit(X_train, y_train)

#Evaluate 
#generate predictions uding scikit learn
y_pred_training = model.predict(X_train)
y_pred_training[:5]

#mean absolute errors for list of prediction
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


#test using a diff datset
X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()


#extract intercept from a model
intercept = round(model.intercept_,2)
print("Model Intercept:", intercept)
assert any([isinstance(intercept, int), isinstance(intercept, float)])

#coeficient 
coefficient = round(model.coef_[0],2)
print('Model coefficient for "surface_covered_in_m2":', coefficient)
assert any([isinstance(coefficient, int), isinstance(coefficient, float)])

#Equation that the model determined for prediction
print(f"apartment_price = {intercept} + {coefficient} * surface_covered")

#plot the model
#relationship between observations
plt.plot(X_train.values, model.predict(X_train), color="red", label="Linear Model")
plt.scatter(X_train, y_train)
plt.xlabel("surface covered [sq meters]")
plt.ylabel("price [usd]")
plt.legend();