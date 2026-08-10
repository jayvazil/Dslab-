import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import VimeoVideo
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)




def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    #split col lat-lon
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns = "lat-lon", inplace = True)
    

    return df




frame1 = wrangle("data/buenos-aires-real-estate-1.csv")
print(frame1.info())
frame1.head()    

#split the lat-lon column
frame1["lat-lon"].str.split(",", expand=True).astype(float)


#open 2nd data frame 
frame2 = wrangle("data/buenos-aires-real-estate-2.csv")

#concatenate the two dsets
df = pd.concat([frame1,frame2], ignore_index=True)
print(df.info())
df.head()

#exploratory data analysis
#mapbox scatter
fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat= "lat",
    lon= "lat",
    width=600,  # Width of map
    height=600,  # Height of map
    color= "price_aprox_usd",
    hover_data=["price_aprox_usd"],  # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()


# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x= "lon"
    y= "lat"
    z= "price_aprox_usd"
    labels={"lon": "longitude", "lat": "latitude", "price_aprox_usd": "price"},
    width=600,
    height=500,
)

# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)

# Display figure
fig.show()


#split 
features = ["lon", "lat"]
X_train = df[features]
X_train.head()

target = "price_aprox_usd"
y_train = df[target]
y_train.head()


#BUILD A MODEL
#summery statistic
y_mean = y_train.mean()
# length
y_pred_baseline = [y_mean] * len(y_train)

#baseline mean absolute error
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

#ITERATE
#instanciate imputer
imputer =SimpleImputer

#fit transformer to training data in scikit learn
imputer.fit(X_train)

#transform 
XT_train = imputer.transform(X_train)
pd.DataFrame(XT_train, columns=X_train.columns).info()


#create pipeline in scikit learn
model = make_pipeline(
    SimpleImputer(),
    LinearRegression()

)

#Fit a model to training data in scikit-learn.
model.fit(X_train,y_train)


#EVALUATE
#prediction using model scikit learn
y_pred_training = model.predict(X_train)

#train mean absolute error 
mae_training = mean_absolute_error(y_train,y_pred_training)
print("Training MAE:", round(mae_training, 2))


#COMMUNICATE RESULTS
#extract intercepts and coefficients
intercept = model.named_steps["linearregression"].intercept_.round()
coefficients = model.named_steps["linearregression"].coef_round()

#print equation
print(
    f"price = {} + ({} * longitude) + ({} * latitude)"
)

#3d scatter in plotly
# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x= "lon",
    y= "lat",
    z= "price_aprox_usd",
    labels={"lon": "longitude", "lat": "latitude", "price_aprox_usd": "price"},
    width=600,
    height=500,
)

# Create x and y coordinates for model representation
x_plane = np.linspace(df["lon"].min(), df["lon"].max(), 10)
y_plane = np.linspace(df["lat"].min(), df["lat"].max(), 10)
xx, yy = np.meshgrid(x_plane, y_plane)

# Use model to predict z coordinates
z_plane = model.predict(pd.DataFrame({"lon": x_plane, "lat": y_plane}))
zz = np.tile(z_plane, (10, 1))

# Add plane to figure
fig.add_trace(go.Surface(x=xx, y=yy, z=zz))

# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)

# Display figure
fig.show()
