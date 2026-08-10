#imports
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

#wrangle fuction
def wrangle(filepath):
    df=pd.read_csv(filepath)
    mask_ba= df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt=df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] <100000
    df=df[mask_ba & mask_apt & mask_price]
     #remove outliers
    low, high = df["surface_covered_in_m2"].quantile([0.1,0.9])
    mask_area =df["surface_covered_in_m2"].between(low, high)
    
    
    df = df[mask_area]

    #split lat-lon to lat and lon

    df[["lat","lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    

    return frame1

# Use this cell to test your wrangle function on the file `mexico-city-real-estate-1.csv`
frame1 = wrangle() 
print("frame1 shape:",frame1.shape)  
frame1.head()

#subset data
mask_ba= frame1["place_with_parent_names"].str.contains("Distrito Federal")
frame1[mask_ba].head()
#property type
mask_apt=frame1["property_type"] == "apartment"
frame1[mask_apt].head()
#cost less than 100000usd
mask_price = frame1["price_aprox_usd"] <100000
frame1[mask_price].head()


#split lat-lon to lat and lon
frame1["lat-lon"].str.split(",", expand=True).astype(float)
frame1[["lat","lon"]].head()


