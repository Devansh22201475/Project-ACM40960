# Project-ACM40960

# Sales Prediction : A Multivariate Approach

Incorporating ML techniques and ensemble models to enhance sales forecast leading to optimized inventory management and more effective sales strategies.


## 🚀 Motivation
In the rapidly changing business landscape, sales forecasting remains crucial for effective decision-making. It enables businesses to adapt to evolving customer behaviors, optimize resource allocation, manage inventory efficiently, and stay ahead of the competition. Accurate sales predictions are essential for navigating uncertainties and driving sustainable growth in the new world.

Machine learning has become increasingly valuable in sales forecasting due to its ability to analyze large volumes of data and uncover complex patterns. ML models can capture non-linear relationships, seasonal trends, and interactions among various factors to generate more accurate predictions. By incorporating ML techniques such as time series analysis and ensemble models, sales forecasting can be enhanced, leading to optimized inventory management and more effective sales strategies.


## Architecture
![image](https://github.com/Devansh22201475/Project-ACM40960/assets/134631225/e6deca46-ede5-4422-a7d3-96f363163314)


## Data Understanding
**sales_train**.csv - the training set. Daily historical data from January 2013 to October 2015.

**test**.csv - the test set. You need to forecast the sales for these shops and products for November 2015.

**sample_submission**.csv - a sample submission file in the correct format.

**items**.csv - supplemental information about the items/products.

**item_categories**.csv  - supplemental information about the items categories.

**shops**.csv- supplemental information about the shops.

## Data fields
**ID** - an Id that represents a (Shop, Item) tuple within the test set

**shop_id** - unique identifier of a shop

**item_id** - unique identifier of a product

**item_category_id** - unique identifier of item category

**item_cnt_day** - number of products sold. You are predicting a monthly amount of this measure

**item_price** - current price of an item

**date** - date in format dd/mm/yyyy

**date_block_num** - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33

**item_name** - name of item

**shop_name** - name of shop

**item_category_name** - name of item category

## Installation

Necessary Installation:

```python
pip install xgboost
pip install lightgbm
```

Necessary Libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error as MSE
```
    
## XGBoost Model

XGBoost is a gradient boosting algorithm that is known for its accuracy and speed, making it an ideal choice for demand forecasting tasks.
By using the XGBoost algorithm for demand forecasting, we can leverage the power of advanced machine learning techniques to make accurate predictions about future demand.


## LightGBM Model
The light gradient boosting machine algorithm – also known as LGBM or LightGBM – is an open-source technique created by Microsoft for machine learning tasks like classification and regression. It is quite similar to XGBoost as it too uses decision trees to classify data.
## Authors

- [@Advayta Zadoo (22201078)](https://www.linkedin.com/in/advayta-zadoo/)
- [@Devansh (22201475)](https://www.linkedin.com/in/devansh-7ab99a8a/)

