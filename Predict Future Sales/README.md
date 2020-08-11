# Predict Future Sales
The Kaggle competition [link](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)

## competition goal
The task is to forecast the total amount of products sold in every shop for the test set. Submissions are evaluated by root mean squared error (RMSE).
## Dataset
You are provided with daily historical sales data.
**File descriptions**
  - sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
  - test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
  - items.csv - supplemental information about the items/products.
  - item_categories.csv  - supplemental information about the items categories.
  - shops.csv- supplemental information about the shops.
## achievement
The final rmse metric is 1.08. I ranked top 56% out of 8,766 competitors.
## approach
### Analytics
  - Look into Top5 items count from 2013 to 2015. Use plotly to generate interactive plot for data visualization.
  - Youtube link  (plotly-animation plot)

  [![plotly-animation-plot](http://img.youtube.com/vi/6HQBWwZxIEg/0.jpg)](http://www.youtube.com/watch?v=6HQBWwZxIEg "plotly-animation-plot")
### modelling
#### XGBoost
  - Data Preprocessing
    - datetime
    - sum up sales on a monthly basis
    - Remove outliers
    - fill the month num without item_cnt_day sum with 0 value
    - Fill the nan value
  - Features
      - **seasonality:** month, date_cat_avg_item_cnt, date_block_num, item_cnt_month
      - **shops:** date_shop_cat_avg_item_cnt
      - **items:** item_category_id, delta_price_lag, item_id, date_cat_avg_item_cnt, date_item_avg_item_cnt, item_avg_item_price, date_item_avg_item_price, price_change_percent

    - Model (XGBoost)
      - Train data -- week 0-32
      - Valid data -- week 33
  - Post Process Approach
    - Take 25% high (shop,item) tuple with item_cnt_month, (median of last 4 or 5 month in last year), replace the pred value with the median and clip (0,20)
    - take the median value of all shop and items from first to last year, get all (items, shop) for median with 0, replace their value with 0 &
    - take the tuple not appear in train set but in test set, replace the pred val with 0
