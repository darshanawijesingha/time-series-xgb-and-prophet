import pandas as pd 
import numpy as np

from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

spar_pickme = pd.read_csv('spk_item_report_finance_purpose-item_csv-3eb52f9c8099-2025-05-28-07-31-44.csv')
spar_pickme.head(5)

spar_uber_11 = pd.read_csv('7ce2ca93-fcb4-4ee0-828e-df312c3ce347-india.csv')
spar_uber_11

spar_uber_6 = pd.read_csv('2d965370-554f-4348-9661-0db7d7add917-india.csv')
spar_uber_6


spar_uber = pd.concat([spar_uber_11, spar_uber_6], ignore_index=True)
spar_uber

up_spar_pickme = spar_pickme[spar_pickme['restaurant'] == "kuruduwaththa"]
up_spar_pickme

spar_uber['Order Date']= pd.to_datetime(spar_uber['Order Date'])
spar_uber

up_spar_pickme.columns


# Ensure created_Date is in datetime format
up_spar_pickme['created_Date'] = pd.to_datetime(up_spar_pickme['created_Date'])
up_spar_pickme


spar_uber = spar_uber.drop(
    ['Store Name', 'Store ID', 'Workflow ID', 
       'Order Accept Time', 'Dining Mode', 'Payment Mode', 'Order Channel',
       'Order Status', 'Customer Uber-Membership Status', 
       'Tax on sales', 'Refunds (excl tax)', 'Tax on Refunds',
       'Price adjustments (excl. tax)', 'Tax on price adjustments',
       'Promotions on items (excl tax)', 'Tax on Promotion on items',
       'Promotions on items (incl tax)', 'Delivery Charge (excl tax)',
       'Tax on Delivery Fee', 'Promotions on delivery charge (excl tax)',
       'Tax on Promotions on delivery charge',
       'Promotions on delivery charge (incl tax)', 'Bag Fee',
       'Delivery Fee (excl tax)', 'Tax on Delivery Fee.1',
       'Delivery Fee (incl tax)', 'Service Fee', 'Tax on Service Fee',
       'Service Fee Discount', 'Net Service fee', 'Discount Adjustments',
       'Profit on Delivery', 'Tax Adjustment', 'Other payments description',
       'Other payments (incl. tax)', 'Total payout ', 'Payout Date',
       'Payout Status', 'Invoice link U2R ', 'Invoice link C2R ',
       'Invoice link R2E ', 'Retailer Loyalty ID', 'Payout reference ID'],
    axis=1
)



up_spar_pickme = up_spar_pickme.drop(
    ['service_group_code', 'ResturantId', 'created_Time', 'pickup_time', 'completed_time', 'item', 'Item_ID', 'district','quantity'],
    axis=1
)



spar_uber_daily = spar_uber.groupby('Order Date').agg({
    'Sales (excl. tax)': 'sum',
    'Order ID': 'count'
}).rename(columns={
    'Sales (excl. tax)': 'ItemTotal',
    'Order ID': 'order_count'
}).reset_index().rename(columns={
    'Order Date': 'created_Date'
}).set_index('created_Date')

print(spar_uber_daily)



up_spar_pickme_daily= up_spar_pickme.groupby('created_Date').agg({
    ##'quantity': 'sum',
    ##'Item_price': 'sum',
    'ItemTotal': 'sum',
    'orderid': 'count'
}).rename(columns={'orderid': 'order_count'})

print(up_spar_pickme_daily)



# Step 2: Merge on index with sum for overlapping dates
combined_daily = up_spar_pickme_daily.add(spar_uber_daily, fill_value=0)

# Step 3 (optional): Round numbers or convert counts to int if needed
combined_daily['order_count'] = combined_daily['order_count'].astype(int)

print(combined_daily)

combined_daily = combined_daily.sort_index()
combined_daily



from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ---- Feature Engineering ----
def create_features(combined_daily, target_col):
    combined_daily = combined_daily.copy()
    combined_daily['dayofweek'] = combined_daily.index.dayofweek
    combined_daily['month'] = combined_daily.index.month
    combined_daily['lag_1'] = combined_daily[target_col].shift(1)
    combined_daily['rolling_3'] = combined_daily[target_col].rolling(3).mean()
    return combined_daily.dropna()

# ---- XGBoost Forecasting ----
def forecast_xgboost(combined_daily, target_col, test_days=30):
    df_feat = create_features(combined_daily, target_col)
    features = ['dayofweek', 'month', 'lag_1', 'rolling_3']

    train = df_feat.iloc[:-test_days]
    test = df_feat.iloc[-test_days:]
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict test set
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Forecast next 30 days
    future_preds = []
    temp_df = combined_daily.copy()

    for i in range(1, 31):
        date = temp_df.index[-1] + timedelta(days=1)
        row = {
            'dayofweek': date.dayofweek,
            'month': date.month,
            'lag_1': temp_df[target_col].iloc[-1],
            'rolling_3': temp_df[target_col].rolling(3).mean().iloc[-1]
        }
        x_pred = pd.DataFrame([row])
        pred = model.predict(x_pred)[0]
        future_preds.append(pred)
        temp_df.loc[date] = [pred if c == target_col else np.nan for c in temp_df.columns]

    future_index = [temp_df.index[-30 + i] for i in range(30)]
    return pd.Series(future_preds, index=future_index), test_rmse

# ---- Prophet Forecasting ----
def forecast_prophet(combined_daily, target_col, test_days=30):
    prophet_df = combined_daily[[target_col]].reset_index()
    prophet_df.columns = ['ds', 'y']
    train = prophet_df.iloc[:-test_days]
    test = prophet_df.iloc[-test_days:]

    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=test_days + 30)
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat']].set_index('ds')

    test_forecast = forecast.iloc[-(test_days + 30):-30]
    test_rmse = np.sqrt(mean_squared_error(test.set_index('ds')['y'], test_forecast['yhat']))
    future_forecast = forecast.iloc[-30:]
    return future_forecast['yhat'], test_rmse

# ---- Example Dataset ----
# Ensure you have a DataFrame named `combined_daily` with a datetime index and columns 'ItemTotal' and 'order_count'.
# Example:
# combined_daily = pd.read_csv("your_data.csv", parse_dates=['date'], index_col='date')

# ---- Main Forecast Loop ----
forecast_results = {}
evaluation_metrics = {}

for col in ['ItemTotal', 'order_count']:
    print(f"\nForecasting {col}...")

    xgb_forecast, xgb_rmse = forecast_xgboost(combined_daily[[col]].copy(), col)
    prophet_forecast, prophet_rmse = forecast_prophet(combined_daily[[col]].copy(), col)

    forecast_results[col] = {
        'xgb': xgb_forecast,
        'prophet': prophet_forecast
    }
    evaluation_metrics[col] = {
        'xgb_test_rmse': xgb_rmse,
        'prophet_test_rmse': prophet_rmse
    }

# ---- Combine Results ----
forecast_df = pd.DataFrame({
    f"{col}_xgb": forecast_results[col]['xgb'] for col in forecast_results
}).join(pd.DataFrame({
    f"{col}_prophet": forecast_results[col]['prophet'] for col in forecast_results
}))

print("\n🔍 Forecast Accuracy (RMSE on Test Set):")
for col, scores in evaluation_metrics.items():
    print(f"{col}:")
    for model, score in scores.items():
        print(f"  {model}: {score:.2f}")

# ---- Plot ----
forecast_df.plot(subplots=True, figsize=(14, 10), title="📈 Forecasts for Next 30 Days")
plt.tight_layout()
plt.show()




# ---- Feature Engineering ----
def create_features(combined_daily, target_col):
    combined_daily = combined_daily.copy()
    combined_daily['dayofweek'] = combined_daily.index.dayofweek
    combined_daily['month'] = combined_daily.index.month
    combined_daily['lag_1'] = combined_daily[target_col].shift(1)
    combined_daily['rolling_3'] = combined_daily[target_col].rolling(3).mean()
    return combined_daily.dropna()

# ---- XGBoost Forecasting ----
def forecast_xgboost(combined_daily, target_col, test_days=30):
    df_feat = create_features(combined_daily, target_col)
    features = ['dayofweek', 'month', 'lag_1', 'rolling_3']

    train = df_feat.iloc[:-test_days]
    test = df_feat.iloc[-test_days:]
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict test set
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Forecast next 30 days
    future_preds = []
    temp_df = combined_daily.copy()

    for i in range(1, 31):
        date = temp_df.index[-1] + timedelta(days=1)
        row = {
            'dayofweek': date.dayofweek,
            'month': date.month,
            'lag_1': temp_df[target_col].iloc[-1],
            'rolling_3': temp_df[target_col].rolling(3).mean().iloc[-1]
        }
        x_pred = pd.DataFrame([row])
        pred = model.predict(x_pred)[0]
        future_preds.append(pred)
        temp_df.loc[date] = [pred if c == target_col else np.nan for c in temp_df.columns]

    future_index = [temp_df.index[-30 + i] for i in range(30)]
    return pd.Series(future_preds, index=future_index), test_rmse, model

# ---- Prophet Forecasting ----
def forecast_prophet(combined_daily, target_col, test_days=30):
    prophet_df = combined_daily[[target_col]].reset_index()
    prophet_df.columns = ['ds', 'y']
    train = prophet_df.iloc[:-test_days]
    test = prophet_df.iloc[-test_days:]

    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=test_days + 30)
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')

    test_forecast = forecast.iloc[-(test_days + 30):-30]
    test_rmse = np.sqrt(mean_squared_error(test.set_index('ds')['y'], test_forecast['yhat']))
    future_forecast = forecast.iloc[-30:]
    return future_forecast['yhat'], test_rmse, model, future_forecast

# ---- Main Forecast Function ----
def generate_forecasts(combined_daily):
    forecast_results = {}
    evaluation_metrics = {}
    models = {}

    for col in ['ItemTotal', 'order_count']:
        print(f"\nForecasting {col}...")

        # Use Prophet for ItemTotal, XGBoost for order_count
        if col == 'ItemTotal':
            forecast, rmse, model, full_forecast = forecast_prophet(combined_daily[[col]].copy(), col)
            forecast_results[col] = {
                'forecast': forecast,
                'rmse': rmse,
                'model': 'prophet',
                'full_forecast': full_forecast
            }
        else:
            forecast, rmse, model = forecast_xgboost(combined_daily[[col]].copy(), col)
            forecast_results[col] = {
                'forecast': forecast,
                'rmse': rmse,
                'model': 'xgboost'
            }
        evaluation_metrics[col] = {'rmse': rmse}
        models[col] = model

    return forecast_results, evaluation_metrics, models

# ---- Extended Forecast Calculation ----
def calculate_extended_forecast(last_value, rmse, days, growth_rate=0):
    """Calculate extended forecast with optional growth rate"""
    daily_values = [last_value * (1 + growth_rate)**i for i in range(days)]
    total = sum(daily_values)
    # Scale RMSE by sqrt of time period
    total_rmse = rmse * np.sqrt(days/30)
    
    return {
        'total': total,
        'lower_bound': total - total_rmse,
        'upper_bound': total + total_rmse,
        'rmse': total_rmse
    }

# ---- Save Forecasts to Separate CSVs ----
def save_forecasts_separate(forecast_results):
    for col in forecast_results:
        if forecast_results[col]['model'] == 'prophet':
            # For Prophet forecasts
            forecast_df = forecast_results[col]['full_forecast'].copy()
            forecast_df.columns = ['forecast', 'lower_bound', 'upper_bound']
        else:
            # For XGBoost forecasts
            forecast_series = forecast_results[col]['forecast']
            rmse = forecast_results[col]['rmse']
            forecast_df = pd.DataFrame({
                'forecast': forecast_series,
                'lower_bound': forecast_series - rmse,
                'upper_bound': forecast_series + rmse
            })
        
        # Save to separate CSV files
        filename = f"{col}_forecast.csv"
        forecast_df.to_csv(filename)
        print(f"Saved {col} forecast to {filename}")

# ---- Plot Forecasts ----
def plot_forecasts(combined_daily, forecast_results):
    for col in forecast_results:
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(combined_daily.index, combined_daily[col], label='Historical', color='blue')
        
        # Plot forecast
        forecast = forecast_results[col]['forecast']
        model = forecast_results[col]['model']
        rmse = forecast_results[col]['rmse']
        
        plt.plot(forecast.index, forecast.values, 
                label=f'{model} Forecast', linestyle='--', color='red')
        
        # Add confidence interval
        if model == 'prophet':
            full_forecast = forecast_results[col]['full_forecast']
            plt.fill_between(full_forecast.index,
                            full_forecast['yhat_lower'],
                            full_forecast['yhat_upper'],
                            color='red', alpha=0.2,
                            label='Confidence Interval')
        else:
            plt.fill_between(forecast.index,
                            forecast.values - rmse,
                            forecast.values + rmse,
                            color='red', alpha=0.2,
                            label='± RMSE Margin')
        
        plt.title(f"{col} - {model.capitalize()} Forecast with Confidence Interval")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ---- Execute Forecasts ----
# Assuming combined_daily is your input DataFrame with datetime index
# combined_daily = pd.read_csv("your_data.csv", parse_dates=['date'], index_col='date')

forecast_results, evaluation_metrics, models = generate_forecasts(combined_daily)

# 1. Plot 30-day forecasts with margins
plot_forecasts(combined_daily, forecast_results)

# 2. Save forecasts to separate CSV files
save_forecasts_separate(forecast_results)

# 3. Calculate 6-month and 1-year projections
print("\n📈 Extended Forecast Projections:")
for col in forecast_results:
    last_value = forecast_results[col]['forecast'].iloc[-1]
    rmse = forecast_results[col]['rmse']
    model = forecast_results[col]['model']
    
    # Calculate projections (assuming 0% growth rate - adjust as needed)
    six_month = calculate_extended_forecast(last_value, rmse, 180)
    one_year = calculate_extended_forecast(last_value, rmse, 365)
    
    print(f"\n{col} Projections ({model}):")
    print(f"6-Month Total: {six_month['total']:,.2f}")
    print(f"  Range: {six_month['lower_bound']:,.2f} to {six_month['upper_bound']:,.2f}")
    print(f"1-Year Total: {one_year['total']:,.2f}")
    print(f"  Range: {one_year['lower_bound']:,.2f} to {one_year['upper_bound']:,.2f}")


print(pd.read_csv('ItemTotal_forecast.csv').head())


print(pd.read_csv('order_count_forecast.csv').head())