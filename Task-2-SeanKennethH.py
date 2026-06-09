import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Dataset for Data Analytics.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

#Check data types of each column
print("Data Types of Each Column:")
print(df.dtypes)

#Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Replacing Missing Data with a Specific Value
df = df.fillna("No Referral") 

#Check missing values after replacement
print("Missing Values after cleaning:")
print(df.isnull().sum())

# Drop Duplicates for Order ID
print("Duplicates before cleaning:", df.duplicated(subset=["OrderID"]).sum())

# Remove duplicates (if any)
df = df.drop_duplicates(subset=["OrderID"], keep="first")
print("Duplicates after cleaning:", df.duplicated(subset=["OrderID"]).sum())

#Cleaned DataFrame
print("Cleaned DataFrame:")
print(df)

# --------------------------
# NEW PROJECT 2 EDA EXTENSION STARTS HERE
# --------------------------
plt.rcParams['figure.figsize'] = (10,5)
sns.set_style("whitegrid")

# 1. Fix Date data type (critical for time analysis)
df['Date'] = pd.to_datetime(df['Date'])
# Create year/month columns for trend analysis
df['OrderYear'] = df['Date'].dt.year
df['OrderMonth'] = df['Date'].dt.month

# 2. Basic Descriptive Statistics (sum, mean, median, min/max for sales numbers)
print("\n===== NUMERICAL DESCRIPTIVE STATISTICS =====")
cart_col = 'NumItemsInCart' if 'NumItemsInCart' in df.columns else 'ItemsInCart'
numeric_cols = ['Quantity','UnitPrice','TotalPrice', cart_col]
print(df[numeric_cols].describe().round(2))

# Manual individual totals for quick report numbers
total_revenue = df['TotalPrice'].sum()
avg_order_value = df['TotalPrice'].mean()
median_order_value = df['TotalPrice'].median()
total_orders = len(df)

print(f"\nTotal Overall Revenue: ${total_revenue:.2f}")
print(f"Average Order Value: ${avg_order_value:.2f}")
print(f"Median Order Value: ${median_order_value:.2f}")
print(f"Total Unique Orders: {total_orders}")

# 3. Categorical breakdown counts
print("\n===== CATEGORICAL COUNTS =====")
print("Order Status Split:\n", df['OrderStatus'].value_counts())
print("\nPayment Method Split:\n", df['PaymentMethod'].value_counts())
print("\nProduct Order Counts:\n", df['Product'].value_counts())
print("\nReferral Source Traffic:\n", df['ReferralSource'].value_counts())
print("\nCoupon Code Usage:\n", df['CouponCode'].value_counts())

# 4. Aggregated Business Group Stats (sum revenue, avg spend per group)
# 4.1 Sales by Product
product_agg = df.groupby('Product').agg(
    Total_Revenue=('TotalPrice','sum'),
    Total_Units_Sold=('Quantity','sum'),
    Avg_Order_Value=('TotalPrice','mean')
).round(2).sort_values('Total_Revenue', ascending=False)
print("\n===== SALES BY PRODUCT =====")
print(product_agg)

# 4.2 Sales by Order Status
status_agg = df.groupby('OrderStatus').agg(
    Order_Count=('OrderID','count'),
    Total_Revenue=('TotalPrice','sum'),
    Avg_Spend=('TotalPrice','mean')
).round(2)
print("\n===== SALES BY ORDER STATUS =====")
print(status_agg)

# 4.3 Sales by Referral Source
source_agg = df.groupby('ReferralSource').agg(
    Order_Count=('OrderID','count'),
    Total_Revenue=('TotalPrice','sum'),
    Avg_Order=('TotalPrice','mean')
).round(2).sort_values('Total_Revenue', ascending=False)
print("\n===== SALES BY REFERRAL SOURCE =====")
print(source_agg)

# 4.4 Yearly Sales Trend
yearly_agg = df.groupby('OrderYear')['TotalPrice'].sum().round(2)
print("\n===== YEARLY TOTAL REVENUE =====")
print(yearly_agg)

# 5. Outlier Detection for TotalPrice (IQR method)
Q1 = df['TotalPrice'].quantile(0.25)
Q3 = df['TotalPrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_orders = df[(df['TotalPrice'] < lower_bound) | (df['TotalPrice'] > upper_bound)]
print(f"\n===== OUTLIER CHECK =====")
print(f"Number of outlier high/low value orders: {len(outlier_orders)}")
print("Sample outlier orders:")
print(outlier_orders[['OrderID','Product','TotalPrice','OrderStatus']].head())

# 6. Visualization Plots (for report/portfolio)
# Plot 1: Revenue by Product Bar Chart
plt.figure()
product_agg['Total_Revenue'].plot(kind='bar', color='steelblue')
plt.title('Total Revenue by Product')
plt.ylabel('Total Revenue ($)')
plt.xlabel('Product')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Plot 2: Total Price Boxplot (shows outliers visually)
plt.figure()
sns.boxplot(x=df['TotalPrice'], color='lightcoral')
plt.title('Distribution of Total Order Price + Outliers')
plt.xlabel('Order Total ($)')
plt.show()

# Plot3: Yearly Sales Trend Line
plt.figure()
yearly_agg.plot(marker='o', linewidth=3, color='darkgreen')
plt.title('Yearly Total Revenue Trend')
plt.ylabel('Revenue ($)')
plt.xlabel('Year')
plt.grid(True, alpha=0.3)
plt.show()

# Plot4: Order Status Pie Chart
plt.figure()
status_agg['Total_Revenue'].plot(kind='pie', autopct='%1.1f%%')
plt.title('Revenue Split by Order Status')
plt.ylabel('')
plt.show()

# 7. Correlation Heatmap for numeric variables
plt.figure()
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Numeric Variable Correlation Heatmap')
plt.tight_layout()
plt.show()

# 8. Export cleaned dataset for submission
df.to_csv("Cleaned_OrderDataset_Project2.csv", index=False)
print("\nCleaned dataset exported as Cleaned_OrderDataset_Project2.csv")