# EV_MARKET-ANALYSIS-AND-FORECASTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})
# 1. LOAD DATA
df = pd.read_csv("ev_market_2026.csv")
# Derived columns
df["price_usd_k"]  = df["price_usd"] / 1_000
df["efficiency_score"] = df["range_miles"] / df["battery_capacity_kwh"]
df["power_to_weight"]  = df["horsepower"] / df["weight_kg"]
df["value_idx"]  = df["range_miles"] / (df["price_usd_k"])
# 2. FIG 5.2 — TOP 10 BRANDS BY TOTAL SALES
top10_brands = (
    df.groupby("brand")["annual_sales_units"]
    .sum()
    .sort_values(ascending=True)
    .tail(10)
)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#e74c3c" if b == "Tesla" else "#4a90d9" for b in top10_brands.index]
bars = ax.barh(top10_brands.index, top10_brands.values / 1e6, color=colors)
for bar, val in zip(bars, top10_brands.values / 1e6):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}M", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Sales (Millions)")
ax.set_title("Fig 5.2 — Top 10 Brands by Total Sales", fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
plt.tight_layout()
plt.savefig("fig5_2_top10_brands.png", dpi=150)
plt.close()
# 3. FIG 5.3 — PRICE vs RANGE BY SEGMENT
seg_colors = {"Budget": "#2ecc71", "Mid-range": "#3498db",
              "Premium": "#f39c12", "Luxury": "#e74c3c"}
fig, ax = plt.subplots(figsize=(11, 6))
for seg, grp in df.groupby("market_segment"):
    ax.scatter(grp["range_miles"], grp["price_usd_k"],
               color=seg_colors.get(seg, "grey"), alpha=0.75, s=60, label=seg)
ax.set_xlabel("Range (miles)")
ax.set_ylabel("Price (USD thousands)")
ax.set_title("Fig 5.3 — Price vs Range by Market Segment", fontweight="bold", pad=12)
ax.legend(title="Segment")
plt.tight_layout()
plt.savefig("fig5_3_price_vs_range.png", dpi=150)
plt.close()
# 4. FIG 5.4 — FEATURE CORRELATION HEATMAP
num_cols = ["price_usd_k", "battery_capacity_kwh", "range_miles",
            "charging_speed_kw", "acceleration_0_60_mph", "horsepower",
            "safety_rating", "annual_sales_units", "customer_rating",
            "efficiency_score", "power_to_weight", "value_idx"]
corr = df[num_cols].rename(columns={
    "price_usd_k": "price_usd", "battery_capacity_kwh": "battery_kwh",
    "charging_speed_kw": "charging_kw", "acceleration_0_60_mph": "accel_060",
    "safety_rating": "safety", "annual_sales_units": "annual_sales",
    "customer_rating": "cust_rating", "power_to_weight": "power_wt"
}).corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.4, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Fig 5.4 — Feature Correlation Heatmap", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("fig5_4_heatmap.png", dpi=150)
plt.close()
# 5. FIG 5.5 — MARKET SEGMENT SHARE (COUNT & SALES)
seg_count = df["market_segment"].value_counts()
seg_sales = df.groupby("market_segment")["annual_sales_units"].sum()
seg_order = ["Budget", "Mid-range", "Premium", "Luxury"]
pie_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in zip(axes,
                            [seg_count.reindex(seg_order), seg_sales.reindex(seg_order)],
                            ["Market Segment Share (Count)", "Market Segment Share (Sales)"]):
    wedges, texts, autotexts = ax.pie(
        data, labels=seg_order, colors=pie_colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title(title, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("fig5_5_segment_share.png", dpi=150)
plt.close()
# 6. FIG 5.6 — EV SALES EVOLUTION BY COUNTRY
country_map = {"US": "US", "Germany": "Germany", "South Korea": "S.Korea",
               "China": "China", "Japan": "Japan"}
country_colors = {"US": "#1f77b4", "Germany": "#d62728", "S.Korea": "#2ca02c",
                  "China": "#ff7f0e", "Japan": "#9467bd"}
sales_by_year_country = (
    df[df["country_of_origin"].isin(country_map.keys())]
    .assign(country=lambda d: d["country_of_origin"].map(country_map))
    .groupby(["year", "country"])["annual_sales_units"].sum()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 6))
for country, grp in sales_by_year_country.groupby("country"):
    grp = grp.sort_values("year")
    ax.plot(grp["year"], grp["annual_sales_units"] / 1e6,
            marker="o", label=country, color=country_colors.get(country))
ax.set_xlabel("Year")
ax.set_ylabel("Sales (Millions)")
ax.set_title("Fig 5.6 — EV Sales Evolution by Country of Origin", fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
plt.savefig("fig5_6_country_sales.png", dpi=150)
plt.close()
# 7. FIG 5.7 — BATTERY CAPACITY vs EFFICIENCY
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df["battery_capacity_kwh"], df["efficiency_score"],
                c=df["price_usd_k"], cmap="YlOrRd", alpha=0.75, s=50)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Price (USD K)")
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("Efficiency (miles/kWh)")
ax.set_title("Fig 5.7 — Battery Capacity vs Efficiency Score", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("fig5_7_battery_efficiency.png", dpi=150)
plt.close()
# 8. FIG 5.8 — TOP 12 EV MODELS BY TOTAL SALES
top12_models = (
    df.groupby(["brand", "model"])["annual_sales_units"]
    .sum()
    .reset_index()
    .sort_values("annual_sales_units", ascending=False)
    .head(12)
    .sort_values("annual_sales_units")
)
brand_color_map = {"Tesla": "#e74c3c", "Volkswagen": "#3498db", "BYD": "#2ecc71"}
model_colors = [brand_color_map.get(b, "#95a5a6") for b in top12_models["brand"]]
model_labels = top12_models["brand"] + " " + top12_models["model"]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(model_labels, top12_models["annual_sales_units"] / 1e6, color=model_colors)
for bar, val in zip(bars, top12_models["annual_sales_units"] / 1e6):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}M", va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Sales (Millions)")
ax.set_title("Fig 5.8 — Top 12 EV Models by Total Sales", fontweight="bold", pad=12)
from matplotlib.patches import Patch
legend_elements = [Patch(fc=c, label=b) for b, c in brand_color_map.items()]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
plt.savefig("fig5_8_top12_models.png", dpi=150)
plt.close()
# 9. ENCODE FEATURES FOR ML
le_seg = LabelEncoder()
le_coo = LabelEncoder()
df["market_segment_enc"] = le_seg.fit_transform(df["market_segment"])
df["country_of_origin_enc"] = le_coo.fit_transform(df["country_of_origin"])
FEATURES = [
    "horsepower", "market_segment_enc", "torque_nm", "customer_rating",
    "charging_speed_kw", "warranty_years", "autopilot_level",
    "acceleration_0_60_mph", "country_of_origin_enc",
    "cargo_volume_cubic_ft", "power_to_weight",
    "battery_capacity_kwh", "efficiency_score", "range_miles", "weight_kg",
]
TARGET = "price_usd_k"
model_df = df[FEATURES + [TARGET]].dropna()
X = model_df[FEATURES]
y = model_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)
# 10. FIG 6.2 — RANDOM FOREST: ACTUAL vs PREDICTED + RESIDUALS
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
residuals = y_test.values - y_pred
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"Random Forest  →  MAE: {mae:.2f} K  |  R²: {r2:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.5, color="#4a90d9", s=25)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual (USD K)")
ax.set_ylabel("Predicted (USD K)")
ax.set_title("Actual vs Predicted Price\n(Random Forest)", fontweight="bold")
ax.legend()
# Residual distribution
ax2 = axes[1]
ax2.hist(residuals, bins=30, color="#4a90d9", edgecolor="white")
ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Residual (USD K)")
ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution", fontweight="bold")
fig.suptitle("Fig 6.2 — Actual vs Predicted & Residual Distribution",
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("fig6_2_rf_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
# 11. FIG 6.3 — TOP 15 FEATURE IMPORTANCES
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
colors_fi = ["#e74c3c" if v > 0.15 else "#4a90d9" for v in feat_imp.values]
bars = ax.barh(feat_imp.index, feat_imp.values, color=colors_fi)
for bar, val in zip(bars, feat_imp.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("Importance Score")
ax.set_title("Fig 6.3 — Top 15 Feature Importances (Random Forest)",
             fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("fig6_3_feature_importance.png", dpi=150)
plt.close()
# 12. FIG 7.1 — GLOBAL EV SALES FORECAST 2027-2031
hist_years  = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
hist_sales  = [3.1,  6.5,  10.2, 13.8, 18.2, 21.6, 22.7]
fore_years  = [2027, 2028, 2029, 2030, 2031]
fore_sales  = [28.4, 35.5, 44.4, 55.5, 69.4]
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(hist_years, hist_sales, color="#4a90d9", label="Historical", width=0.7)
ax.bar(fore_years, fore_sales, color="#e07070", label="Forecasted", width=0.7)
for yr, val in zip(hist_years + fore_years, hist_sales + fore_sales):
    ax.text(yr, val + 0.5, f"{val}M", ha="center", fontsize=8, fontweight="bold")
ax.plot(hist_years + fore_years, hist_sales + fore_sales,
        "o-", color="black", linewidth=1.2, markersize=4)
ax.set_xlabel("Year")
ax.set_ylabel("Sales (Millions)")
ax.set_title("Fig 7.1 — Global EV Sales Forecast 2027–2031", fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
plt.savefig("fig7_1_global_forecast.png", dpi=150)
plt.close()
# 13. FIG 7.2 — BRAND-LEVEL SALES FORECAST (TOP 5 BRANDS)
top5 = ["Tesla", "BYD", "Volkswagen", "Kia", "Hyundai"]
brand_colors_map = {
    "Tesla": "#1f4e79", "BYD": "#c00000",
    "Volkswagen": "#375623", "Kia": "#e6a817", "Hyundai": "#7b3f96"
}
# Historical brand data per year from dataset
brand_hist = (
    df[df["brand"].isin(top5)]
    .groupby(["year", "brand"])["annual_sales_units"]
    .sum()
    .reset_index()
)
# Simple GBR per brand for forecast
fore_years_b  = list(range(2027, 2032))
brand_forecasts = {}
for brand in top5:
    b_df = brand_hist[brand_hist["brand"] == brand].sort_values("year")
    X_b  = b_df["year"].values.reshape(-1, 1)
    y_b  = b_df["annual_sales_units"].values / 1e6
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr.fit(X_b, y_b)
    brand_forecasts[brand] = gbr.predict(np.array(fore_years_b).reshape(-1, 1))
fig, ax = plt.subplots(figsize=(11, 6))
for brand in top5:
    b_df = brand_hist[brand_hist["brand"] == brand].sort_values("year")
    hist_y = b_df["year"].values
    hist_v = b_df["annual_sales_units"].values / 1e6
    c = brand_colors_map[brand]
    ax.plot(hist_y, hist_v, "o-", color=c, label=brand, linewidth=1.8)
    ax.plot(fore_years_b, brand_forecasts[brand], "o--", color=c, linewidth=1.5)
ax.set_xlabel("Year")
ax.set_ylabel("Sales (Millions)")
ax.set_title("Fig 7.2 — Brand-Level Sales Forecast (Top 5 Brands, 2027–2031)",
             fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
plt.savefig("fig7_2_brand_forecast.png", dpi=150)
plt.close()
