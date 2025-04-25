import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="SuperStore Dashboard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f7f9fa; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
@st.cache_data

def load_data():
    df = pd.read_csv("superstore.csv")
    df['Profit Margin'] = df['Profit'] / df['Sales']
    return df

df = load_data()

# Ensure 'Order Date' is parsed correctly
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# ---------- SIDEBAR OPTIONS ----------

# Stylish name header for Ritunjay
st.sidebar.markdown("""
    <h2 style='font-family: cursive; color: goldenrod; text-align: center;'>Ritunjay</h2>
""", unsafe_allow_html=True)

st.sidebar.title("Filters")
region = st.sidebar.multiselect("Select Region", df['Region'].unique(), df['Region'].unique())
category = st.sidebar.multiselect("Select Category", df['Category'].unique(), df['Category'].unique())
sales_range = st.sidebar.slider("Sales Range", float(df['Sales'].min()), float(df['Sales'].max()), (float(df['Sales'].min()), float(df['Sales'].max())))

selected_region = st.sidebar.selectbox("Drill into Region", options=["All"] + sorted(df['Region'].unique()))

guide_mode = st.sidebar.checkbox("📖 Enable Guided Tour")

filtered_df = df[(df['Region'].isin(region)) & 
                 (df['Category'].isin(category)) & 
                 (df['Sales'].between(sales_range[0], sales_range[1]))]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    st.sidebar.markdown(f"📍 Showing data for **{selected_region}**")

# ---------- HEADER ----------
if 'tour_step' not in st.session_state:
    st.session_state.tour_step = 0

tour_steps = [
    "👋 **Welcome Ritunjay!** Let's walk through your SuperStore dashboard step-by-step.",
    "🧮 **Metrics Section** — Quick glance at Sales, Profit, Margins, and Orders.",
    "📊 **Sales Charts** — Analyze Category Sales across product lines.",
    "🌍 **Regional Performance** — See how different regions compare on Sales and Profit.",
    "🏙️ **City Drilldown** — Deep dive into city-level performance inside a region.",
    "📊 **Sales vs Profit Dual-Axis** — Compare Sales and Profit trends simultaneously by category.",
    "🎛️ **What-If Simulator** — Adjust sales, discount, quantity to predict profits.",
    "🌲 **Treemap Visualization** — Explore sales distribution across sub-categories.",
    "🧾 **Raw Data Explorer** — Select columns, search, and export filtered data.",
    "🧾 **Enhanced Data Explorer** — Advanced search, dynamic column control, CSV/Excel downloads.",
    "📈 **Discount Trendline** — Discover how discounts impact profit margins.",
    "🔗 **Correlation Heatmap** — Understand relationships between key business metrics.",
    "📏 **Benchmark Comparison** — Compare real performance against adjustable industry targets.",
    "✅ **That's the tour!** Now explore and uncover insights on your own. 🚀"
]

if guide_mode:
    with st.chat_message("assistant"):
        st.markdown(tour_steps[st.session_state.tour_step])
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.session_state.tour_step > 0:
                if st.button("⬅ Previous", key="prev"):
                    st.session_state.tour_step -= 1
        with col_next:
            if st.session_state.tour_step < len(tour_steps) - 1:
                if st.button("Next ➡", key="next"):
                    st.session_state.tour_step += 1
st.title("📊 SuperStore Interactive Dashboard")
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Hi Ritunjay! Welcome to your personalized dashboard tour.**

        Let's walk through the most powerful parts of your SuperStore analytics experience:

        1. 🧮 **Metrics Section** — Quickly glance at Total Sales, Profit, and Margins.
        2. 📊 **Category & Region Charts** — See what's performing best.
        3. 🏙️ **City Drilldown** — Dive deeper into a selected region.
        4. 🎛️ **What-If Simulator** — Adjust sales, discount, and quantity to see profit predictions.
        5. 🤖 **AI Insights** — Smart text analysis that speaks your data.
        6. 📄 **Report Export** — Generate a snapshot summary in one click.

        👉 Use the filters in the sidebar to customize your view. I'm here to help you uncover what matters most! 🔍
        """)

st.markdown("This dashboard provides a visual and analytical overview of the **SuperStore** dataset. Use the filters on the left to explore different regions, categories, and sales ranges.")

# ---------- 📈 PERFORMANCE OVERVIEW ----------
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        🧮 **Metrics Section** gives you a quick glance at:
        - 💰 Total Sales
        - 📈 Total Profit
        - 🧮 Profit Margin
        - 📦 Order Count
        Use these numbers to judge overall health at a glance.
        """)
total_sales_filtered = filtered_df['Sales'].sum()
total_profit_filtered = filtered_df['Profit'].sum()
order_count = filtered_df.shape[0]
profit_margin = total_profit_filtered / total_sales_filtered if total_sales_filtered else 0

total_sales_all = df['Sales'].sum()
total_profit_all = df['Profit'].sum()

is_filtered = (len(region) < len(df['Region'].unique())) or \
              (len(category) < len(df['Category'].unique())) or \
              (sales_range != (df['Sales'].min(), df['Sales'].max()))

delta_sales = total_sales_filtered - total_sales_all if is_filtered else None
delta_profit = total_profit_filtered - total_profit_all if is_filtered else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Sales", f"${total_sales_filtered:,.0f}", delta=f"${delta_sales:,.0f}" if delta_sales is not None else None)
col2.metric("📈 Total Profit", f"${total_profit_filtered:,.0f}", delta=f"${delta_profit:,.0f}" if delta_profit is not None else None)
col3.metric("🧮 Profit Margin", f"{profit_margin:.2%}")
col4.metric("📦 Order Count", f"{order_count}")

# ---------- AI-GENERATED INSIGHT ----------
if total_sales_filtered > 0:
    top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    top_state = filtered_df.groupby("State")["Profit"].sum().idxmax()
    top_cat = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    avg_discount = filtered_df['Discount'].mean()
    st.markdown(f"**🤖 Summary Insight:** In the current view, **{top_region}** region is leading sales. Top profit comes from **{top_state}**, with **{top_cat}** category dominating. Average discount is **{avg_discount:.0%}**, influencing the overall **{profit_margin:.2%}** profit margin.")

# ---------- 🧭 DEEP DIVE VISUALS ----------
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        📊 **Category & Region Charts**:
        - Discover which product categories and regions are contributing the most.
        - Use the color-coded bar charts to compare sales and profits visually.
        """)
st.subheader("📁 Sales by Category")
cat_sales = filtered_df.groupby("Category").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
fig1 = px.bar(cat_sales, x="Category", y="Sales", color="Profit", title="Sales by Category with Profit Scale")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌍 Regional Performance")
region_perf = filtered_df.groupby("Region")[['Sales', 'Profit']].sum().reset_index()
fig2 = px.bar(region_perf, x="Region", y=["Sales", "Profit"], barmode="group", title="Regional Sales and Profit")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🏙️ City Drilldown (if Region selected)")
if selected_region != "All":
    city_perf = filtered_df.groupby("City")[['Sales', 'Profit']].sum().sort_values("Sales", ascending=False).reset_index()
    fig_city = px.bar(city_perf, x="City", y="Sales", color="Profit", title=f"City-level Performance in {selected_region}")
    st.plotly_chart(fig_city, use_container_width=True)

# ---------- 🔮 SIMULATORS & PREDICTIVE TOOLS ----------
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        🎛️ **What-If Simulator**:
        - Adjust the sliders to simulate a hypothetical scenario.
        - Predict how changing Sales, Discount, and Quantity affects your profit.
        - Compare against historical averages instantly.
        """)
st.subheader("🎛️ What-If Profit Simulator")
X = filtered_df[['Sales', 'Discount', 'Quantity']]
y = filtered_df['Profit']
model = LinearRegression().fit(X, y)

sales_input = st.slider("Sales", 0.0, 2000.0, 200.0)
discount_input = st.slider("Discount", 0.0, 0.9, 0.1)
quantity_input = st.slider("Quantity", 1, 10, 2)
predicted_profit = model.predict([[sales_input, discount_input, quantity_input]])[0]
st.success(f"💡 Predicted Profit: ${predicted_profit:.2f}")

# Charting the prediction vs historical average
avg_profit = y.mean()
fig_sim = px.bar(x=["Your Prediction", "Avg Historical Profit"], y=[predicted_profit, avg_profit],
                 title="Predicted Profit vs Historical Average", labels={'x': ""}, color=["Your Prediction", "Avg Historical Profit"])
st.plotly_chart(fig_sim, use_container_width=True)

# ---------- 📄 REPORT & DATA EXPORT ----------
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        📄 **Report Export**:
        - Click the link to download an HTML summary of your filtered view.
        - Great for sharing a quick performance snapshot with others.
        """)
def create_html_report():
    return f"""
    <h2>SuperStore Summary Report</h2>
    <p><strong>Total Sales:</strong> ${total_sales_filtered:,.0f}</p>
    <p><strong>Total Profit:</strong> ${total_profit_filtered:,.0f}</p>
    <p><strongProfit Margin:</strong> {profit_margin:.2%}</p>
    <p><strong>Order Count:</strong> {order_count}</p>
    <p><strong>Insight:</strong> {top_region} leads in sales, {top_state} in profit, driven by {top_cat}.</p>
    <p><strong>Average Discount:</strong> {avg_discount:.0%}</p>
    """

html = create_html_report()

b64 = base64.b64encode(html.encode()).decode()
href = f'<a href="data:text/html;base64,{b64}" download="SuperStore_Report.html">📄 Download Summary as HTML Report</a>'
st.markdown(href, unsafe_allow_html=True)

# ---------- 📌 ANALYTICAL ENHANCEMENTS ----------

# Dual-Axis Sales vs Profit Chart
st.subheader("📊 Sales vs Profit by Category (Dual-Axis)")
dual_df = filtered_df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
fig_dual = px.bar(dual_df, x="Category", y="Sales", labels={'value': 'Amount'}, title="Sales and Profit by Category")
fig_dual.add_scatter(x=dual_df["Category"], y=dual_df["Profit"], mode='lines+markers', name='Profit', yaxis="y2")
fig_dual.update_layout(
    yaxis2=dict(title="Profit", overlaying='y', side='right'),
    yaxis=dict(title="Sales")
)
st.plotly_chart(fig_dual, use_container_width=True)

# Treemap Visualization
st.subheader("🌲 Treemap of Sales by Category and Sub-Category")
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        🌲 **Treemap Visualization**:
        - Explore how categories and sub-categories contribute to total sales.
        - The larger the block, the bigger the sales share.
        """)
tree_df = filtered_df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
fig_tree = px.treemap(tree_df, path=["Category", "Sub-Category"], values="Sales", title="Sales Distribution")
st.plotly_chart(fig_tree, use_container_width=True)



# Raw Data with Column Selector
st.subheader("🧾 Raw Data Explorer with Column Selection")
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        🧾 **Raw Data Explorer**:
        - Customize which columns to view.
        - Search and filter instantly to zoom into specific records.
        """)
all_cols = list(filtered_df.columns)
selected_cols = st.multiselect("Select Columns to Display", all_cols, default=all_cols)
st.dataframe(filtered_df[selected_cols])



# Add Date Range Filter if Order Date is available
if 'Order Date' in df.columns:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Date Filter")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    date_min = df['Order Date'].min().date()
    date_max = df['Order Date'].max().date()
    date_range = st.sidebar.date_input("Select Order Date Range", (date_min, date_max))
    if len(date_range) == 2:
        filtered_df = filtered_df[(df['Order Date'].dt.date >= date_range[0]) & (df['Order Date'].dt.date <= date_range[1])]

# Enhanced Raw Data Table with search and export
st.subheader("🧾 Enhanced Data Explorer")
with st.expander("View & Export Raw Data"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to display:", options=all_columns, default=all_columns[:8])
    search_term = st.text_input("Search keyword in table:", "")
    display_df = filtered_df[selected_columns] if selected_columns else filtered_df
    if search_term:
        mask = np.column_stack([display_df[col].astype(str).str.contains(search_term, case=False, na=False) for col in display_df.columns])
        display_df = display_df.loc[mask.any(axis=1)]
    st.dataframe(display_df, height=300)

    # Export buttons
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download CSV", csv, "filtered_data.csv", "text/csv")
    with col_dl2:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            display_df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_buffer.seek(0)
        st.download_button("📊 Download Excel", excel_buffer, "filtered_data.xlsx")

# OLS Trendline for Discount vs Profit Margin
st.subheader("📈 Discount Impact on Profit Margin with Trendline")
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        📈 **Discount Impact Analysis**:
        - See how discount rates affect profit margins.
        - Trendline reveals underlying patterns.
        """)
if 'Discount' in filtered_df.columns and 'Profit Margin' in filtered_df.columns:
    fig_trend = px.scatter(filtered_df, x='Discount', y='Profit Margin', trendline='ols', color='Category',
                           title='Trend of Discount vs Profit Margin')
    st.plotly_chart(fig_trend, use_container_width=True)

# Correlation Heatmap
st.subheader("🔗 Correlation Heatmap")
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        🔗 **Correlation Heatmap**:
        - Understand relationships between Sales, Quantity, Discount, and Profit.
        - Darker colors = stronger relationships.
        """)
if {'Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin'}.issubset(filtered_df.columns):
    corr_data = filtered_df[['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin']].corr()
    fig_corr = px.imshow(corr_data, text_auto=True, title="Correlation between Key Metrics")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------- 🎓 ADVANCED BENCHMARK COMPARISONS ----------

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- Benchmark Comparison (Profit Margin by Category) ---

# Define default industry benchmarks
industry_benchmarks = {
    'Furniture': 0.18,
    'Technology': 0.25,
    'Office Supplies': 0.20
}
st.subheader("📏 Benchmark Profit Margin Comparison")
if guide_mode:
    with st.chat_message("assistant"):
        st.markdown("""
        📏 **Benchmark Comparison**:
        - Compare your actual performance against industry targets.
        - Adjust benchmarks dynamically to simulate goals.
        """)
custom_benchmarks = {}
def get_benchmark(cat):
    return st.number_input(f"Benchmark for {cat}", min_value=0.0, max_value=1.0, value=industry_benchmarks.get(cat, 0.2), step=0.01)

for cat in ['Furniture', 'Technology', 'Office Supplies']:
    custom_benchmarks[cat] = get_benchmark(cat)

if 'Category' in filtered_df.columns and 'Profit Margin' in filtered_df.columns:
    bench_df = filtered_df.groupby('Category')['Profit Margin'].mean().reset_index()
    bench_df['Benchmark'] = bench_df['Category'].map(custom_benchmarks)
    bench_df['Delta'] = bench_df['Profit Margin'] - bench_df['Benchmark']
    bench_df = bench_df.dropna()

    fig_bench = px.bar(bench_df, x='Category', y=['Profit Margin', 'Benchmark'], barmode='group',
                       title="Actual vs Benchmark Profit Margin by Category")
    st.plotly_chart(fig_bench, use_container_width=True)

# ---------- ✅ END OF ENHANCEMENTS ----------
