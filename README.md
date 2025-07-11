![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Visualized%20with-Plotly-blue?logo=plotly)
![Status](https://img.shields.io/badge/Live-Dashboard-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python)

# 📊 Retail Analytics Dashboard — Diagnostics & Performance Insights

An interactive analytics dashboard built using **Streamlit** and **Plotly**, designed to evaluate and explore retail sales, profit, discount trends, and regional performance using the SuperStore dataset.

---

## 🎯 Purpose

This dashboard was created specifically to explore the SuperStore dataset through interactive diagnostics, visual analytics, and scenario simulations. It serves as a modular, data-driven interface to analyze performance across sales, profit, discount behavior, and regional contributions.

The project highlights:
- Building **custom analytics dashboards** with real business context  
- Creating **repeatable evaluation workflows** using filters, visual tools, and simulations  
- Delivering **insight-driven interfaces** adaptable to decision-making settings  

---

## 🚀 Live Demo

▶️ **[Launch Dashboard](https://del-dash-ritunjay.streamlit.app/)**

---

## 🧭 How to Use the Dashboard

**Filters:**  
Use the sidebar to filter by **Region**, **Category**, **Sales Range**, and **Order Date** to customize the data view.

**Guided Tour:**  
Enable the "📖 Guided Tour" checkbox in the sidebar for a step-by-step walkthrough of all dashboard sections.

**Special Features:**
- 📊 Dual-Axis chart comparing **Sales vs Profit** by Category  
- 🎛️ What-If Simulator to predict profit under different scenarios  
- 🧾 Export filtered data to **CSV** or **Excel**  
- 🌲 Interactive Treemap, 🔗 Correlation Heatmap, and 📏 dynamic Benchmark Comparisons  

All visualizations are fully interactive and respond in real-time to your filters.

---

## 🧠 Key Features

| Feature                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| 📖 Guided Tour                 | Step-by-step assistant walking through each section                        |
| 🔍 Interactive Filters         | Region, Category, Sales Range, Date filtering                              |
| 📈 KPI Metrics                 | Sales, Profit, Order Count, Profit Margin                                  |
| 📊 Dual-Axis Chart            | Compare Sales vs Profit across Categories                                  |
| 🌍 Regional + City Drilldown  | Explore performance at geographic levels                                   |
| 🌲 Treemap View               | Hierarchical sales view by Category & Sub-Category                         |
| 🎛️ What-If Simulator         | Adjust variables to predict profit outcomes                                |
| 🧾 Raw Data + Export          | Custom column views with CSV/Excel export                                  |
| 📏 Benchmarking               | Compare real performance vs dynamic industry targets                       |
| 📈 Trendline Analysis         | Visualize discount impact using OLS regression                             |
| 🔗 Correlation Heatmap        | Understand relationships across metrics                                    |

---

## 🛠️ Tech Stack

- **Python** / **Streamlit**
- **Plotly Express**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Statsmodels**
- **Matplotlib**, **XlsxWriter**

---

## 📦 Installation

```bash
git clone https://github.com/ritunjaym/retail-analytics-dashboard.git
cd retail-analytics-dashboard
pip install -r requirements.txt
streamlit run app.py
