# Demand Forecasting & Inventory Optimization - UI Visual Assets

This directory documents the visual assets and user interface screens for **DS-01: Nationwide Demand Forecasting & Inventory Optimization**.

## Interactive Pre-rendered HTML Visualizations
Pre-rendered standalone Plotly HTML artifacts are available directly in `DS-01-Nationwide-Demand-Forecasting/docs/assets/`:
- `forecast_fan_chart.html`: 60-day historical trend, 28-day $P_{10}/P_{50}/P_{90}$ forecast prediction intervals with zero quantile crossing.
- `inventory_risk_matrix.html`: Interactive 2D scatter visualization of on-hand shelf inventory versus $P_{90}$ tail demand.

## Recommended Dashboard Captures (from Live App at `http://localhost:8050`)
When generating publication screenshots for presentations or reports:
1. `01_dashboard_header.png`: Executive title, data provenance badge, and hierarchical coherence badge.
2. `02_filter_controls.png`: Store, category, department, horizon, and model selection dropdowns.
3. `03_kpi_cards.png`: Projected demand ($P_{50}$ sum), stockout exposure count, working capital requirement, and service level targets.
4. `04_forecast_fan_chart.png`: Forecast fan chart with confidence intervals.
5. `05_inventory_risk_matrix.png`: Critical stockout risk quadrant vs balanced stock items.
6. `06_replenishment_orders.png`: Constrained purchase order recommendations table and what-if simulation sliders.

*Instructions for running the live UI:* See [docs/DEMO_GUIDE.md](../../DS-01-Nationwide-Demand-Forecasting/docs/DEMO_GUIDE.md).

