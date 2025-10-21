"""
Streamlit dashboard for Thailand dengue forecasting.

Interactive UI for entering climate data and visualizing forecasts.
Author: Mustafa
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

# Add parent dir to path so we can import pipeline modules
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.forecast import forecast, compute_vpd_from_temp_humidity, load_model_and_artifacts

# Configure page
st.set_page_config(
    page_title="Thailand Dengue Forecasting Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_training_data_sample():
    """Try to load some sample data from training set to prefill the grid."""
    try:
        data_path = Path("data")
        
        # Load from original CSV files in data folder
        if any(data_path.glob("*.csv")):
            # Load from original data
            csv_file = list(data_path.glob("*.csv"))[0]
            df = pd.read_csv(csv_file)
            
            # Try different date formats
            try:
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            except:
                try:
                    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                except:
                    df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values('date').tail(52)  # Last 52 weeks
            
            # Select required columns
            required_cols = ['date', 'rainfall', 'rain_days_ge1mm', 'temperature', 'humidity', 'vpd_kpa']
            available_cols = [col for col in required_cols if col in df.columns]
            return df[available_cols].copy()
        
    except Exception as e:
        st.warning(f"Could not load sample data: {str(e)}")
    
    # If no data file, just generate some random but reasonable values
    dates = pd.date_range(start='2024-01-01', periods=52, freq='W-MON')
    sample_data = {
        'date': dates,
        'rainfall': np.abs(np.random.normal(35, 20, 52)),  # Ensure positive
        'rain_days_ge1mm': np.random.randint(1, 6, 52),
        'temperature': np.random.normal(26, 2, 52).clip(20, 35),  # Reasonable range
        'humidity': np.random.normal(70, 10, 52).clip(40, 95),  # Reasonable range
        'vpd_kpa': np.abs(np.random.normal(0.8, 0.3, 52))  # Ensure positive
    }
    return pd.DataFrame(sample_data)

def validate_data_entry(df):
    """Validate the entered data."""
    errors = []
    warnings_list = []
    
    if df.empty:
        errors.append("Data cannot be empty")
        return errors, warnings_list
    
    # Check for required columns
    required_cols = ['date', 'rainfall', 'rain_days_ge1mm', 'temperature', 'humidity']
    missing_cols = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check dates are Mondays
    df['date'] = pd.to_datetime(df['date'])
    non_mondays = df[df['date'].dt.weekday != 0]
    if not non_mondays.empty:
        errors.append(f"Found {len(non_mondays)} dates that are not Mondays")
    
    # Check consecutive dates
    if len(df) > 1:
        date_diffs = df['date'].diff().dt.days
        non_weekly = date_diffs[(date_diffs != 7) & (date_diffs.notna())]
        if len(non_weekly) > 0:
            warnings_list.append(f"Found {len(non_weekly)} date gaps that are not exactly 7 days")
    
    # Check value ranges
    if 'temperature' in df.columns:
        if df['temperature'].min() < 15 or df['temperature'].max() > 40:
            warnings_list.append("Temperature values outside typical range (15-40°C)")
    
    if 'humidity' in df.columns:
        if df['humidity'].min() < 0 or df['humidity'].max() > 100:
            errors.append("Humidity must be between 0 and 100%")
    
    if 'rainfall' in df.columns:
        if df['rainfall'].min() < 0:
            errors.append("Rainfall cannot be negative")
    
    return errors, warnings_list

def calculate_kpi_metrics(df):
    """Calculate KPI metrics for the last 4 weeks."""
    if len(df) < 4:
        return None
    
    last_4_weeks = df.tail(4)
    
    kpis = {
        'weeks_covered': len(df),
        'avg_temperature': last_4_weeks['temperature'].mean(),
        'avg_humidity': last_4_weeks['humidity'].mean(),
        'total_rainfall': last_4_weeks['rainfall'].sum()
    }
    
    return kpis

def create_climate_chart(df):
    """Create modern multi-line climate chart."""
    if df.empty:
        return go.Figure()
    
    # Use last 12 weeks
    chart_data = df.tail(12).copy()
    
    # Create subplots with secondary y-axis for rainfall
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Temperature - pink/red color
    fig.add_trace(
        go.Scatter(
            x=chart_data['date'],
            y=chart_data['temperature'],
            mode='lines+markers',
            name='Temperature (°C)',
            line=dict(color='#E91E63', width=3, shape='spline', smoothing=0.3),
            marker=dict(size=7, color='#E91E63')
        ),
        secondary_y=False
    )
    
    # Humidity - blue color
    fig.add_trace(
        go.Scatter(
            x=chart_data['date'],
            y=chart_data['humidity'],
            mode='lines+markers',
            name='Humidity (%)',
            line=dict(color='#2196F3', width=3, shape='spline', smoothing=0.3),
            marker=dict(size=7, color='#2196F3')
        ),
        secondary_y=False
    )
    
    # Rainfall on secondary axis - teal/green color
    fig.add_trace(
        go.Scatter(
            x=chart_data['date'],
            y=chart_data['rainfall'],
            mode='lines+markers',
            name='Rainfall (mm)',
            line=dict(color='#26A69A', width=3, shape='spline', smoothing=0.3),
            marker=dict(size=7, color='#26A69A')
        ),
        secondary_y=True
    )
    
    # Update layout with modern styling
    fig.update_layout(
        title=dict(
            text="Climate Data (Last 12 Weeks)",
            font=dict(size=20, color='#1F2937')
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor='#E5E7EB',
            tickformat='%b %d'
        ),
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=80)
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="",
        secondary_y=False,
        showgrid=True,
        gridcolor='#E5E7EB',
        range=[0, 120]
    )
    fig.update_yaxes(
        title_text="",
        secondary_y=True,
        showgrid=False,
        range=[0, max(chart_data['rainfall'].max() * 1.2, 150)]
    )
    
    return fig

def create_forecast_chart(forecast_df, show_95ci=True):
    """Create modern forecast visualization with confidence intervals."""
    if forecast_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add 95% CI shaded area FIRST (so it's in the background)
    if show_95ci and 'hi95' in forecast_df.columns and 'lo95' in forecast_df.columns:
        # Create the shaded area
        fig.add_trace(go.Scatter(
            x=list(forecast_df['week_start']) + list(forecast_df['week_start'][::-1]),
            y=list(forecast_df['hi95']) + list(forecast_df['lo95'][::-1]),
            fill='toself',
            fillcolor='rgba(244, 143, 177, 0.3)',
            line=dict(color='rgba(244, 143, 177, 0.5)', width=0.5),
            name='95% Confidence Interval',
            showlegend=True,
            hoverinfo="skip",
            legendrank=2
        ))
        
        # Add upper 95% CI line
        fig.add_trace(go.Scatter(
            x=forecast_df['week_start'],
            y=forecast_df['hi95'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='#F48FB1', width=2, dash='dot'),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Add lower 95% CI line
        fig.add_trace(go.Scatter(
            x=forecast_df['week_start'],
            y=forecast_df['lo95'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='#F48FB1', width=2, dash='dot'),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # Add forecast line - pink/magenta (on TOP of CI)
    # Prepare custom hover text with CI information
    hover_text = []
    for idx, row in forecast_df.iterrows():
        date_str = row['week_start'].strftime('%b %d, %Y')
        pred = int(row['y_pred'])
        
        # Build hover text with CI if available and valid
        hover = f"<b>{date_str}</b><br>Predicted Cases: {pred:,}"
        
        if ('lo95' in forecast_df.columns and 'hi95' in forecast_df.columns and 
            pd.notna(row['lo95']) and pd.notna(row['hi95'])):
            lo = int(row['lo95'])
            hi = int(row['hi95'])
            hover += f"<br>95% CI: {lo:,} - {hi:,}"
        
        hover += "<br><extra></extra>"
        hover_text.append(hover)
    
    fig.add_trace(go.Scatter(
        x=forecast_df['week_start'],
        y=forecast_df['y_pred'],
        mode='lines+markers',
        name='Predicted Cases',
        line=dict(color='#E91E63', width=3, shape='spline', smoothing=0.3),
        marker=dict(size=8, color='#E91E63'),
        hovertemplate=hover_text,
        legendrank=1
    ))
    
    fig.update_layout(
        title=dict(
            text="Forecast (Weekly Predictions)",
            font=dict(size=20, color='#1F2937')
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor='#E5E7EB',
            tickformat='%b %d',
            tickfont=dict(size=11, color='#374151'),
            showline=True,
            linewidth=1,
            linecolor='#D1D5DB'
        ),
        yaxis=dict(
            title="Cases",
            title_font=dict(size=14, color='#6B7280'),
            showgrid=True,
            gridcolor='#E5E7EB'
        ),
        height=400,
        hovermode='x',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=80)
    )
    
    return fig

def get_risk_tier(predicted_cases):
    """Determine risk tier based on historical percentiles."""
    # These thresholds are approximations - in a real system, 
    # you'd calculate them from historical data
    try:
        # Try to load historical data to calculate percentiles
        data_path = Path("data")
        if any(data_path.glob("*.csv")):
            csv_file = list(data_path.glob("*.csv"))[0]
            df = pd.read_csv(csv_file)
            
            p50 = df['dengue_cases'].quantile(0.5)
            p75 = df['dengue_cases'].quantile(0.75)
        else:
            # Fallback thresholds
            p50 = 800
            p75 = 1500
    except:
        # Fallback thresholds
        p50 = 800
        p75 = 1500
    
    if predicted_cases < p50:
        return "Low", "green"
    elif predicted_cases < p75:
        return "Moderate", "orange"
    else:
        return "High", "red"

def generate_analysis_summary(climate_df, forecast_df):
    """Generate automatic analysis summary."""
    if len(climate_df) < 4:
        return "Insufficient data for analysis."
    
    # Analyze last 4-8 weeks
    recent_weeks = climate_df.tail(8)
    last_4_weeks = climate_df.tail(4)
    
    avg_rainfall = last_4_weeks['rainfall'].mean()
    avg_humidity = last_4_weeks['humidity'].mean()
    avg_temp = last_4_weeks['temperature'].mean()
    
    # Generate summary
    summary_parts = []
    
    # Rainfall assessment
    if avg_rainfall < 20:
        rainfall_desc = "Low rainfall"
    elif avg_rainfall < 50:
        rainfall_desc = "Moderate rainfall"
    else:
        rainfall_desc = "High rainfall"
    
    # Humidity assessment
    if avg_humidity < 60:
        humidity_desc = "low humidity"
    elif avg_humidity < 75:
        humidity_desc = "moderate humidity"
    else:
        humidity_desc = "high humidity"
    
    # Temperature assessment
    if avg_temp < 24:
        temp_desc = "cooler temperatures"
    elif avg_temp < 28:
        temp_desc = "moderate temperatures"
    else:
        temp_desc = "warm temperatures"
    
    summary_parts.append(f"{rainfall_desc} (~{avg_rainfall:.1f} mm/week) and {humidity_desc} (~{avg_humidity:.1f}%) with {temp_desc} (~{avg_temp:.1f}°C) over the past 4 weeks.")
    
    # Risk assessment based on forecast
    if not forecast_df.empty:
        avg_forecast = forecast_df['y_pred'].mean()
        risk_tier, _ = get_risk_tier(avg_forecast)
        summary_parts.append(f"Transmission risk: {risk_tier.lower()}.")
    
    return " ".join(summary_parts)

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🦟 Thailand Dengue Forecasting Dashboard")
    st.markdown("### Interactive dengue transmission forecasting using climate data")
    
    # Introductory description
    st.markdown("""
    This dashboard helps you forecast dengue cases in Thailand based on climate conditions. 
    
    **How it works:**
    1. **Enter Climate Data**: Input temperature, humidity, and rainfall for recent weeks
    2. **Generate Forecast**: The model predicts dengue cases for upcoming weeks
    3. **View Results**: See predictions with confidence intervals and risk levels
    
    The forecasting system uses machine learning trained on historical dengue and climate patterns to provide reliable predictions.
    """)
    
    st.divider()
    
    # Check if model exists
    try:
        load_model_and_artifacts()
        model_available = True
    except Exception as e:
        st.error(f"⚠️ Model not available: {str(e)}")
        st.info("Please run the training pipeline first: `python -m pipeline.train`")
        model_available = False
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Forecast Settings")
        
        # Forecast horizon
        horizon_weeks = st.slider(
            "Forecast Horizon (weeks)",
            min_value=1,
            max_value=12,
            value=4,
            help="Number of weeks to forecast ahead"
        )
        
        # Confidence interval toggle
        show_95ci = st.checkbox("Show 95% Confidence Interval", value=True)
        
        st.divider()
        
        # Data entry weeks selector
        st.header("📝 Data Entry")
        
        # Date range for data entry
        st.markdown("**When does your data start?**")
        data_start_date = st.date_input(
            "Week 1 starts on (Monday)",
            value=pd.Timestamp.now().normalize() - pd.Timedelta(weeks=12),
            help="Select the Monday when your first week of data begins"
        )
        
        weeks_to_enter = st.number_input(
            "Weeks to enter",
            min_value=12,
            max_value=52,
            value=12,
            step=1,
            help="Minimum 12 weeks of data required for accurate forecasts"
        )
    
    # Load sample data for prefilling
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = load_training_data_sample()
    
    # Data entry section
    st.header("📊 Data Entry")
    
    st.markdown("""
    Enter the climate conditions for each week. The model needs recent weather data to make accurate predictions.
    
    - **Rainfall**: Total rainfall in millimeters for the week
    - **Rain Days**: Number of days with measurable rain (≥1mm)
    - **Temperature**: Average temperature in Celsius
    - **Humidity**: Average relative humidity percentage
    """)
    
    # Info box with value ranges
    st.info("💡 **Value Ranges:** Rainfall: 0-200mm | Rain Days: 0-7 | Temperature: 15-40°C | Humidity: 0-100%")
    
    # Prepare data for editing without dates
    edit_data = st.session_state.sample_data.tail(int(weeks_to_enter)).copy()
    
    # Generate dates based on user's start date
    start_date = pd.Timestamp(data_start_date)
    # Ensure it's a Monday
    if start_date.dayofweek != 0:
        start_date = start_date - pd.Timedelta(days=start_date.dayofweek)
    
    # Create date range for the weeks
    dates = [start_date + pd.Timedelta(weeks=i) for i in range(int(weeks_to_enter))]
    edit_data['date'] = dates[:len(edit_data)]
    
    # Create simplified data structure with week numbers
    simplified_data = pd.DataFrame({
        'Week': [f'Week {i+1}' for i in range(len(edit_data))],
        'Rainfall (mm)': edit_data['rainfall'].values,
        'Rain Days (≥1mm)': edit_data['rain_days_ge1mm'].values,
        'Temperature (°C)': edit_data['temperature'].values,
        'Humidity (%)': edit_data['humidity'].values
    })
    
    # Data editor with validation
    edited_simple = st.data_editor(
        simplified_data,
        column_config={
            "Week": st.column_config.TextColumn(
                "Week",
                help="Week identifier",
                disabled=True,
                width="small"
            ),
            "Rainfall (mm)": st.column_config.NumberColumn(
                "Rainfall (mm)",
                help="Weekly rainfall: 0-200mm",
                min_value=0.0,
                max_value=200.0,
                format="%.1f",
                required=True
            ),
            "Rain Days (≥1mm)": st.column_config.NumberColumn(
                "Rain Days (≥1mm)",
                help="Days with rainfall ≥1mm: 0-7",
                min_value=0,
                max_value=7,
                step=1,
                required=True
            ),
            "Temperature (°C)": st.column_config.NumberColumn(
                "Temperature (°C)",
                help="Average temperature: 15-40°C",
                min_value=15.0,
                max_value=40.0,
                format="%.1f",
                required=True
            ),
            "Humidity (%)": st.column_config.NumberColumn(
                "Humidity (%)",
                help="Relative humidity: 0-100%",
                min_value=0.0,
                max_value=100.0,
                format="%.1f",
                required=True
            )
        },
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key="data_editor"
    )
    
    # Convert back to required format with dates
    edited_data = edit_data.copy()
    edited_data['rainfall'] = edited_simple['Rainfall (mm)'].values
    edited_data['rain_days_ge1mm'] = edited_simple['Rain Days (≥1mm)'].values
    edited_data['temperature'] = edited_simple['Temperature (°C)'].values
    edited_data['humidity'] = edited_simple['Humidity (%)'].values
    
    # Auto-compute VPD
    edited_data['vpd_kpa'] = compute_vpd_from_temp_humidity(
        edited_data['temperature'], 
        edited_data['humidity']
    )
    
    # Validate data
    errors, warnings_list = validate_data_entry(edited_data)
    
    if errors:
        st.error("❌ Data validation errors:")
        for error in errors:
            st.error(f"• {error}")
    
    if warnings_list:
        st.warning("⚠️ Data validation warnings:")
        for warning in warnings_list:
            st.warning(f"• {warning}")
    
    # Only proceed if no errors
    if not errors:
        # Convert dates back to datetime
        edited_data['date'] = pd.to_datetime(edited_data['date'])
        
        # KPI metrics
        kpis = calculate_kpi_metrics(edited_data)
        if kpis:
            st.header("📈 Summary Metrics (Last 4 Weeks)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Weeks Covered", f"{kpis['weeks_covered']}")
            
            with col2:
                st.metric("Avg Temperature", f"{kpis['avg_temperature']:.1f}°C")
            
            with col3:
                st.metric("Avg Humidity", f"{kpis['avg_humidity']:.1f}%")
            
            with col4:
                st.metric("Total Rainfall", f"{kpis['total_rainfall']:.1f}mm")
        
        # Climate chart
        st.header("🌡️ Climate Overview")
        climate_fig = create_climate_chart(edited_data)
        st.plotly_chart(climate_fig, use_container_width=True)
        
        # Generate forecast
        try:
            with st.spinner("Generating forecast..."):
                # Get last dengue cases if available
                last_dengue = None
                try:
                    # Try to get from training data
                    data_path = Path("data")
                    if any(data_path.glob("*.csv")):
                        csv_file = list(data_path.glob("*.csv"))[0]
                        df = pd.read_csv(csv_file)
                        if 'dengue_cases' in df.columns:
                            last_dengue = df['dengue_cases'].iloc[-1]
                except:
                    pass
                
                # Generate forecast
                forecast_df = forecast(horizon_weeks, edited_data, last_dengue)
            
            # Forecast visualization
            st.header("🔮 Forecast Results")
            
            st.markdown("""
            The chart below shows predicted dengue cases for upcoming weeks based on your climate data.
            
            - **Predicted Cases** (dark pink line): The model's best estimate for each week
            - **95% Confidence Interval** (pink shaded area): Range where we're 95% confident the actual cases will fall
            - **Risk Tiers**: Classification based on historical dengue patterns (Low 🟢, Moderate 🟡, High 🔴)
            
            Higher temperatures and humidity combined with rainfall often increase dengue transmission risk.
            """)
            
            forecast_fig = create_forecast_chart(forecast_df, show_95ci)
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Detailed forecast table
            st.subheader("Detailed Forecast")
            
            # Prepare table data
            table_data = forecast_df.copy()
            table_data['Week Start'] = table_data['week_start'].dt.strftime('%b %d, %Y')
            table_data['Predicted Cases'] = table_data['y_pred'].round().astype(int)
            table_data['95% CI'] = table_data.apply(
                lambda x: f"{int(x['lo95'])} - {int(x['hi95'])}", axis=1
            )
            
            # Add risk tiers
            risk_info = table_data['y_pred'].apply(get_risk_tier)
            table_data['Risk Tier'] = [info[0] for info in risk_info]
            
            # Display table with modern styling
            display_cols = ['Week Start', 'Predicted Cases', '95% CI', 'Risk Tier']
            
            # Create a more visual table display using styled dataframe
            styled_table = table_data[display_cols].copy()
            
            # Apply styling based on risk tier using emojis for better visual appeal
            def style_risk(val):
                if val == 'Low':
                    return '🟢 Low'
                elif val == 'Moderate':
                    return '🟡 Moderate'
                else:
                    return '🔴 High'
            
            styled_table['Risk Tier'] = styled_table['Risk Tier'].apply(style_risk)
            
            st.dataframe(
                styled_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Week Start": st.column_config.TextColumn(
                        "Week Start",
                        help="Start date of the forecast week",
                        width="medium"
                    ),
                    "Predicted Cases": st.column_config.NumberColumn(
                        "Predicted Cases",
                        help="Predicted dengue cases for the week",
                        width="medium"
                    ),
                    "95% CI": st.column_config.TextColumn(
                        "95% CI",
                        help="95% confidence interval",
                        width="medium"
                    ),
                    "Risk Tier": st.column_config.TextColumn(
                        "Risk Tier",
                        help="Risk level based on historical percentiles",
                        width="medium"
                    )
                }
            )
            
            # Analysis summary
            st.subheader("📝 Analysis Summary")
            summary = generate_analysis_summary(edited_data, forecast_df)
            st.info(summary)
            
        except Exception as e:
            st.error(f"❌ Forecast generation failed: {str(e)}")
            st.info("Please check your data and try again.")
    
    # Footer at bottom of content
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #6B7280; padding: 20px 0;">
            Developed by <strong>Mohamed Mustaf Ahmed</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
