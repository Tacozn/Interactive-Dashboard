import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, MATCH, ALL
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import base64
import io
import os

# --- GOOGLE FONTS (Inter) ---
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
FONT_FAMILY = "'Inter', 'Segoe UI', Arial, sans-serif"

# Load and clean data
def load_data(filepath):
    """Load and clean the electricity consumption data"""
    try:
        df = pd.read_csv(filepath)
        
        # Clean column names - remove extra spaces and standardize
        df.columns = df.columns.str.strip()
        
        # Map to shorter, cleaner column names
        column_mapping = {
            'Local consumption Industrial commercial and mining in Million kilowatt hours': 'commercial_industrial',
            'Local consumption Domestic and public lighting in Million kilowatt hours': 'domestic',
            'Exports in Million kilowatt hours': 'exports',
            'Losses in Million kilowatt hours': 'losses'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Create proper date column
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01')
        
        # Clean numeric columns - handle commas and convert to numeric
        numeric_cols = ['commercial_industrial', 'domestic', 'exports', 'losses']
        for col in numeric_cols:
            if col in df.columns:
                # Handle both string and numeric data
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Remove rows with missing critical data
        df = df.dropna(subset=['Date', 'commercial_industrial', 'domestic'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data
df = load_data("electric_dataset.csv")


if df is None:
    print("Failed to load data. Please check the file path and format.")
    exit()

# Use a dark Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, FONT_URL])

# Gradient background style
BG_GRADIENT = {
    "background": "linear-gradient(135deg, #232b3e 0%, #243b55 100%)",
    "minHeight": "100vh",
    "padding": 0,
    "fontFamily": FONT_FAMILY
}
CARD_BG = "#283655"
ACCENT = "#00eaff"

# Add theme support
THEMES = {
    'dark': dbc.themes.CYBORG,
    'light': dbc.themes.FLATLY
}

# --- SIDEBAR ---
sidebar = html.Div([
    html.H3("FUSION SMART", className="mb-0", style={"color": ACCENT, "fontWeight": "bold", "letterSpacing": "2px", "fontFamily": FONT_FAMILY}),
    html.Small("BETA", style={"color": "#ff4c6d", "fontWeight": "bold", "marginLeft": "4px", "fontFamily": FONT_FAMILY}),
    html.Hr(style={"borderColor": ACCENT}),
    dbc.Nav([
        dbc.NavLink([
            DashIconify(icon="mdi:view-dashboard-outline", width=24, style={"marginRight": "10px"}),
            "Dashboard"
        ], href="#", active=True, style={"color": "#fff", "fontFamily": FONT_FAMILY}),
    ], vertical=True, pills=True, className="mb-4"),
    html.Label("Date Range", style={"color": ACCENT, "fontWeight": "bold", "fontFamily": FONT_FAMILY}),
    dcc.DatePickerRange(
        id='date-range-picker',
        start_date=df['Date'].min(),
        end_date=df['Date'].max(),
        min_date_allowed=df['Date'].min(),
        max_date_allowed=df['Date'].max(),
        display_format='MMM YYYY',
        style={"marginBottom": "1.5rem", "backgroundColor": CARD_BG, "fontFamily": FONT_FAMILY}
    ),
    # Sector visibility toggle
    html.Label("Show Sectors", style={"color": ACCENT, "fontWeight": "bold", "fontFamily": FONT_FAMILY}),
    dcc.Checklist(
        id='sector-visibility-checklist',
        options=[
            {'label': 'Commercial & Industrial', 'value': 'commercial_industrial'},
            {'label': 'Domestic & Public', 'value': 'domestic'}
        ],
        value=['commercial_industrial', 'domestic'],
        inline=False,
        style={"marginBottom": "1.5rem", "color": "#fff", "fontFamily": FONT_FAMILY}
    ),
    dbc.Button("Reset Filters", id="reset-filters-btn", color="warning", className="mb-2 w-100", style={"fontWeight": "bold", "fontFamily": FONT_FAMILY}),
    dbc.Button("Download Data", id="download-btn", color="info", className="mb-2 w-100", style={"fontWeight": "bold", "fontFamily": FONT_FAMILY}),
    dcc.Download(id="download-dataframe-csv"),
    dbc.Button("Download Summary", id="download-summary-btn", color="secondary", className="mb-2 w-100", style={"fontWeight": "bold", "fontFamily": FONT_FAMILY}),
    dcc.Download(id="download-summary-csv"),
    html.Hr(style={"backgroundColor": ACCENT}),
    html.Div("Malaysia Energy Insights", style={"color": ACCENT, "marginTop": "auto", "fontWeight": "bold", "fontSize": "1.1rem", "fontFamily": FONT_FAMILY}),
], style={
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem 1rem 1.5rem",
    "background": "#1a2235",
    "zIndex": 1000,
    "display": "flex",
    "flexDirection": "column",
    "fontFamily": FONT_FAMILY
})

# --- TOP BAR ---
def topbar():
    return dbc.Row([
        dbc.Col(html.H4("Energy Dashboard", style={"color": ACCENT, "fontWeight": "bold", "fontFamily": FONT_FAMILY}), md=8),
        dbc.Col(html.Div("2018-2023", style={"color": "#fff", "textAlign": "right", "fontFamily": FONT_FAMILY}), md=4)
    ], align="center", className="mb-4")

# --- SUMMARY CARDS GRID ---
def summary_cards_grid(stats, trend_stats):
    icons = {
        'commercial_industrial': "mdi:factory",
        'domestic': "mdi:home-city",
        'exports': "mdi:export",
        'losses': "mdi:transmission-tower"
    }
    badges = {
        'commercial_industrial': "danger",
        'domestic': "success",
        'exports': "info",
        'losses': "warning"
    }
    colors = {
        'commercial_industrial': '#00eaff',
        'domestic': '#ff4c6d',
        'exports': '#FFD369',
        'losses': '#4ecdc4'
    }
    # Mini-cards for each sector
    sector_cards = []
    for sector in ['commercial_industrial', 'domestic', 'exports', 'losses']:
        pct_icon = "mdi:arrow-up-bold" if stats[sector]['pct_change'] >= 0 else "mdi:arrow-down-bold"
        pct_color = colors[sector] if stats[sector]['pct_change'] >= 0 else '#ff4c6d'
        sector_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        DashIconify(icon=icons[sector], width=28, style={"color": colors[sector], "marginRight": "8px"}),
                        dbc.Badge(sector.replace('_', ' ').title(), color=badges[sector], className="me-2", style={"fontFamily": FONT_FAMILY, "fontSize": "1rem"}),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"}),
                    html.H2(f"{stats[sector]['total']:,.0f}", style={"color": colors[sector], "fontWeight": "bold", "fontFamily": FONT_FAMILY, "marginBottom": "0.2rem"}),
                    html.Div("Total (MkWh)", style={"color": "#fff", "fontSize": "0.95rem", "fontFamily": FONT_FAMILY}),
                    html.Hr(style={"backgroundColor": colors[sector], "margin": "0.5rem 0"}),
                    html.Div([
                        html.Span(f"Avg: {stats[sector]['avg']:,.1f}", style={"marginRight": "1.2rem", "color": "#fff", "fontSize": "0.95rem"}),
                        html.Span(f"Max: {stats[sector]['max']:,.0f}", style={"marginRight": "1.2rem", "color": "#fff", "fontSize": "0.95rem"}),
                        html.Span(f"Min: {stats[sector]['min']:,.0f}", style={"marginRight": "1.2rem", "color": "#fff", "fontSize": "0.95rem"}),
                        DashIconify(icon=pct_icon, width=18, style={"color": pct_color, "verticalAlign": "middle"}),
                        html.Span(f" {stats[sector]['pct_change']:+.1f}%", style={"color": pct_color, "fontWeight": "bold", "fontSize": "0.95rem"})
                    ], style={"marginTop": "0.2rem"})
                ])
            ], style={"background": CARD_BG, "border": "none", "borderRadius": "16px", "boxShadow": "0 2px 12px #111", "marginBottom": "1.5rem", "fontFamily": FONT_FAMILY})
        )
    # 2x2 grid for sector cards
    grid = dbc.Row([
        dbc.Col(sector_cards[0], md=3, xs=12),
        dbc.Col(sector_cards[1], md=3, xs=12),
        dbc.Col(sector_cards[2], md=3, xs=12),
        dbc.Col(sector_cards[3], md=3, xs=12),
    ], className="g-3 mb-2")
    # Grand total card
    grand_card = dbc.Card([
        dbc.CardBody([
            html.Div([
                DashIconify(icon="mdi:chart-donut", width=28, style={"color": ACCENT, "marginRight": "10px"}),
                html.Span("Grand Total", style={"color": ACCENT, "fontWeight": "bold", "fontSize": "1.2rem", "fontFamily": FONT_FAMILY})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"}),
            html.H2(f"{stats['grand_total']:,.0f}", style={"color": ACCENT, "fontWeight": "bold", "fontFamily": FONT_FAMILY, "marginBottom": "0.2rem"}),
            html.Div("Total Consumption (MkWh)", style={"color": "#fff", "fontSize": "0.95rem", "fontFamily": FONT_FAMILY}),
            html.Hr(style={"backgroundColor": ACCENT, "margin": "0.5rem 0"}),
            html.Div([
                html.Span(f"Exports/Total: {stats['exports_ratio']:.1%}", style={"marginRight": "2rem", "color": "#FFD369", "fontWeight": "bold"}),
                html.Span(f"Losses/Total: {stats['losses_ratio']:.1%}", style={"color": "#4ecdc4", "fontWeight": "bold"})
            ], style={"fontSize": "1rem"})
        ])
    ], style={"background": CARD_BG, "border": "none", "borderRadius": "16px", "boxShadow": "0 2px 12px #111", "marginBottom": "1.5rem", "fontFamily": FONT_FAMILY})
    # Trend highlights card
    trend_card_data = {
        'highest_consumption': {'title': 'Highest Consumption', 'icon': 'mdi:trending-up', 'color': '#4ecdc4', 'value': trend_stats['highest_consumption']['value'], 'date': trend_stats['highest_consumption']['date']},
        'lowest_consumption': {'title': 'Lowest Consumption', 'icon': 'mdi:trending-down', 'color': '#ff4c6d', 'value': trend_stats['lowest_consumption']['value'], 'date': trend_stats['lowest_consumption']['date']},
        'largest_increase': {'title': 'Largest Increase (MoM)', 'icon': 'mdi:arrow-top-right-thick', 'color': '#00eaff', 'value': trend_stats['largest_increase']['value'], 'date': trend_stats['largest_increase']['date']},
        'largest_decrease': {'title': 'Largest Decrease (MoM)', 'icon': 'mdi:arrow-bottom-right-thick', 'color': '#FFD369', 'value': trend_stats['largest_decrease']['value'], 'date': trend_stats['largest_decrease']['date']}
    }
    
    trend_cards = []
    for key, data in trend_card_data.items():
        trend_cards.append(dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div([
                    DashIconify(icon=data['icon'], width=24, style={"color": data['color'], "marginRight": "8px"}),
                    html.Span(data['title'], style={"color": "#fff", "fontWeight": "bold", "fontSize": "0.9rem"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.4rem"}),
                html.H4(f"{data['value']:,.0f}", style={"color": data['color'], "fontWeight": "bold", "margin": "0.2rem 0"}),
                html.Div("MkWh", style={"color": "#fff", "fontSize": "0.8rem", "marginBottom": "0.4rem"}),
                html.Div(f"{data['date']}", style={"color": "#fff", "fontSize": "0.9rem", "fontWeight": "bold"})
            ])
        ], style={"background": CARD_BG, "border": "none", "borderRadius": "16px", "height": "100%"}), md=6, xs=12, className="mb-2"))

    # Return the grid and cards
    return html.Div([
        grid,
        dbc.Row([
            dbc.Col(grand_card, md=6, xs=12),
            dbc.Col(html.Div([
                html.Div([
                    DashIconify(icon="mdi:star-outline", width=24, style={"color": ACCENT, "marginRight": "10px"}),
                    html.Span("Monthly Trend Highlights", style={"color": ACCENT, "fontWeight": "bold", "fontSize": "1.1rem"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "1rem"}),
                dbc.Row(trend_cards)
            ]), md=6, xs=12)
        ], className="g-3")
    ])

# --- PIE CHART CARD ---
def pie_chart_card(stats):
    values = [stats[sector]['total'] for sector in ['commercial_industrial', 'domestic']]
    labels = ['Commercial & Industrial', 'Domestic & Public']
    colors = ['#00eaff', '#ff4c6d']
    fig = go.Figure(go.Pie(
        values=values,
        labels=labels,
        marker_colors=colors,
        hole=0.5,
        textinfo='label+percent',
        pull=[0.05 if v == max(values) else 0 for v in values]
    ))
    fig.update_layout(
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="#fff", family=FONT_FAMILY),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return dbc.Card([
        dbc.CardHeader([
            DashIconify(icon="mdi:chart-donut", width=28, style={"color": ACCENT, "marginRight": "10px"}),
            html.Span("Sector Share (Pie Chart)", style={"color": ACCENT, "fontWeight": "bold", "fontSize": "1.2rem", "fontFamily": FONT_FAMILY})
        ], style={"background": "none", "border": "none"}),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "320px"})
        ])
    ], style={"background": CARD_BG, "border": "none", "borderRadius": "18px", "boxShadow": "0 2px 12px #111", "marginBottom": "2rem", "fontFamily": FONT_FAMILY})

# --- YEAR-OVER-YEAR CARD ---
def yoy_card(filtered_df):
    # Only show if at least 2 years in range
    if filtered_df['Date'].dt.year.nunique() < 2:
        return dbc.Card([
            dbc.CardHeader([
                DashIconify(icon="mdi:calendar-range", width=28, style={"color": ACCENT, "marginRight": "10px"}),
                html.Span("Year-over-Year Comparison", style={"color": ACCENT, "fontWeight": "bold", "fontSize": "1.2rem", "fontFamily": FONT_FAMILY})
            ], style={"background": "none", "border": "none"}),
            dbc.CardBody([
                html.Div("Not enough years in selected range for comparison.", style={"color": "#fff", "fontFamily": FONT_FAMILY})
            ])
        ], style={"background": CARD_BG, "border": "none", "borderRadius": "18px", "boxShadow": "0 2px 12px #111", "marginBottom": "2rem", "fontFamily": FONT_FAMILY})
    # Compute totals and percent change for each year
    df_year = filtered_df.copy()
    df_year['Year'] = df_year['Date'].dt.year
    yearly = df_year.groupby('Year')[['commercial_industrial', 'domestic']].sum()
    yearly['Total'] = yearly['commercial_industrial'] + yearly['domestic']
    yearly['Pct Change'] = yearly['Total'].pct_change() * 100
    rows = []
    for idx, row in yearly.iterrows():
        rows.append(html.Tr([
            html.Td(str(idx), style={"fontFamily": FONT_FAMILY}),
            html.Td(f"{row['Total']:,.0f}", style={"fontFamily": FONT_FAMILY}),
            html.Td(f"{row['Pct Change']:+.1f}%" if not pd.isnull(row['Pct Change']) else '-', style={"fontFamily": FONT_FAMILY})
        ]))
    return dbc.Card([
        dbc.CardHeader([
            DashIconify(icon="mdi:calendar-range", width=28, style={"color": ACCENT, "marginRight": "10px"}),
            html.Span("Year-over-Year Comparison", style={"color": ACCENT, "fontWeight": "bold", "fontSize": "1.2rem", "fontFamily": FONT_FAMILY})
        ], style={"background": "none", "border": "none"}),
        dbc.CardBody([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Year", style={"fontFamily": FONT_FAMILY}),
                        html.Th("Total Consumption (MkWh)", style={"fontFamily": FONT_FAMILY}),
                        html.Th("% Change", style={"fontFamily": FONT_FAMILY})
                    ])
                ]),
                html.Tbody(rows)
            ], style={"width": "100%", "color": "#fff", "fontFamily": FONT_FAMILY, "fontSize": "1.1rem"})
        ])
    ], style={"background": CARD_BG, "border": "none", "borderRadius": "18px", "boxShadow": "0 2px 12px #111", "marginBottom": "2rem", "fontFamily": FONT_FAMILY})

# --- CHART CARDS ---
def chart_card(title, fig, icon, accent=ACCENT, chart_id=None):
    # Add a download button for the chart
    download_btn = dbc.Button(
        [DashIconify(icon="mdi:download", width=20, style={"marginRight": "6px"}), "Download Chart"],
        id={"type": "download-chart-btn", "index": chart_id or title},
        color="secondary",
        outline=True,
        size="sm",
        style={"float": "right", "marginBottom": "0.5rem", "fontFamily": FONT_FAMILY}
    ) if chart_id else None
    graph_id = chart_id if chart_id is not None else title.replace(' ', '_').lower()
    return dbc.Card([
        dbc.CardHeader([
            DashIconify(icon=icon, width=28, style={"color": accent, "marginRight": "10px"}),
            html.Span(title, style={"color": accent, "fontWeight": "bold", "fontSize": "1.2rem", "fontFamily": FONT_FAMILY}),
            download_btn
        ], style={"background": "none", "border": "none", "display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={"displayModeBar": False, "toImageButtonOptions": {"format": "png", "filename": title.replace(' ', '_').lower()}}, id=graph_id, style={"height": "320px"})
        ])
    ], style={"background": CARD_BG, "border": "none", "borderRadius": "18px", "boxShadow": "0 2px 12px #111", "marginBottom": "2rem", "fontFamily": FONT_FAMILY})

# --- APP LAYOUT ---
app.layout = html.Div([
    sidebar,
    html.Div([
        html.Div(topbar(), style={"marginLeft": "1rem", "marginRight": "1rem"}),
        dbc.Row([
            dbc.Col(id="summary-stats-col", md=12),
        ], style={"marginLeft": "1rem", "marginRight": "1rem"}),
        dbc.Row([
            dbc.Col(id="pie-chart-col", md=4),
            dbc.Col(id="consumption-chart-col", md=4),
            dbc.Col(id="exports-chart-col", md=4),
        ], style={"marginLeft": "1rem", "marginRight": "1rem"}),
        dbc.Row([
            dbc.Col(id="losses-chart-col", md=4),
            dbc.Col(id="yoy-card-col", md=8),
        ], style={"marginLeft": "1rem", "marginRight": "1rem"}),
    ], style={"marginLeft": "19rem", **BG_GRADIENT, "fontFamily": FONT_FAMILY})
])

# --- CALLBACKS ---
@app.callback(
    Output('date-range-picker', 'start_date'),
    Output('date-range-picker', 'end_date'),
    Output('sector-visibility-checklist', 'value'),
    Input('reset-filters-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks_reset):
    """Reset all filters to default values"""
    if n_clicks_reset:
        return df['Date'].min(), df['Date'].max(), ['commercial_industrial', 'domestic']
    return df['Date'].min(), df['Date'].max(), ['commercial_industrial', 'domestic']

@app.callback(
    Output('consumption-chart-col', 'children'),
    Output('exports-chart-col', 'children'),
    Output('losses-chart-col', 'children'),
    Output('summary-stats-col', 'children'),
    Output('pie-chart-col', 'children'),
    Output('yoy-card-col', 'children'),
    Output('download-dataframe-csv', 'data'),
    Output('download-summary-csv', 'data'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input('download-btn', 'n_clicks'),
    Input('download-summary-btn', 'n_clicks'),
    Input('sector-visibility-checklist', 'value'),
    prevent_initial_call=False
)
def update_dashboard(start_date, end_date, n_clicks_data, n_clicks_summary, selected_sectors):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()

    # --- Compute statistics for each sector ---
    stats = {}
    for sector in ['commercial_industrial', 'domestic', 'exports', 'losses']:
        if sector in filtered_df.columns:
            total = filtered_df[sector].sum()
            avg = filtered_df[sector].mean()
            max_val = filtered_df[sector].max()
            min_val = filtered_df[sector].min()
            max_date = filtered_df.loc[filtered_df[sector].idxmax(), 'Date'].strftime('%b %Y') if not filtered_df[sector].isnull().all() else '-'
            min_date = filtered_df.loc[filtered_df[sector].idxmin(), 'Date'].strftime('%b %Y') if not filtered_df[sector].isnull().all() else '-'
            first = filtered_df[sector].iloc[0] if not filtered_df[sector].empty else 0
            last = filtered_df[sector].iloc[-1] if not filtered_df[sector].empty else 0
            pct_change = ((last - first) / first * 100) if first else 0
            stats[sector] = {
                'total': total,
                'avg': avg,
                'max': max_val,
                'min': min_val,
                'max_date': max_date,
                'min_date': min_date,
                'pct_change': pct_change
            }
        else:
            stats[sector] = {'total': 0, 'avg': 0, 'max': 0, 'min': 0, 'max_date': '-', 'min_date': '-', 'pct_change': 0}
    # Grand total and ratios
    grand_total = sum(stats[sector]['total'] for sector in stats)
    exports_ratio = stats['exports']['total'] / grand_total if grand_total else 0
    losses_ratio = stats['losses']['total'] / grand_total if grand_total else 0
    stats['grand_total'] = grand_total
    stats['exports_ratio'] = exports_ratio
    stats['losses_ratio'] = losses_ratio

    # --- Tooltips for summary ---
    tooltips = {
        'total': 'Sum of all values for the selected period.',
        'avg': 'Average per month for the selected period.',
        'max': 'Maximum monthly value and when it occurred.',
        'min': 'Minimum monthly value and when it occurred.',
        'pct_change': 'Percent change from first to last month in the selected period.'
    }

    # --- Monthly Trend Highlights ---
    filtered_df['Total_Consumption'] = filtered_df['commercial_industrial'] + filtered_df['domestic']
    trend_stats = {}
    if not filtered_df.empty and filtered_df['Total_Consumption'].notna().any():
        max_month_row = filtered_df.loc[filtered_df['Total_Consumption'].idxmax()]
        min_month_row = filtered_df.loc[filtered_df['Total_Consumption'].idxmin()]
        diffs = filtered_df['Total_Consumption'].diff().fillna(0)
        max_increase = diffs.max()
        max_decrease = diffs.min()
        max_increase_month = filtered_df.iloc[diffs.idxmax()]['Date'].strftime('%b %Y') if diffs.idxmax() in filtered_df.index and max_increase > 0 else '-'
        max_decrease_month = filtered_df.iloc[diffs.idxmin()]['Date'].strftime('%b %Y') if diffs.idxmin() in filtered_df.index and max_decrease < 0 else '-'
        
        trend_stats = {
            'highest_consumption': {'value': max_month_row['Total_Consumption'], 'date': max_month_row['Date'].strftime('%b %Y')},
            'lowest_consumption': {'value': min_month_row['Total_Consumption'], 'date': min_month_row['Date'].strftime('%b %Y')},
            'largest_increase': {'value': max_increase, 'date': max_increase_month},
            'largest_decrease': {'value': max_decrease, 'date': max_decrease_month},
        }
    else:
        trend_stats = {
            'highest_consumption': {'value': 0, 'date': '-'},
            'lowest_consumption': {'value': 0, 'date': '-'},
            'largest_increase': {'value': 0, 'date': '-'},
            'largest_decrease': {'value': 0, 'date': '-'}
        }

    # --- Consumption Chart (only selected sectors) ---
    fig_consumption = go.Figure()
    sector_colors = {
        'commercial_industrial': '#00eaff',
        'domestic': '#ff4c6d'
    }
    if selected_sectors:
        for sector in ['commercial_industrial', 'domestic']:
            if sector in selected_sectors and sector in filtered_df.columns:
                fig_consumption.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df[sector],
                    mode='lines',
                    fill='tozeroy',
                    name='Commercial & Industrial' if sector == 'commercial_industrial' else 'Domestic & Public',
                    line=dict(width=0),
                    fillcolor=sector_colors.get(sector)
                ))
        fig_consumption.update_layout(
            xaxis_title="Date",
            yaxis_title="Consumption (Million kWh)",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font=dict(color="#fff", family=FONT_FAMILY),
            margin=dict(l=20, r=20, t=20, b=20)
        )
    else:
        fig_consumption.add_annotation(text="No sector selected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # --- Exports Chart ---
    fig_exports = go.Figure()
    if 'exports' in filtered_df.columns:
        fig_exports.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['exports'],
            mode='lines+markers',
            name='Exports',
            line=dict(color='#FFD369', width=3)
        ))
        fig_exports.update_layout(
            xaxis_title="Date",
            yaxis_title="Exports (Million kWh)",
            hovermode='x unified',
            template="plotly_dark",
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font=dict(color="#fff", family=FONT_FAMILY),
            margin=dict(l=20, r=20, t=20, b=20)
        )
    else:
        fig_exports.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # --- Losses Chart ---
    fig_losses = go.Figure()
    if 'losses' in filtered_df.columns:
        fig_losses.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['losses'],
            mode='lines+markers',
            name='Losses',
            line=dict(color='#4ecdc4', width=3)
        ))
        fig_losses.update_layout(
            xaxis_title="Date",
            yaxis_title="Losses (Million kWh)",
            hovermode='x unified',
            template="plotly_dark",
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font=dict(color="#fff", family=FONT_FAMILY),
            margin=dict(l=20, r=20, t=20, b=20)
        )
    else:
        fig_losses.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # --- Pie Chart ---
    pie_card = pie_chart_card(stats)
    # --- Year-over-Year Card ---
    yoy = yoy_card(filtered_df)
    # --- Summary stats ---
    summary = summary_cards_grid(stats, trend_stats)

    # --- Download logic ---
    download_data = None
    download_summary = None
    if triggered_id == 'download-btn':
        out_df = filtered_df[['Date', 'commercial_industrial', 'domestic', 'exports', 'losses']].copy()
        out_df['Date'] = out_df['Date'].dt.strftime('%Y-%m')
        csv_string = out_df.to_csv(index=False, encoding='utf-8')
        download_data = dict(content=csv_string, filename="filtered_electricity_data.csv")
    if triggered_id == 'download-summary-btn':
        # Prepare summary stats as CSV
        summary_df = pd.DataFrame({
            'Sector': ['Commercial & Industrial', 'Domestic & Public', 'Exports', 'Losses'],
            'Total (MkWh)': [stats[s]['total'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']],
            'Average (MkWh/mo)': [stats[s]['avg'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']],
            'Max (MkWh)': [stats[s]['max'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']],
            'Max Date': [stats[s]['max_date'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']],
            'Min (MkWh)': [stats[s]['min'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']],
            'Min Date': [stats[s]['min_date'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']],
            '% Change': [stats[s]['pct_change'] for s in ['commercial_industrial', 'domestic', 'exports', 'losses']]
        })
        summary_csv = summary_df.to_csv(index=False, encoding='utf-8')
        download_summary = dict(content=summary_csv, filename="summary_statistics.csv")

    return (
        chart_card("Consumption (Commercial & Domestic)", fig_consumption, "mdi:factory"),
        chart_card("Exports", fig_exports, "mdi:export", accent="#FFD369"),
        chart_card("Losses", fig_losses, "mdi:transmission-tower", accent="#4ecdc4"),
        summary,
        pie_card,
        yoy,
        download_data,
        download_summary
    )

# Error handling for running the app
if __name__ == '__main__':
    try:
        # Get port from environment variable (for Render/Heroku) or use default
        port = int(os.environ.get('PORT', 8050))
        # For local development and deployment
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Error running app: {e}")
        print("Make sure port 8050 is available or change the port number.")
        # Try alternative port if 8050 is busy
        try:
            print("Trying port 8080...")
            app.run(debug=False, host='0.0.0.0', port=8080)
        except Exception as e2:
            print(f"Error on port 8080: {e2}")
            print("Please check if any other application is using these ports.")