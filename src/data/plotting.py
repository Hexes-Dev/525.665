import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc
import numpy as np

from src.data.data_tools import ddmm_to_decimal
from src.ekf.ekf_utils import ekf_to_coor, quaternion_to_heading


def gps_scatter_map(map_fig, gps_readings):
    gps_lats, gps_lons, gps_times = zip(*[(ddmm_to_decimal(gps.latitude), ddmm_to_decimal(gps.longitude), (gps.timestamp - gps_readings[0].timestamp).total_seconds()) for gps in gps_readings])

    df = pd.DataFrame({
        'lat': gps_lats,
        'lon': gps_lons,
        'time': gps_times
    })

    map_fig.add_trace(go.Scattermap(
        lat=df['lat'], lon=df['lon'], mode='lines+markers', name='GPS',
        line=dict(color='green', width=2), marker=dict(size=6, color='green'),
        customdata=df[['time']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "Location: (%{lat}, %{lon})" +
            "<extra></extra>" # Removes the secondary box with trace name
        )
    ))

    # Add Start Point
    map_fig.add_trace(go.Scattermap(
        lat=[gps_lats[0]], lon=[gps_lons[0]], mode='markers', name='Start',
        marker=dict(size=24, color='green'),
    ))

def imu_scatter_map(map_fig, start_lat, start_lon, label, T, Q, V, P):

    start_lat = ddmm_to_decimal(start_lat)
    start_lon = ddmm_to_decimal(start_lon)

    lats, lons, alts = zip(*ekf_to_coor(start_lat, start_lon, P))

    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'time': T
    })

    # # add high fidelity trace line using every sample
    map_fig.add_trace(go.Scattermap(
        lat=df['lat'], lon=df['lon'], mode='lines', name=f'{label} Positions',
        line=dict(color='royalblue', width=2), marker=dict(size=6, color='royalblue'),customdata=df[['time']],
        hovertemplate=(
        "<b>%{customdata[0]}</b><br>" +
        "Location: (%{lat}, %{lon})" +
        "<extra></extra>" # Removes the secondary box with trace name
        )
    ))

    down_sample = 100
    pose_df = pd.DataFrame({
        'lat': lats[::down_sample],
        'lon': lons[::down_sample],
        'headings': [quaternion_to_heading(q) for q in Q[::down_sample]],
        'time': T[::down_sample]
    })

    map_fig.add_trace(go.Scattermap(
        lat=pose_df['lat'], lon=pose_df['lon'], mode='markers', name=f'{label} Poses',
        marker=dict(size=6, allowoverlap=True, color='royalblue', symbol='rocket', angle=pose_df['headings']),
        customdata=pose_df[['time', 'headings']],
        hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Location: (%{lat}, %{lon})</br>" +
                "Heading: %{customdata[1]}" +
                "<extra></extra>"  # Removes the secondary box with trace name
        )
    ))