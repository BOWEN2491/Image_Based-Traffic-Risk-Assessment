from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

def generate_weak_label(df1: pd.DataFrame):
    df = df1.copy()
    #get rid of non_number features (mostly related to traffic light color)
    num_cols = ['n_person','n_vehicle','n_rider','n_ts',
            'near_person_count','near_vehicle_count',
            'max_box_area_person','max_box_area_vehicle',
            'sum_box_area_person','sum_box_area_vehicle',
            'bottom_half_person_ratio','center_region_vehicle_ratio',
            'object_density_per_mp','mean_overlap_iou']
    X = df[num_cols].copy()
    X = X.dropna()

    #standardlize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #apply KMeans
    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=50)
    labels = kmeans.fit_predict(X_scaled)

    #add the cluster results back to df
    df_clustered = df.loc[X.index].copy()
    df_clustered["cluster"] = labels
   
    #Pick features that could reflect the risk level according to clusters
    DANGER_COLS = [
        "n_person", "n_rider", "n_ts,max_box_area_person","sum_box_area_person", "sum_box_area_vehicle", "bottom_half_person_ratio", "object_density_per_mp", "mean_overlap_iou"
    ]

    #Gereate risk_weak label according to KMeans
    weights = {
        "n_person": 2.0,
        "n_rider": 1.5,
        "n_ts": 1.0,
        "max_box_area_person": 2.0,
        "sum_box_area_person": 2.0,
        "sum_box_area_vehicle": 1.5,
        "bottom_half_person_ratio": 2.0,
        "object_density_per_mp": 1.5,
        "mean_overlap_iou": 1.0,
    }
    
    use_danger = [c for c in DANGER_COLS if c in X.columns]
    W = np.array([weights[c] for c in use_danger])
    cluster_means = X.groupby(labels)[use_danger].mean()
    danger_score = (cluster_means * W).sum(axis=1)

    order = danger_score.sort_values().index.tolist()
    risk_map = {order[0]:0, order[1]:1, order[2]:2}
    risk_weak = pd.Series(labels, index=X.index).map(risk_map)

    df.loc[X.index, "risk_weak"] = risk_weak
    
    return df

