import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

def load_trekking_data(filename='trekking_data.csv'):
    """Load the generated trekking data"""
    print(f"Loading trekking data from {filename}...")
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} GPS points for {df['trekker_id'].nunique()} trekkers")
    return df

def load_route_metadata(filename='route_metadata.json'):
    """Load route metadata"""
    try:
        with open(filename, 'r') as f:
            route_metadata = json.load(f)
        print(f"Loaded metadata for {len(route_metadata)} trekking routes")
        return route_metadata
    except FileNotFoundError:
        print(f"Route metadata file {filename} not found")
        return {}

def calculate_trekking_features(df):
    """Calculate trekking-specific features"""
    print("Calculating trekking movement features...")
    
    df = df.sort_values(['trekker_id', 'timestamp']).reset_index(drop=True)
    df['speed_kmh'] = 0.0
    df['distance_km'] = 0.0
    df['elevation_change'] = 0.0
    df['direction_change_degrees'] = 0.0
    
    for trekker_id in df['trekker_id'].unique():
        trekker_mask = df['trekker_id'] == trekker_id
        trekker_data = df[trekker_mask].copy()
        
        for i in range(1, len(trekker_data)):
            # Get consecutive points
            prev_point = trekker_data.iloc[i-1]
            curr_point = trekker_data.iloc[i]
            
            # Calculate distance using Haversine formula
            distance = haversine_distance(
                prev_point['latitude'], prev_point['longitude'],
                curr_point['latitude'], curr_point['longitude']
            )
            
            # Calculate time difference in hours
            time_diff = (curr_point['timestamp'] - prev_point['timestamp']).total_seconds() / 3600
            
            # Calculate speed (km/h)
            speed = distance / max(time_diff, 0.001)  # Avoid division by zero
            
            # Update dataframe
            df.loc[trekker_data.index[i], 'distance_km'] = distance
            df.loc[trekker_data.index[i], 'speed_kmh'] = speed
            
            # Calculate direction change (simplified)
            if i > 1:
                prev_prev_point = trekker_data.iloc[i-2]
                bearing1 = calculate_bearing(
                    prev_prev_point['latitude'], prev_prev_point['longitude'],
                    prev_point['latitude'], prev_point['longitude']
                )
                bearing2 = calculate_bearing(
                    prev_point['latitude'], prev_point['longitude'],
                    curr_point['latitude'], curr_point['longitude']
                )
                direction_change = abs(bearing2 - bearing1)
                direction_change = min(direction_change, 360 - direction_change)
                df.loc[trekker_data.index[i], 'direction_change_degrees'] = direction_change
    
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points using Haversine formula"""
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two GPS points"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360

def extract_time_features(df):
    """Extract time-based features from timestamp"""
    print("Extracting time features...")
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 5)).astype(int)
    df['is_peak_trekking_time'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
    df['is_dangerous_time'] = ((df['hour'] >= 18) | (df['hour'] <= 6)).astype(int)
    
    return df

def create_route_compliance_features(df, route_metadata):
    """Create features related to route compliance"""
    print("Creating route compliance features...")
    
    df['route_compliance_score'] = df['is_on_route'].astype(int)
    df['viewpoint_proximity'] = 0
    df['is_at_viewpoint'] = (df['distance_to_viewpoint'] < 0.1).astype(int)
    df['crowd_density_ratio'] = 0.0
    df['overcrowding_risk'] = 0
    
    for _, row in df.iterrows():
        route_name = row['route_name']
        if route_name in route_metadata:
            viewpoints = route_metadata[route_name]['viewpoints']
            
            # Find matching viewpoint
            for vp in viewpoints:
                if vp['name'] == row['nearest_viewpoint']:
                    max_capacity = vp['max_capacity']
                    crowd_ratio = row['crowd_at_viewpoint'] / max_capacity
                    df.loc[row.name, 'crowd_density_ratio'] = crowd_ratio
                    df.loc[row.name, 'overcrowding_risk'] = 1 if crowd_ratio > 0.8 else 0
                    break
        
        # Viewpoint proximity score
        if row['distance_to_viewpoint'] < 0.05:  # Very close
            df.loc[row.name, 'viewpoint_proximity'] = 3
        elif row['distance_to_viewpoint'] < 0.1:  # Close
            df.loc[row.name, 'viewpoint_proximity'] = 2
        elif row['distance_to_viewpoint'] < 0.5:  # Moderate distance
            df.loc[row.name, 'viewpoint_proximity'] = 1
    
    return df

def create_behavioral_features(df):
    """Create features based on trekker behavior patterns"""
    print("Creating behavioral features...")
    
    df = df.sort_values(['trekker_id', 'timestamp']).reset_index(drop=True)
    
    # Initialize behavioral features
    df['stationary_duration'] = 0
    df['avg_speed_last_3_hours'] = 0
    df['route_deviation_count'] = 0
    df['restricted_area_visits'] = 0
    df['total_distance_traveled'] = 0
    df['time_at_viewpoints'] = 0
    
    for trekker_id in df['trekker_id'].unique():
        trekker_mask = df['trekker_id'] == trekker_id
        trekker_data = df[trekker_mask].copy()
        
        cumulative_distance = 0
        restricted_visits = 0
        
        for i in range(len(trekker_data)):
            current_idx = trekker_data.index[i]
            
            # Calculate cumulative distance
            if i > 0:
                cumulative_distance += trekker_data.iloc[i]['distance_km']
            df.loc[current_idx, 'total_distance_traveled'] = cumulative_distance
            
            # Calculate stationary duration (consecutive low-speed points)
            stationary_count = 0
            for j in range(max(0, i-5), i+1):
                if j < len(trekker_data) and trekker_data.iloc[j]['speed_kmh'] < 1.0:
                    stationary_count += 1
            df.loc[current_idx, 'stationary_duration'] = stationary_count
            
            # Calculate average speed in last 3 hours (3 points = 3 hours)
            if i >= 3:
                last_3_speeds = trekker_data.iloc[i-2:i+1]['speed_kmh']
                df.loc[current_idx, 'avg_speed_last_3_hours'] = last_3_speeds.mean()
            
            # Count route deviations
            deviation_count = trekker_data.iloc[:i+1]['is_on_route'].apply(lambda x: 0 if x else 1).sum()
            df.loc[current_idx, 'route_deviation_count'] = deviation_count
            
            # Count restricted area entries (based on anomaly type)
            if 'restricted_area_entry' in trekker_data.iloc[:i+1]['anomaly_type'].values:
                restricted_visits += 1
            df.loc[current_idx, 'restricted_area_visits'] = restricted_visits
            
            # Time spent at viewpoints
            viewpoint_time = trekker_data.iloc[:i+1]['is_at_viewpoint'].sum()
            df.loc[current_idx, 'time_at_viewpoints'] = viewpoint_time
    
    return df

def select_features_for_ml(df):
    """Select and prepare final features for ML model"""
    print("Selecting features for ML model...")
    
    # Features for anomaly detection
    feature_columns = [
        'latitude', 'longitude', 'speed_kmh', 'distance_km', 'direction_change_degrees',
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_peak_trekking_time',
        'route_compliance_score', 'viewpoint_proximity', 'is_at_viewpoint',
        'crowd_density_ratio', 'overcrowding_risk', 'distance_to_viewpoint',
        'stationary_duration', 'avg_speed_last_3_hours', 'route_deviation_count',
        'restricted_area_visits', 'total_distance_traveled', 'time_at_viewpoints',
        'crowd_at_viewpoint'
    ]
    
    # Keep necessary columns
    ml_features = df[feature_columns + ['trekker_id', 'route_name', 'timestamp', 'is_anomaly', 'anomaly_type', 'severity']].copy()
    
    # Handle any remaining NaN values
    ml_features = ml_features.fillna(0)
    
    # Create feature matrix for ML
    X = ml_features[feature_columns]
    y = ml_features['is_anomaly']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features selected: {len(feature_columns)}")
    
    return ml_features, X, y

def save_processed_data(df, X, y, filename='processed_trekking_features.csv'):
    """Save processed features to CSV"""
    df.to_csv(filename, index=False)
    
    # Also save just the feature matrix
    feature_df = pd.concat([X, y], axis=1)
    feature_df.to_csv('trekking_ml_features.csv', index=False)
    
    print(f"Processed data saved to {filename}")
    print(f"ML features saved to trekking_ml_features.csv")
    
    # Print statistics
    print(f"\nProcessing Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Feature dimensions: {X.shape}")
    print(f"Anomaly rate: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    
    # Anomaly breakdown
    print(f"\nAnomaly Types:")
    anomaly_breakdown = df[df['is_anomaly'] == True]['anomaly_type'].value_counts()
    for anomaly_type, count in anomaly_breakdown.items():
        print(f"  {anomaly_type.replace('_', ' ').title()}: {count}")
    
    print(f"\nFeature Statistics:")
    print(X.describe())

def main():
    """Main preprocessing pipeline"""
    print("=== TREKKING SAFETY DATA PREPROCESSING ===\n")
    
    # Load data
    df = load_trekking_data('trekking_data.csv')
    route_metadata = load_route_metadata('route_metadata.json')
    
    # Feature engineering pipeline
    df = calculate_trekking_features(df)
    df = extract_time_features(df)
    df = create_route_compliance_features(df, route_metadata)
    df = create_behavioral_features(df)
    
    # Prepare for ML
    processed_df, X, y = select_features_for_ml(df)
    
    # Save processed data
    save_processed_data(processed_df, X, y)
    
    print("\n=== PREPROCESSING COMPLETE ===")
    print("Next step: Run anomaly_detection.py to train AI model")

if __name__ == "__main__":
    main()
