import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import json

class TrekkingRouteGenerator:
    def __init__(self):
        # Define Northeast India trekking routes with specific waypoints
        self.trekking_routes = {
            "Dzukou_Valley_Trek": {
                "base_camp": (25.5788, 93.8933),  # Near Kohima
                "authorized_route": [
                    (25.5788, 93.8933),  # Base camp
                    (25.5845, 93.9012),  # Checkpoint 1
                    (25.5901, 93.9089),  # Forest entry
                    (25.5967, 93.9156),  # Midway point
                    (25.6023, 93.9234),  # Valley approach
                    (25.6089, 93.9301),  # Dzukou Valley viewpoint
                ],
                "viewpoints": [
                    {"name": "Valley Viewpoint", "coords": (25.6089, 93.9301), "max_capacity": 50},
                    {"name": "Sunrise Point", "coords": (25.6045, 93.9267), "max_capacity": 30},
                    {"name": "Photo Point", "coords": (25.5967, 93.9156), "max_capacity": 25},
                ],
                "restricted_areas": [
                    {"name": "Wildlife Sanctuary Buffer", "center": (25.6150, 93.9400), "radius_km": 2},
                    {"name": "Landslide Prone Area", "center": (25.5900, 93.9400), "radius_km": 1.5},
                ],
                "difficulty": "moderate"
            },
            
            "Living_Root_Bridge_Trek": {
                "base_camp": (25.2993, 91.7362),  # Cherrapunji
                "authorized_route": [
                    (25.2993, 91.7362),  # Village start
                    (25.2956, 91.7398),  # Descent begin
                    (25.2923, 91.7445),  # Steep section
                    (25.2889, 91.7489),  # Bridge approach
                    (25.2867, 91.7523),  # Double decker bridge
                ],
                "viewpoints": [
                    {"name": "Double Decker Bridge", "coords": (25.2867, 91.7523), "max_capacity": 40},
                    {"name": "Waterfall View", "coords": (25.2889, 91.7489), "max_capacity": 20},
                    {"name": "Valley Overlook", "coords": (25.2923, 91.7445), "max_capacity": 15},
                ],
                "restricted_areas": [
                    {"name": "Unstable Cliff Area", "center": (25.2800, 91.7600), "radius_km": 1},
                    {"name": "Sacred Forest", "center": (25.2950, 91.7300), "radius_km": 0.8},
                ],
                "difficulty": "challenging"
            },
            
            "Mechuka_Valley_Trek": {
                "base_camp": (28.7833, 94.2167),  # Mechuka
                "authorized_route": [
                    (28.7833, 94.2167),  # Mechuka town
                    (28.7789, 94.2234),  # River crossing
                    (28.7756, 94.2301),  # Hill climb start
                    (28.7712, 94.2367),  # Ridge point
                    (28.7678, 94.2434),  # Peak approach
                    (28.7645, 94.2501),  # Summit
                ],
                "viewpoints": [
                    {"name": "Summit View", "coords": (28.7645, 94.2501), "max_capacity": 35},
                    {"name": "Valley Panorama", "coords": (28.7712, 94.2367), "max_capacity": 45},
                    {"name": "River Bend View", "coords": (28.7789, 94.2234), "max_capacity": 30},
                ],
                "restricted_areas": [
                    {"name": "Military Buffer Zone", "center": (28.7500, 94.2700), "radius_km": 3},
                    {"name": "Avalanche Risk Area", "center": (28.7600, 94.2600), "radius_km": 2},
                ],
                "difficulty": "hard"
            }
        }
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371  # Earth radius in km
    
    def is_on_authorized_route(self, lat, lng, route_name, tolerance_km=0.5):
        """Check if location is within authorized route tolerance"""
        route = self.trekking_routes[route_name]["authorized_route"]
        
        for route_lat, route_lng in route:
            distance = self.calculate_distance(lat, lng, route_lat, route_lng)
            if distance <= tolerance_km:
                return True
        return False
    
    def is_in_restricted_area(self, lat, lng, route_name):
        """Check if location is in restricted area"""
        restricted_areas = self.trekking_routes[route_name]["restricted_areas"]
        
        for area in restricted_areas:
            center_lat, center_lng = area["center"]
            distance = self.calculate_distance(lat, lng, center_lat, center_lng)
            if distance <= area["radius_km"]:
                return True, area["name"]
        return False, None
    
    def get_nearest_viewpoint(self, lat, lng, route_name):
        """Get nearest viewpoint and distance to it"""
        viewpoints = self.trekking_routes[route_name]["viewpoints"]
        min_distance = float('inf')
        nearest_viewpoint = None
        
        for viewpoint in viewpoints:
            vp_lat, vp_lng = viewpoint["coords"]
            distance = self.calculate_distance(lat, lng, vp_lat, vp_lng)
            if distance < min_distance:
                min_distance = distance
                nearest_viewpoint = viewpoint
        
        return nearest_viewpoint, min_distance
    
    def generate_trekker_data(self, num_trekkers=150, days=3):
        """Generate GPS data for trekkers on various routes"""
        print(f"Generating trekking data for {num_trekkers} trekkers over {days} days...")
        
        all_data = []
        trekker_id = 0
        
        for route_name in self.trekking_routes.keys():
            route_trekkers = num_trekkers // len(self.trekking_routes)
            
            for i in range(route_trekkers):
                trekker_data = self.generate_single_trekker_journey(
                    trekker_id, route_name, days
                )
                all_data.extend(trekker_data)
                trekker_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=[
            'trekker_id', 'route_name', 'timestamp', 'latitude', 'longitude',
            'is_on_route', 'nearest_viewpoint', 'distance_to_viewpoint',
            'crowd_at_viewpoint', 'is_anomaly', 'anomaly_type', 'severity'
        ])
        
        # Add crowd-based anomalies
        df = self.add_crowd_anomalies(df)
        
        # Add route deviation anomalies
        df = self.add_route_deviation_anomalies(df)
        
        return df
    
    def generate_single_trekker_journey(self, trekker_id, route_name, days):
        """Generate GPS track for single trekker"""
        route_info = self.trekking_routes[route_name]
        authorized_route = route_info["authorized_route"]
        
        journey_data = []
        start_time = datetime.now() - timedelta(days=days)
        
        # Trekker behavior patterns
        behavior_types = ['normal', 'explorer', 'photographer', 'speedy', 'cautious']
        behavior = random.choice(behavior_types)
        
        # Generate points along the route
        total_points = days * 24  # One point per hour over multiple days
        route_progress = 0
        
        for point_idx in range(total_points):
            timestamp = start_time + timedelta(hours=point_idx)
            
            # Determine position along route based on behavior and time
            if behavior == 'speedy':
                route_progress = min(len(authorized_route) - 1, point_idx * 0.8)
            elif behavior == 'cautious':
                route_progress = min(len(authorized_route) - 1, point_idx * 0.3)
            else:
                route_progress = min(len(authorized_route) - 1, point_idx * 0.5)
            
            route_idx = int(route_progress)
            route_fraction = route_progress - route_idx
            
            # Interpolate position along route
            if route_idx < len(authorized_route) - 1:
                current_point = authorized_route[route_idx]
                next_point = authorized_route[route_idx + 1]
                
                lat = current_point[0] + (next_point[0] - current_point[0]) * route_fraction
                lng = current_point[1] + (next_point[1] - current_point[1]) * route_fraction
            else:
                lat, lng = authorized_route[-1]  # At destination
            
            # Add some natural variation
            if behavior == 'explorer':
                lat += np.random.normal(0, 0.001)  # More wandering
                lng += np.random.normal(0, 0.001)
            elif behavior == 'photographer':
                # Photographers spend more time at viewpoints
                if random.random() < 0.3:  # 30% chance to stay at viewpoint longer
                    lat += np.random.normal(0, 0.0005)
                    lng += np.random.normal(0, 0.0005)
            else:
                lat += np.random.normal(0, 0.0003)  # Normal variation
                lng += np.random.normal(0, 0.0003)
            
            # Check if on authorized route
            is_on_route = self.is_on_authorized_route(lat, lng, route_name)
            
            # Find nearest viewpoint
            nearest_vp, distance_to_vp = self.get_nearest_viewpoint(lat, lng, route_name)
            
            # Simulate crowd at viewpoint
            crowd_count = 0
            if distance_to_vp < 0.1:  # Within 100m of viewpoint
                # Simulate realistic crowd patterns
                hour = timestamp.hour
                if 6 <= hour <= 18:  # Daytime
                    base_crowd = random.randint(5, nearest_vp["max_capacity"])
                    # Peak hours have more people
                    if 10 <= hour <= 16:
                        crowd_count = min(nearest_vp["max_capacity"], 
                                        int(base_crowd * random.uniform(0.8, 1.2)))
                    else:
                        crowd_count = int(base_crowd * 0.6)
                else:
                    crowd_count = random.randint(0, 5)  # Very few at night
            
            journey_data.append([
                trekker_id, route_name, timestamp, lat, lng,
                is_on_route, nearest_vp["name"] if nearest_vp else "None",
                distance_to_vp, crowd_count, False, "normal", "low"
            ])
        
        return journey_data
    
    def add_crowd_anomalies(self, df):
        """Add crowd-based anomalies"""
        print("Adding crowd-based anomalies...")
        
        for route_name in self.trekking_routes.keys():
            route_data = df[df['route_name'] == route_name].copy()
            viewpoints = self.trekking_routes[route_name]["viewpoints"]
            
            for viewpoint in viewpoints:
                vp_name = viewpoint["name"]
                max_capacity = viewpoint["max_capacity"]
                
                # Find points near this viewpoint
                vp_mask = (route_data['nearest_viewpoint'] == vp_name) & \
                         (route_data['distance_to_viewpoint'] < 0.1)
                
                # Group by hour to check crowd levels
                vp_data = route_data[vp_mask].copy()
                if len(vp_data) > 0:
                    vp_data['hour'] = vp_data['timestamp'].dt.hour
                    hourly_crowds = vp_data.groupby('hour')['crowd_at_viewpoint'].max()
                    
                    # Find overcrowded hours
                    overcrowded_hours = hourly_crowds[hourly_crowds > max_capacity * 0.8]
                    
                    for hour in overcrowded_hours.index:
                        hour_mask = (df['route_name'] == route_name) & \
                                   (df['nearest_viewpoint'] == vp_name) & \
                                   (df['timestamp'].dt.hour == hour) & \
                                   (df['distance_to_viewpoint'] < 0.1)
                        
                        df.loc[hour_mask, 'is_anomaly'] = True
                        df.loc[hour_mask, 'anomaly_type'] = 'overcrowding'
                        df.loc[hour_mask, 'severity'] = 'high' if hourly_crowds[hour] > max_capacity else 'medium'
        
        return df
    
    def add_route_deviation_anomalies(self, df):
        """Add route deviation and restricted area anomalies"""
        print("Adding route deviation anomalies...")
        
        # Add some trekkers who deviate from route
        deviation_trekkers = random.sample(list(df['trekker_id'].unique()), 
                                         k=min(20, len(df['trekker_id'].unique()) // 5))
        
        for trekker_id in deviation_trekkers:
            trekker_data = df[df['trekker_id'] == trekker_id].copy()
            route_name = trekker_data['route_name'].iloc[0]
            
            # Select random points to create deviations
            deviation_points = random.sample(list(trekker_data.index), 
                                           k=min(5, len(trekker_data) // 10))
            
            for point_idx in deviation_points:
                # Create route deviation
                original_lat = df.loc[point_idx, 'latitude']
                original_lng = df.loc[point_idx, 'longitude']
                
                # Move point away from authorized route
                deviation_lat = original_lat + random.uniform(-0.01, 0.01)
                deviation_lng = original_lng + random.uniform(-0.01, 0.01)
                
                df.loc[point_idx, 'latitude'] = deviation_lat
                df.loc[point_idx, 'longitude'] = deviation_lng
                df.loc[point_idx, 'is_on_route'] = False
                df.loc[point_idx, 'is_anomaly'] = True
                
                # Check if in restricted area
                is_restricted, area_name = self.is_in_restricted_area(
                    deviation_lat, deviation_lng, route_name
                )
                
                if is_restricted:
                    df.loc[point_idx, 'anomaly_type'] = 'restricted_area_entry'
                    df.loc[point_idx, 'severity'] = 'critical'
                else:
                    df.loc[point_idx, 'anomaly_type'] = 'route_deviation'
                    df.loc[point_idx, 'severity'] = 'medium'
        
        return df
    
    def visualize_routes_and_anomalies(self, df):
        """Create visualization of routes and detected anomalies"""
        print("Creating route visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        route_names = list(self.trekking_routes.keys())
        
        for idx, route_name in enumerate(route_names):
            if idx < 3:  # Only plot first 3 routes
                ax = axes[idx]
                route_data = df[df['route_name'] == route_name]
                route_info = self.trekking_routes[route_name]
                
                # Plot authorized route
                auth_route = route_info["authorized_route"]
                route_lats = [point[0] for point in auth_route]
                route_lngs = [point[1] for point in auth_route]
                ax.plot(route_lngs, route_lats, 'g-', linewidth=3, label='Authorized Route', alpha=0.7)
                
                # Plot viewpoints
                for vp in route_info["viewpoints"]:
                    vp_lat, vp_lng = vp["coords"]
                    ax.scatter(vp_lng, vp_lat, c='blue', s=200, marker='^', 
                             label='Viewpoint' if vp == route_info["viewpoints"][0] else "", 
                             alpha=0.8)
                    ax.annotate(vp["name"], (vp_lng, vp_lat), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
                
                # Plot restricted areas
                for area in route_info["restricted_areas"]:
                    center_lat, center_lng = area["center"]
                    circle = plt.Circle((center_lng, center_lat), area["radius_km"]/111, 
                                      color='red', fill=False, linestyle='--', linewidth=2,
                                      label='Restricted Area' if area == route_info["restricted_areas"][0] else "")
                    ax.add_patch(circle)
                
                # Plot normal trekker positions
                normal_data = route_data[route_data['is_anomaly'] == False]
                if len(normal_data) > 0:
                    ax.scatter(normal_data['longitude'], normal_data['latitude'], 
                             c='green', s=10, alpha=0.6, label='Normal Position')
                
                # Plot anomalies
                anomaly_data = route_data[route_data['is_anomaly'] == True]
                if len(anomaly_data) > 0:
                    colors = {'overcrowding': 'orange', 'route_deviation': 'red', 
                             'restricted_area_entry': 'darkred'}
                    
                    for anomaly_type in colors.keys():
                        type_data = anomaly_data[anomaly_data['anomaly_type'] == anomaly_type]
                        if len(type_data) > 0:
                            ax.scatter(type_data['longitude'], type_data['latitude'],
                                     c=colors[anomaly_type], s=50, marker='X',
                                     label=f'{anomaly_type.replace("_", " ").title()}')
                
                ax.set_title(f'{route_name.replace("_", " ")} Trek', fontsize=12, fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Summary plot
        ax = axes[3]
        anomaly_counts = df[df['is_anomaly'] == True]['anomaly_type'].value_counts()
        if len(anomaly_counts) > 0:
            colors = ['orange', 'red', 'darkred']
            ax.pie(anomaly_counts.values, labels=anomaly_counts.index, autopct='%1.1f%%',
                  colors=colors[:len(anomaly_counts)])
            ax.set_title('Anomaly Distribution')
        
        plt.tight_layout()
        plt.savefig('trekking_routes_anomalies.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'trekking_routes_anomalies.png'")
        plt.show()
    
    def save_data_with_metadata(self, df, filename='trekking_data.csv'):
        """Save data with route metadata"""
        df.to_csv(filename, index=False)
        
        # Save route metadata
        with open('route_metadata.json', 'w') as f:
            json.dump(self.trekking_routes, f, indent=2, default=str)
        
        print(f"Trekking data saved to {filename}")
        print(f"Route metadata saved to route_metadata.json")
        
        # Print statistics
        total_points = len(df)
        anomalies = len(df[df['is_anomaly'] == True])
        trekkers = df['trekker_id'].nunique()
        
        print(f"\nDataset Statistics:")
        print(f"Total GPS points: {total_points}")
        print(f"Number of trekkers: {trekkers}")
        print(f"Routes covered: {df['route_name'].nunique()}")
        print(f"Normal points: {total_points - anomalies}")
        print(f"Anomalous points: {anomalies} ({anomalies/total_points*100:.1f}%)")
        
        print(f"\nAnomaly Breakdown:")
        anomaly_counts = df[df['is_anomaly'] == True]['anomaly_type'].value_counts()
        for anomaly_type, count in anomaly_counts.items():
            print(f"  {anomaly_type.replace('_', ' ').title()}: {count}")
        
        print(f"\nRoute Distribution:")
        route_counts = df['route_name'].value_counts()
        for route, count in route_counts.items():
            print(f"  {route.replace('_', ' ')}: {count} points")

def main():
    """Main function to generate trekking data"""
    print("=== NORTHEAST INDIA TREKKING SAFETY DATA GENERATOR ===\n")
    
    # Initialize generator
    generator = TrekkingRouteGenerator()
    
    # Generate data
    df = generator.generate_trekker_data(num_trekkers=150, days=3)
    
    # Create visualization
    generator.visualize_routes_and_anomalies(df)
    
    # Save data
    generator.save_data_with_metadata(df, 'trekking_data.csv')
    
    print("\n=== DATA GENERATION COMPLETE ===")
    print("Next steps:")
    print("1. Run preprocess.py to extract ML features")
    print("2. Run anomaly_detection.py to train AI model")
    print("\nFiles created:")
    print("- trekking_data.csv (GPS data with anomalies)")
    print("- route_metadata.json (route information)")
    print("- trekking_routes_anomalies.png (visualization)")

if __name__ == "__main__":
    main()
