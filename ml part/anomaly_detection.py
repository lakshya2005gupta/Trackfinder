import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TrekkingSafetyAnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.route_metadata = {}
        
    def load_data(self, filename='trekking_ml_features.csv'):
        """Load processed ML features"""
        print(f"Loading ML features from {filename}...")
        df = pd.read_csv(filename)
        
        # Separate features and labels
        self.feature_columns = [col for col in df.columns if col != 'is_anomaly']
        X = df[self.feature_columns]
        y = df['is_anomaly']
        
        print(f"Loaded {len(X)} samples with {len(self.feature_columns)} features")
        print(f"Anomaly rate: {y.mean()*100:.1f}%")
        
        return X, y
    
    def load_route_metadata(self, filename='route_metadata.json'):
        """Load route metadata for contextual alerts"""
        try:
            with open(filename, 'r') as f:
                self.route_metadata = json.load(f)
            print(f"Loaded metadata for {len(self.route_metadata)} routes")
        except FileNotFoundError:
            print(f"Route metadata file {filename} not found")
    
    def train_model(self, X, y, contamination=0.1):
        """Train Isolation Forest model"""
        print(f"\nTraining Isolation Forest with contamination={contamination}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Fit on training data
        self.model.fit(X_train_scaled)
        self.is_trained = True
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
        train_pred_binary = (train_pred == -1).astype(int)
        test_pred_binary = (test_pred == -1).astype(int)
        
        print(f"\nModel Training Results:")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        print(f"\nTest Set Performance:")
        print(classification_report(y_test, test_pred_binary))
        
        return X_test, y_test, test_pred_binary
    
    def predict_anomaly(self, trekker_data):
        """Predict anomaly for single trekker data point"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # Prepare features
        features = pd.DataFrame([trekker_data])[self.feature_columns]
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.decision_function(features_scaled)[0]
        
        is_anomaly = prediction == -1
        confidence = abs(anomaly_score)
        
        result = {
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "anomaly_score": float(anomaly_score),
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate alert if anomaly detected
        if is_anomaly:
            result["alert"] = self.generate_trekking_alert(trekker_data, confidence)
        
        return result
    
    def generate_trekking_alert(self, trekker_data, confidence):
        """Generate contextual alert message for trekking scenarios"""
        alert_details = {
            "severity": "critical" if confidence > 0.4 else "high" if confidence > 0.2 else "medium",
            "location": {
                "lat": trekker_data.get('latitude', 0),
                "lng": trekker_data.get('longitude', 0)
            },
            "factors": [],
            "recommended_actions": [],
            "alert_type": "trekking_safety"
        }
        
        # Analyze factors contributing to anomaly
        
        # Route compliance issues
        if trekker_data.get('route_compliance_score', 1) == 0:
            alert_details["factors"].append("Trekker deviated from authorized route")
            alert_details["recommended_actions"].append("Guide trekker back to authorized path")
        
        # Overcrowding at viewpoints
        if trekker_data.get('overcrowding_risk', 0) == 1:
            crowd_ratio = trekker_data.get('crowd_density_ratio', 0)
            alert_details["factors"].append(f"Overcrowding detected - {crowd_ratio*100:.0f}% capacity reached")
            alert_details["recommended_actions"].append("Implement crowd control measures at viewpoint")
        
        # Speed anomalies
        speed = trekker_data.get('speed_kmh', 0)
        if speed > 15:  # Very high speed for trekking
            alert_details["factors"].append(f"Unusually high speed: {speed:.1f} km/h for trekking")
            alert_details["recommended_actions"].append("Check for emergency situation or vehicle use")
        elif speed < 0.5 and trekker_data.get('stationary_duration', 0) > 4:
            alert_details["factors"].append(f"Trekker stationary for extended period")
            alert_details["recommended_actions"].append("Verify trekker welfare - possible injury or fatigue")
        
        # Time-based risks
        if trekker_data.get('is_night', 0) == 1:
            hour = trekker_data.get('hour', 12)
            alert_details["factors"].append(f"Night-time trekking activity at {hour}:00")
            alert_details["recommended_actions"].append("Ensure proper lighting and guide assistance")
        
        # Restricted area visits
        restricted_visits = trekker_data.get('restricted_area_visits', 0)
        if restricted_visits > 0:
            alert_details["factors"].append(f"Trekker entered restricted area {restricted_visits} times")
            alert_details["recommended_actions"].append("Immediate intervention required - escort to safe zone")
            alert_details["severity"] = "critical"
        
        # Distance anomalies
        total_distance = trekker_data.get('total_distance_traveled', 0)
        if total_distance > 20:  # Very long distance for trekking
            alert_details["factors"].append(f"Excessive distance traveled: {total_distance:.1f} km")
            alert_details["recommended_actions"].append("Check if trekker is lost or took wrong route")
        
        # Direction change anomalies
        direction_change = trekker_data.get('direction_change_degrees', 0)
        if direction_change > 90:
            alert_details["factors"].append(f"Erratic movement pattern - {direction_change:.0f}° direction changes")
            alert_details["recommended_actions"].append("Monitor for disorientation or panic behavior")
        
        # Default factors if none detected
        if not alert_details["factors"]:
            alert_details["factors"].append("AI detected unusual trekking behavior pattern")
            alert_details["recommended_actions"].append("Verify trekker status and provide assistance if needed")
        
        return alert_details
    
    def save_model(self, filename='trekking_anomaly_model.pkl'):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'route_metadata': self.route_metadata
            }
            joblib.dump(model_data, filename)
            print(f"Model saved to {filename}")
    
    def load_model(self, filename='trekking_anomaly_model.pkl'):
        """Load pre-trained model"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.route_metadata = model_data.get('route_metadata', {})
            self.is_trained = True
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False

def simulate_real_time_trekking_detection(detector, num_simulations=20):
    """Simulate real-time GPS stream and anomaly detection for trekking"""
    print("\n=== REAL-TIME TREKKING ANOMALY DETECTION SIMULATION ===")
    print(f"Simulating {num_simulations} trekker GPS data points...")
    print("Press Ctrl+C to stop\n")
    
    # Load some test data for simulation
    try:
        df = pd.read_csv('processed_trekking_features.csv')
        test_samples = df.sample(n=num_simulations).to_dict('records')
    except:
        # Generate dummy trekking data if file not available
        test_samples = []
        routes = ['Dzukou_Valley_Trek', 'Living_Root_Bridge_Trek', 'Mechuka_Valley_Trek']
        
        for i in range(num_simulations):
            sample = {
                'latitude': 25.5788 + np.random.normal(0, 0.01),
                'longitude': 93.8933 + np.random.normal(0, 0.01),
                'speed_kmh': np.random.exponential(5),  # Lower speeds for trekking
                'hour': np.random.randint(6, 20),  # Daytime trekking
                'is_night': np.random.choice([0, 1], p=[0.9, 0.1]),
                'route_compliance_score': np.random.choice([0, 1], p=[0.1, 0.9]),
                'overcrowding_risk': np.random.choice([0, 1], p=[0.85, 0.15]),
                'distance_to_viewpoint': np.random.exponential(0.3),
                'crowd_density_ratio': np.random.beta(2, 5),  # Skewed towards lower ratios
                'stationary_duration': np.random.poisson(2),
                'restricted_area_visits': np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
            }
            
            # Add other required features with defaults
            required_features = detector.feature_columns
            for feature in required_features:
                if feature not in sample:
                    sample[feature] = 0
            test_samples.append(sample)
    
    alert_count = 0
    critical_alerts = 0
    
    try:
        for i, trekker_data in enumerate(test_samples):
            print(f"\n--- Trekker GPS Point {i+1}/{num_simulations} ---")
            print(f"Location: ({trekker_data['latitude']:.4f}, {trekker_data['longitude']:.4f})")
            print(f"Speed: {trekker_data['speed_kmh']:.1f} km/h | Time: {trekker_data['hour']}:00")
            print(f"Route Compliance: {'✅' if trekker_data['route_compliance_score'] == 1 else '❌'}")
            
            # Run anomaly detection
            result = detector.predict_anomaly(trekker_data)
            
            if result.get('is_anomaly', False):
                alert_count += 1
                print(f"🚨 TREKKING ANOMALY DETECTED! (Confidence: {result['confidence']:.3f})")
                
                alert = result.get('alert', {})
                severity = alert.get('severity', 'unknown').upper()
                print(f"   Severity: {severity}")
                
                if severity == 'CRITICAL':
                    critical_alerts += 1
                    print(f"   🔴 CRITICAL ALERT - IMMEDIATE ACTION REQUIRED")
                
                factors = alert.get('factors', [])
                actions = alert.get('recommended_actions', [])
                
                if factors:
                    print(f"   Factors: {'; '.join(factors[:2])}")  # Limit to first 2 factors
                if actions:
                    print(f"   Action: {actions[0]}")  # Show primary action
                    
            else:
                print(f"✅ Normal trekking behavior (Score: {result['anomaly_score']:.3f})")
            
            time.sleep(1.5)  # Simulate real-time delay
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    
    print(f"\n=== TREKKING SAFETY SIMULATION SUMMARY ===")
    print(f"Total trekker GPS points processed: {i+1}")
    print(f"Anomalies detected: {alert_count}")
    print(f"Critical alerts: {critical_alerts}")
    print(f"Alert rate: {alert_count/(i+1)*100:.1f}%")
    print(f"Critical alert rate: {critical_alerts/(i+1)*100:.1f}%")

def main():
    """Main function to train model and run trekking detection"""
    print("=== TREKKING SAFETY AI ANOMALY DETECTION SYSTEM ===\n")
    
    # Initialize detector
    detector = TrekkingSafetyAnomalyDetector()
    
    # Load route metadata
    detector.load_route_metadata('route_metadata.json')
    
    # Try to load existing model
    if not detector.load_model('trekking_anomaly_model.pkl'):
        print("No existing model found. Training new model...")
        
        # Load data and train
        try:
            X, y = detector.load_data()
            X_test, y_test, predictions = detector.train_model(X, y, contamination=0.1)
            detector.save_model('trekking_anomaly_model.pkl')
        except FileNotFoundError:
            print("Error: ML features file not found!")
            print("Please run preprocess.py first to generate trekking_ml_features.csv")
            return
    
    # Run real-time simulation
    print("\nTrekking safety AI model ready for anomaly detection!")
    
    choice = input("\nRun real-time trekking simulation? (y/n): ").lower().strip()
    if choice == 'y':
        simulate_real_time_trekking_detection(detector, num_simulations=15)
    
    print("\n=== TREKKING SAFETY AI SYSTEM READY ===")
    print("The model can now be integrated with:")
    print("- Mobile trekking apps")
    print("- Tourism police dashboards") 
    print("- Emergency response systems")
    print("- Blockchain tourist ID verification")

if __name__ == "__main__":
    main()
