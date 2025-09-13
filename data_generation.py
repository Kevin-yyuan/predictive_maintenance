
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Number of records
NUM_TRUCKS = 50
NUM_MAINTENANCE_RECORDS = 500
NUM_SENSOR_READINGS = 20000
NUM_DRIVER_BEHAVIOR_RECORDS = 5000
NUM_PREDICT_RECORDS = 100

# --- 1. Vehicle Information ---
def generate_vehicle_info(num_trucks):
    trucks = []
    for i in range(num_trucks):
        trucks.append({
            'truck_id': f'TRUCK_{i:03}',
            'model': random.choice(['Model A', 'Model B', 'Model C']),
            'year': random.randint(2010, 2023),
            'total_mileage': random.randint(50000, 500000)
        })
    return pd.DataFrame(trucks)

# --- 2. Maintenance Logs ---
def generate_maintenance_logs(num_records, trucks_df):
    maintenance = []
    for _ in range(num_records):
        maintenance.append({
            'maintenance_id': fake.uuid4(),
            'truck_id': random.choice(trucks_df['truck_id']),
            'maintenance_date': fake.date_time_between(start_date='-5y', end_date='now'),
            'part_replaced': random.choice(['Engine', 'Tires', 'Brakes', 'Transmission', 'Battery']),
            'reason': random.choice(['Scheduled Maintenance', 'Failure', 'Wear and Tear'])
        })
    return pd.DataFrame(maintenance)

# --- 3. Sensor Data ---
def generate_sensor_data(num_readings, trucks_df):
    sensors = []
    for _ in range(num_readings):
        sensors.append({
            'reading_id': fake.uuid4(),
            'truck_id': random.choice(trucks_df['truck_id']),
            'timestamp': fake.date_time_between(start_date='-5y', end_date='now'),
            'engine_temp': random.uniform(85.0, 105.0),
            'oil_pressure': random.uniform(40.0, 80.0),
            'tire_pressure': random.uniform(90.0, 120.0),
            'vibration': random.uniform(0.1, 2.5)
        })
    return pd.DataFrame(sensors)

# --- 4. Driver Behavior Data ---
def generate_driver_behavior(num_records, trucks_df):
    behavior = []
    for _ in range(num_records):
        behavior.append({
            'behavior_id': fake.uuid4(),
            'truck_id': random.choice(trucks_df['truck_id']),
            'timestamp': fake.date_time_between(start_date='-5y', end_date='now'),
            'event': random.choice(['Harsh Braking', 'Harsh Acceleration', 'Speeding']),
            'severity': random.uniform(0.1, 1.0)
        })
    return pd.DataFrame(behavior)

# --- 5. Prediction Set ---
def generate_predict_set(num_records, trucks_df):
    predict_set = []
    for _ in range(num_records):
        predict_set.append({
            'truck_id': random.choice(trucks_df['truck_id']),
            'timestamp': fake.date_time_between(start_date='now', end_date='+30d'),
            'engine_temp': random.uniform(85.0, 105.0),
            'oil_pressure': random.uniform(40.0, 80.0),
            'tire_pressure': random.uniform(90.0, 120.0),
            'vibration': random.uniform(0.1, 2.5),
            'event': random.choice(['Harsh Braking', 'Harsh Acceleration', 'Speeding', None]),
            'severity': random.uniform(0.1, 1.0) if random.random() > 0.5 else 0.0
        })
    return pd.DataFrame(predict_set)

if __name__ == '__main__':
    # Generate data
    vehicle_info_df = generate_vehicle_info(NUM_TRUCKS)
    maintenance_logs_df = generate_maintenance_logs(NUM_MAINTENANCE_RECORDS, vehicle_info_df)
    sensor_data_df = generate_sensor_data(NUM_SENSOR_READINGS, vehicle_info_df)
    driver_behavior_df = generate_driver_behavior(NUM_DRIVER_BEHAVIOR_RECORDS, vehicle_info_df)

    # Save to CSV
    vehicle_info_df.to_csv('sample_data/vehicle_info.csv', index=False)
    maintenance_logs_df.to_csv('sample_data/maintenance_logs.csv', index=False)
    sensor_data_df.to_csv('sample_data/sensor_data.csv', index=False)
    driver_behavior_df.to_csv('sample_data/driver_behavior.csv', index=False)

    # Generate prediction set
    predict_set_df = generate_predict_set(NUM_PREDICT_RECORDS, vehicle_info_df)
    predict_set_df.to_csv('sample_data/predict_set.csv', index=False)

    print("Generated predictive maintenance datasets successfully!")
    print(f"  - vehicle_info.csv: {len(vehicle_info_df)} records")
    print(f"  - maintenance_logs.csv: {len(maintenance_logs_df)} records")
    print(f"  - sensor_data.csv: {len(sensor_data_df)} records")
    print(f"  - driver_behavior.csv: {len(driver_behavior_df)} records")
    print(f"  - predict_set.csv: {len(predict_set_df)} records")
