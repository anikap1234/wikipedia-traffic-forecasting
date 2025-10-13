"""
Quick test to ensure everything is installed correctly
Run this BEFORE the full pipeline
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

print("✓ All imports successful!")
print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Pandas version: {pd.__version__}")

# Test data loading
try:
    data = pd.read_csv('data/train_2.csv')
    print(f"✓ Data loaded: {data.shape[0]} pages, {data.shape[1]-1} days")
except FileNotFoundError:
    print("✗ ERROR: train_2.csv not found! Download it first.")
    exit(1)

print("\n✓✓✓ All tests passed! Ready to run full pipeline.")