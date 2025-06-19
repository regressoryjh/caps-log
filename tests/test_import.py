import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_basic_imports():
    import streamlit
    import pandas
    import numpy
    import plotly
    import sklearn
    import catboost
    import app
