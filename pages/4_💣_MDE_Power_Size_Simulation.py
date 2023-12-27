import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from scipy.stats import norm, binom, mannwhitneyu
from joblib import Parallel, delayed
import time

st.markdown('## MDE-Power-Size Simulation')

np.random.seed(1)

def simulate(lift, n, data):
    results = []
    control = data[0:n]
    test = control * lift
    for _ in range(simulations):
        is_control = binom.rvs(1, 0.5, size=n)
        _, p = mannwhitneyu(control[is_control == True], test[is_control == False])
        results.append((lift, n, p))
    return results


def calculate_tpr(df):
    return pd.Series({"tpr": sum(df['pvalue'] < 0.05) / simulations})

# data inputs
col1, col2, col3 = st.columns(3)

with col1:
    mu = st.number_input(label='Mean', min_value=0.0001, value=None)
    lift_left_bound = st.number_input(label='Left bound of Lift', min_value=1.0001, value=None)
    size_left_bound = st.number_input(label='Left bound of Size', min_value=1, value=None)
with col2:
    sd = st.number_input(label='Standart deviation', min_value=0.0001, value=None)
    lift_right_bound = st.number_input(label='Right bound of Lift', min_value=1.0001, value=None)
    size_right_bound = st.number_input(label='Right bound of Size', min_value=1, value=None)
with col3:
    alpha = st.number_input(label='Alpha', min_value=0.0001, max_value=1.0, 
                                step= 0.01, value = 0.05,
                                placeholder= '0.05 is a standart value, but you can chise any')
    lift_step = st.number_input(label='Step of Lift', min_value=0.00001, value=None)
    size_step = st.number_input(label='Step of Size', min_value=1, value=None)



simulations = 1000

values = [mu, sd, alpha, lift_left_bound, lift_right_bound, lift_step, size_left_bound, size_right_bound, size_step]
st.write(values)
has_none = any(v is None for v in values)

if has_none:
    st.write('The simulation will start after you specify all parameters')
else:
    lifts = np.arange(lift_left_bound, lift_right_bound, lift_step)
    sizes = np.arange(size_left_bound, size_right_bound+1, size_step)

    if len(lifts) > 10 or len(sizes) > 10:
        st.write('The more values of lifts and sizes you want to simulate the longer it will take to process ☝️ ')
        st.write('But anyway you can proceed 😎')
    else:
        pass
        
    data = norm.rvs(loc=mu, scale=sd, size=10000)
    
    button_result = st.button("Go!", type="primary")
    if button_result:
    
        start_time = time.time()
        
        sim_results = Parallel(n_jobs=-1)(delayed(simulate)(lift, n, data) for lift, n in product(lifts, sizes))
        sim_res=pd.DataFrame([item for sublist in sim_results for item in sublist], columns=["lift", "n", "pvalue"])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        progress= f"Total runtime: {elapsed_time:.2f} seconds 🚀"
        st.write(progress)
        
        res = sim_res.groupby(["lift", "n"]).apply(calculate_tpr).reset_index()

        tab1, tab2 = st.tabs(["Chart", "Table"])

        with tab1:
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=res, x='n', y='tpr', hue='lift', marker='o', ax=ax)
            ax.set_xlabel('Size')
            ax.set_ylabel('Power')
            ax.legend(title='Lift')
            st.pyplot(fig)
        with tab2:
            
            st.table(data=res)
        

    else:
        pass
    
    
