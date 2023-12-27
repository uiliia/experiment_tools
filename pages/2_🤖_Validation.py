import streamlit as st
import numpy as np
from scipy.stats import ttest_ind, norm, ttest_ind_from_stats
from statsmodels.stats.proportion import proportions_ztest
from collections import namedtuple
from math import sqrt, ceil
from lib.validation import ztest, ttest, check_sample_ratio
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt


st.markdown('# 🤖 Validation Module')

st.markdown('## Sample Ratio Check')

value_ = 1.0
ratio_plan = st.number_input('Planned Ratio of test group and total (test+control)', value = 0.5)
control_fact = st.number_input('Actual size of a control group', value = value_)
test_fact = st.number_input('Actual size of a test group', value = value_)
alpha = st.number_input('Significance level (alpha)', value = 0.05)

res = check_sample_ratio(control_fact, test_fact, ratio_plan)

if control_fact==value_ or test_fact == value_:
    st.write('The result will appear here after you specify the group size ✍️')

else:
    # res = check_sample_ratio(control_fact, test_fact, ratio_plan)
    if res > alpha:
        st.write('Everything is fine👏 ')
        st.write('The actual ratio is not significantly different from the planned one')
        st.write('p_value:', res)
    else:
        st.write('Something went wrong 😕')
        st.write('The actual ratio is significantly different from the planned one')
        st.write('p_value:', res)
    

st.markdown('## Test Validation')

criteria = st.selectbox('Select Criteria', ['t-test','z-test', 'mw', 'bayes'])


if criteria == 't-test':
    col1, col2= st.columns(2)

    with col1:
        alpha = st.number_input(label='Alpha', min_value=0.0001, max_value=1.0, 
                                step= 0.01, value = 0.05,
                                placeholder= '0.05 is a standart value, but you can chose any')
        significance = (1-alpha)*100
    with col2:
        alternative = st.selectbox('Alternative', ['two-sided', 'less', 'greater'])
   
    st.divider()

    col_control, col_test= st.columns(2)
    with col_control: 
        mean_control = st.number_input(label='Mean in a Control group', min_value=0.00001, value=None)
        var_control = st.number_input(label='Variance in a Control group', min_value=0.00001, value=None)
        size_control = st.number_input(label='Size of a Control group', min_value=1, value=None)
        
    with col_test:
        mean_test = st.number_input(label='Mean in a Test group', min_value=0.00001, value=None)
        var_test = st.number_input(label='Variance in a Test group', min_value=0.00001, value=None)
        size_test = st.number_input(label='Size of a Test group', min_value=1, value=None)
    
    values = [mean_control, var_control, size_control, mean_test, var_test, size_test]
    has_none = any(v is None for v in values)

    if has_none:
        st.write('The result will appear here after you specify all parameters ✍️')
    else:
        result = ttest(mean_control, mean_test, var_control, var_test, size_control, size_test, alpha, alternative)
        pvalue = result['p_value']
        significance = 100 * result['significance_level']
        rel_effect = result['relative_effect']
        ci_left_rel = result['ci_left_rel']
        ci_right_rel = result['ci_right_rel']
        abs_effect = result['absolute_effect']
        ci_left_abs = result['ci_left_abs']
        ci_right_abs = result['ci_right_abs']
        rel_ci = f"{100 * ci_left_rel}, {100 * ci_right_rel}"
        abs_ci = f"{ci_left_abs}, {ci_right_abs}"
        
        if pvalue < alpha:
            st.markdown('**The result is significant 👏 !**')
            col1, col2, col3 = st.columns(3)
            col1.metric(label="p-Value", value=pvalue)
            col1.metric(label="Significance level", value=f"{significance}%")
            col2.metric(label="Relative Effect", value=f"{100*rel_effect}%")
            col2.metric(label="CI of Relative Effect", value=f"[{rel_ci}]")
            col3.metric(label="Absolute Effect", value=f"{abs_effect}", help="empty")
            col3.metric(label="CI of Absolute Effect", value=f"[{abs_ci}]")
      
        else:
            st.markdown('The result is unsignificant 😢')
            col1, col2, col3 = st.columns(3)
            col1.metric(label="p-Value", value=pvalue)
            col1.metric(label="Significance level", value=f"{significance}%")
            col2.metric(label="Relative Effect", value=f"{100 * rel_effect}%")
            col2.metric(label="CI of Relative Effect", value=f"[{rel_ci}]")
            col3.metric(label="Absolute Effect", value=f"{abs_effect}", help="difference between means")
            col3.metric(label="CI of Absolute Effect", value=f"[{abs_ci}]")
            

    
elif criteria == 'z-test':
    col1, col2= st.columns(2)

    with col1:
        alpha = st.number_input(label='Alpha', min_value=0.0001, max_value=1.0, 
                                step= 0.01, value = 0.05,
                                placeholder= '0.05 is a standart value, but you can chose any')
        significance = (1-alpha)*100
    with col2:
        alternative = st.selectbox('Alternative', ['two-sided', 'less', 'greater'])
   
    st.divider()

    col_control, col_test= st.columns(2)
    with col_control: 
        success_control = st.number_input(label='Number of conversions in a Control group', min_value=0.00001, value=None)
        z_size_control = st.number_input(label='Size of a Control group', min_value=1, value=None)
        
    with col_test:
        success_test = st.number_input(label='Number of conversions in a Test group', min_value=0.00001, value=None)
        z_size_test = st.number_input(label='Size of a Test group', min_value=1, value=None)
    
    values = [success_control , z_size_control, success_test, z_size_test]
    has_none = any(v is None for v in values)

    if has_none:
        st.write('The result will appear here after you specify all parameters ✍️')
    else:
        result = ztest(z_size_test, z_size_control, success_test, success_control, alpha, alternative)
        pvalue = result['p_value']
        significance = 100*result['significance_level']
        rel_effect = result['relative_effect']
        ci_left_rel = result['ci_left_rel']
        ci_right_rel = result['ci_right_rel']
        abs_effect = result['absolute_effect']
        ci_left_abs = result['ci_left_abs']
        ci_right_abs = result['ci_right_abs']
        rel_ci = f"{100*ci_left_rel}, {100*ci_right_rel}"
        abs_ci = f"{ci_left_abs}, {ci_right_abs}"

        if pvalue < alpha:
            st.markdown('**The result is significant 👏 !**')
            col1, col2, col3 = st.columns(3)
            col1.metric(label="p-Value", value=pvalue)
            col1.metric(label="Significance level", value=f"{significance}%")
            col2.metric(label="Relative Effect", value=f"{100*rel_effect}%")
            col2.metric(label="CI of Relative Effect", value=f"[{rel_ci}]")
            col3.metric(label="Absolute Effect", value=f"{abs_effect}", help="difference between conversion rates")
            col3.metric(label="CI of Absolute Effect", value=f"[{abs_ci}]")

      
        else:
            st.write('The result is unsignificant 😢')
            col1, col2, col3 = st.columns(3)
            col1.metric(label="p-Value", value=pvalue)
            col1.metric(label="Significance level", value=f"{significance}%")
            col2.metric(label="Relative Effect", value=f"{100*rel_effect}%")
            col2.metric(label="CI of Relative Effect", value=f"[{rel_ci}]")
            col3.metric(label="Absolute Effect", value=f"{abs_effect}", help="difference between conversion rates")
            col3.metric(label="CI of Absolute Effect", value=f"[{abs_ci}]")
  

    
elif criteria == 'mw':
    st.write('Not released yet 😢')


    
elif criteria == 'bayes':
    st.write('Not released yet 😢')

