import streamlit as st
from lib.validation import sample_size_calc_ttest, sample_size_calc_ztest


st.markdown('# 🦜 Design Module')

st.markdown('## Sample Size Calc')
criteria = st.selectbox('Select Criteria', ['t-test', 'z-test'])
result = None

col1, col2 = st.columns(2)

with col1:
    alpha = st.number_input(label='Alpha', min_value=0.0001, max_value=1.0,
                            step=0.01, value=0.05)
    alternative = st.selectbox('Alternative', ['two-sided', 'larger', 'smaller'])
with col2:
    power = st.number_input(label='Power', min_value=0.0001, max_value=1.0,
                            step=0.01, value=0.8)
    ratio = st.number_input('Ratio of control and test samples', value=1.0)

if criteria is not None:
    if criteria == 't-test':

        mean = st.number_input('Mean of control group', value=None)
        std = st.number_input('Standard Deviation of control group', min_value=0.01, value=None)
        lift = st.number_input('Lift', min_value=0.01, value=1.1)

        values = [alpha, power, std, mean, lift]
        has_none = any(v is None for v in values)

        if has_none:
            st.write('The result will appear here after you specify all parameters ✍️')
        else:
            result = sample_size_calc_ttest(alpha, power, lift, ratio, alternative, mean, std)

    elif criteria == 'z-test':

        mean = st.number_input('Conversion Rate of control group', value=None, help='as a decimal number from 0 to 1')
        lift = st.number_input('Lift', min_value=0.01, value=1.1, help='')

        values = [alpha, power, mean, lift]
        has_none = any(v is None for v in values)

        if has_none:
            st.write('The result will appear here after you specify all parameters ✍️')
        else:
            result = sample_size_calc_ztest(alpha, power, lift, ratio, alternative, mean)
    else:
        st.write('Error')
        
if result is not None:
    # st.write(result)
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Test Sample Size", value=result['test_size'])
    col2.metric(label="Control Sample Size", value=result['control_size'])
    col3.metric(label="Total", value=result['total_size'])

else:
    pass
