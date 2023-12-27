import streamlit as st
from lib.validation import bayes
import plotly.graph_objects as go
import numpy as np
from scipy.stats import beta


st.markdown('# 🤖 Bayesian Testing')

col1, col2 = st.columns(2)

with col1:
    trials_control = st.number_input(label='Trials in Control Group', min_value=1,
                            step=1, value=1)
    successes_control = st.number_input(label='Successes in Control Group', min_value=1,
                            step=1, value=None)
with col2:
    trials_test = st.number_input(label='Trials in Test Group', min_value=1,
                                     step=1, value=1)
    successes_test = st.number_input(label='Successes in Test Group', min_value=1,
                                        step=1, value=None)
st.divider()

values = [trials_control, successes_control, trials_test, successes_test]
has_none = any(v is None for v in values)

if has_none:
    st.write('The result will appear here after you specify all parameters ✍️')
else:
    result = bayes(trials_control, successes_control, trials_test, successes_test)
    st.write(f"Probability that success rate is higher after the test: {result:.2%}")

    # plotting
    x = np.linspace(0, 1, 1000)
    alpha_prior, beta_prior = 1 + successes_control, 1 + trials_control - successes_control
    alpha_posterior, beta_posterior = alpha_prior + successes_test, beta_prior + trials_test - successes_test
    mode_posterior = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, alpha_prior, beta_prior),
                             mode='lines', name='Prior Distribution'))

    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, alpha_posterior, beta_posterior),
                             mode='lines', name='Posterior Distribution'))

    fig.add_vline(x=mode_posterior, line_width=1, line_dash="dash", line_color="red")

    fig.add_annotation(x=mode_posterior, y=0,
                       text=f'Most Probable Success Rate: {mode_posterior:.2%}',
                       showarrow=False, yshift=10)

    fig.update_layout(
        # title='Bayesian Update of Success Probability',
        xaxis_title='Probability of Success',
        yaxis_title='Probability Density',
        # legend_title='Distributions',
        annotations=[
            dict(
                x=0.6,
                y=max(beta.pdf(x, alpha_posterior, beta_posterior)),
                xref="paper",
                yref="y",
                # text="Higher density indicates higher likelihood for the success probability",
                showarrow=False,
                align="center"
            )
        ]
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


