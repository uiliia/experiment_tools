import numpy as np
import scipy as sp
from scipy.stats import norm, beta, ttest_ind_from_stats, t
from math import ceil
import statsmodels.stats.api as sms
import statsmodels.stats.power as smp


def sample_size_calc_ttest(alpha, power, lift, ratio, alternative, mean, std):
    mean_control = mean
    mean_test = mean_control * lift

    effect_size = abs((mean_test - mean_control) / std)
    nobs1 = ceil(smp.tt_ind_solve_power(effect_size=effect_size, nobs1=None,
                                        alpha=alpha, power=power, alternative=alternative))
    nobs2 = ceil(nobs1 * ratio)
    size = nobs1 + nobs2

    res = {'test_size': nobs1,
           'control_size': nobs2,
           'total_size': size,
           'alpha': alpha,
           'power': power,
           'lift': lift,
           'ratio': ratio,
           'mean': mean
           }
    return res


def sample_size_calc_ztest(alpha, power, lift, ratio, alternative, mean):
    mean_control = mean
    mean_test = mean_control * lift

    effect_size = sms.proportion_effectsize(mean_test, mean_control)
    nobs1 = ceil(smp.zt_ind_solve_power(effect_size=effect_size, nobs1=None,
                                        alpha=alpha, power=power, alternative=alternative))
    nobs2 = ceil(nobs1 * ratio)
    size = nobs1 + nobs2

    res = {'test_size': nobs1,
           'control_size': nobs2,
           'total_size': size,
           'alpha': alpha,
           'power': power,
           'lift': lift,
           'ratio': ratio,
           'mean': mean
           }
    return res


def check_sample_ratio(control_fact, test_fact, ratio_plan):
    """
    Checks the statistical significance of the difference between the actual and planned ratio of sample sizes.

    :param control_fact: The size of the control group.
    :param test_fact: The size of the test group.
    :param ratio_plan: The planned ratio of the test group size to the total sample size.
    :return: p-value indicating the statistical significance of the difference between actual and planned ratios.
    """
    total_fact = control_fact + test_fact
    ratio_fact = test_fact / total_fact
    var_fact = ratio_fact * (1 - ratio_fact)

    total_plan = test_fact / ratio_plan
    var_plan = ratio_plan * (1 - ratio_plan)

    z = (ratio_plan - ratio_fact) / np.sqrt(var_fact / total_fact + var_plan / total_plan)
    return calculate_p_value_from_z_score(z)


def ttest(mean_control, mean_test, std_control, std_test, size_control, size_test, alpha, alternative):
    """
    Calculates the p-value, significance level, relative and absolute effects,
    and confidence intervals for the effects using a t-test.

    :param mean_control: Mean of the control group.
    :param mean_test: Mean of the test group.
    :param std_control: Standard deviation of the control group.
    :param std_test: Standard deviation of the test group.
    :param size_control: Sample size of the control group.
    :param size_test: Sample size of the test group.
    :param alpha: Significance level for the test.
    :param alternative: Specifies the alternative hypothesis. The options are 'two-sided', 'greater' or 'less'.
    :return: A dictionary containing the p-value, significance level,
    relative and absolute effects, and confidence intervals.
    """
    var_mean_control = std_control**2 / size_control
    var_mean_test = std_test**2 / size_test

    # difference_mean = mean_test - mean_control
    difference_mean_var = var_mean_control + var_mean_test
    # df = size_control + size_test - 2

    # t_stat = difference_mean / np.sqrt(difference_mean_var)
    # difference_distribution = t(df=df, loc=difference_mean, scale=np.sqrt(difference_mean_var))

    _, pvalue = ttest_ind_from_stats(mean1=mean_control, std1=std_control, nobs1=size_control,
                                     mean2=mean_test, std2=std_test, nobs2=size_test,
                                     equal_var=False, alternative=alternative)
    p_value = round(pvalue, 4)

    absolute_effect, relative_effect = calculate_effect_sizes(mean_control, mean_test,
                                                              np.sqrt(difference_mean_var), mean_control)
    margin_of_error = norm.ppf(1-alpha/2) * np.sqrt(difference_mean_var)

    ci_left_abs, ci_right_abs = calculate_confidence_interval(absolute_effect, margin_of_error)
    ci_left_rel, ci_right_rel = calculate_confidence_interval(relative_effect, margin_of_error / mean_control)

    return {
        'p_value': p_value,
        'significance_level': 1-alpha,
        'relative_effect': relative_effect,
        'ci_left_rel': ci_left_rel,
        'ci_right_rel': ci_right_rel,
        'absolute_effect': absolute_effect,
        'ci_left_abs': ci_left_abs,
        'ci_right_abs': ci_right_abs,
    }


def ztest(z_size_test, z_size_control, success_test, success_control, alpha, alternative):
    """
    Calculates the p-value, significance level, relative and absolute effects,
    and confidence intervals for the effects using a z-test.

    :param z_size_test: Sample size of the test group.
    :param z_size_control: Sample size of the control group.
    :param success_test: Number of successes in the test group.
    :param success_control: Number of successes in the control group.
    :param alpha: Significance level for the test.
    :param alternative: Specifies the alternative hypothesis. The options are 'two-sided', 'greater', or 'less'.
    :return: A dictionary containing the p-value, significance level,
    relative and absolute effects, and confidence intervals.
    """

    conv_test = success_test / z_size_test
    conv_control = success_control / z_size_control

    pooled_prob = (conv_test * z_size_test + conv_control * z_size_control) / (z_size_test + z_size_control)
    std_dev = np.sqrt(pooled_prob * (1 - pooled_prob) * (1 / z_size_test + 1 / z_size_control))

    z_score = (conv_test - conv_control) / std_dev
    p_value = calculate_p_value_from_z_score(z_score, alternative=alternative)

    absolute_effect, relative_effect = calculate_effect_sizes(conv_control, conv_test, std_dev, conv_control)
    margin_of_error = norm.ppf(1 - alpha / 2) * std_dev

    ci_left_abs, ci_right_abs = calculate_confidence_interval(absolute_effect, margin_of_error)
    ci_left_rel, ci_right_rel = calculate_confidence_interval(relative_effect, margin_of_error / conv_control)
    return {
        'p_value': p_value,
        'significance_level': 1-alpha,
        'relative_effect': relative_effect,
        'ci_left_rel': ci_left_rel,
        'ci_right_rel': ci_right_rel,
        'absolute_effect': absolute_effect,
        'ci_left_abs': ci_left_abs,
        'ci_right_abs': ci_right_abs,
    }


def bayes(trials_control, successes_control, trials_test, successes_test):
    alpha_prior = 1 + successes_control
    beta_prior = 1 + trials_control - successes_control
    alpha_posterior, beta_posterior = alpha_prior + successes_test, beta_prior + trials_test - successes_test
    # mode_posterior = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)

    n_simulations = 100000
    sim_before = beta.rvs(alpha_prior, beta_prior, size=n_simulations)
    sim_after = beta.rvs(alpha_posterior, beta_posterior, size=n_simulations)

    p_after_higher = (sim_after > sim_before).mean()
    return p_after_higher


def calculate_confidence_interval(value, margin_of_error):
    """
    Calculates the confidence interval.

    :param value: The central value for calculating the confidence interval.
    :param margin_of_error: The margin of error for calculating the confidence interval.
    :return: A tuple containing the lower and upper bounds of the confidence interval.
    """
    return round(value - margin_of_error, 4), round(value + margin_of_error, 4)


def calculate_effect_sizes(mean_control, mean_test, std_dev, conv_control):
    """
    Calculates the absolute and relative effect sizes.

    :param mean_control: The mean value in the control group.
    :param mean_test: The mean value in the test group.
    :param std_dev: The standard deviation used for calculating the effect size.
    :param conv_control: The conversion rate in the control group.
    :return: A tuple containing the absolute and relative effect sizes.
    """
    absolute_effect = round(mean_test - mean_control, 4)
    relative_effect = round(absolute_effect / conv_control, 4)
    return absolute_effect, relative_effect


def calculate_p_value_from_z_score(z_score, alternative='two-sided'):
    """
    Calculates the p-value based on the Z-score.

    :param z_score: The Z-score used for calculating the p-value.
    :param alternative: Type of the alternative hypothesis ('two-sided', 'smaller', 'greater'). Default is 'two-sided'.
    :return: The calculated p-value.
    """
    if alternative == 'two-sided':
        return 2 * (1 - norm.cdf(np.abs(z_score)))
    elif alternative == 'smaller':
        return norm.cdf(z_score)
    else:
        return 1 - norm.cdf(z_score)
