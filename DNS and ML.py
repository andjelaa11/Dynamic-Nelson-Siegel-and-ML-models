import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error

def factor_stats(beta, gamma):
    print("beta level", np.mean(beta[:, 0]), np.std(beta[:, 0]), np.min(beta[:, 0]), np.max(beta[:, 0]))
    print("beta slope", np.mean(beta[:, 1]), np.std(beta[:, 1]), np.min(beta[:, 1]), np.max(beta[:, 1]))
    print("beta curvature", np.mean(beta[:, 2]), np.std(beta[:, 2]), np.min(beta[:, 2]), np.max(beta[:, 2]))

    print("gama level", np.mean(gamma[:, 0]), np.std(gamma[:, 0]), np.min(gamma[:, 0]), np.max(gamma[:, 0]))
    print("gama slope", np.mean(gamma[:, 1]), np.std(gamma[:, 1]), np.min(gamma[:, 1]), np.max(gamma[:, 1]))
    print("gama curvature", np.mean(gamma[:, 2]), np.std(gamma[:, 2]), np.min(gamma[:, 2]), np.max(gamma[:, 2]))

    factors_df = pd.DataFrame({
        'beta1': beta[:, 0],
        'beta2': beta[:, 1],
        'beta3': beta[:, 2],
        'gama1': gamma[:, 0],
        'gama2': gamma[:, 1],
        'gama3': gamma[:, 2]
    })

    dns_corr = factors_df[['beta1', 'beta2', 'beta3']].corr()
    rdns_corr = factors_df[['gama1', 'gama2', 'gama3']].corr()

    print("dns corr", dns_corr.round(4))
    print("rdns corr", rdns_corr.round(4))


def data_loading_fp(filepath):
    data = pd.read_excel(filepath, usecols="A:Q")
    yield_data = data.iloc[:, 1:].values
    maturity = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 108, 120])
    # maturity = np.array([3, 6, 9, 12, 15, 18, 21, 24])
    CPI = pd.read_excel(filepath, usecols="R", header=0)
    IP = pd.read_excel(filepath, usecols="S", header=0)

    return data, yield_data, maturity, CPI, IP


def data_loading_lw_test(file_path):
    data = pd.read_excel(file_path, usecols="A:Q")
    yield_data = data.iloc[:, 1:].values
    maturity = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 108, 120])
    # maturity = np.array([3, 6, 9, 12, 15, 18, 21, 24])
    CPI = pd.read_excel(file_path, usecols="R", header=0)
    IP = pd.read_excel(file_path, usecols="S", header=0)

    return data, yield_data, maturity, CPI, IP


def RNS_factor_loading(la, m):
    A = np.array([[1, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    H = NS_factor_loading(la, m)
    G = np.dot(H, np.linalg.inv(A))

    return G, A


def RNS_factor_loading_tau_s(la, m, tau_s):
    exp_term = np.exp(-la * tau_s)
    A_tau_s = np.array([
        [1, (1 - exp_term) / (la * tau_s), ((1 - exp_term) / (la * tau_s)) - exp_term],
        [0, -((1 - exp_term) / (la * tau_s)), -(((1 - exp_term) / (la * tau_s)) - exp_term)],
        [0, 1 - ((1 - exp_term) / (la * tau_s)), 1 - (((1 - exp_term) / (la * tau_s)) - exp_term)]
    ])

    H = NS_factor_loading(la, m)
    G = np.dot(H, np.linalg.inv(A_tau_s))
    return G, A_tau_s


def plot_beta_factors_NS_RNS(beta, NS_rmse, gamma, RNS_rmse, niz_datuma):
    fig, axs = plt.subplots(1, 2, figsize=(17, 7))
    niz_datuma_str = [datum.strftime("%Y-%m-%d") for datum in niz_datuma]

    axs[0].plot(beta[:, 0], lw=2, color='blue')  # Level
    axs[0].plot(beta[:, 1], lw=2, color='green')  # Slope
    axs[0].plot(beta[:, 2], lw=2, color='red')  # Curvature

    xticks = range(0, len(niz_datuma), 50)  # plotting every 11th date on the x-axis
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels([niz_datuma_str[i] for i in xticks], rotation=45, ha='right')

    # axs[0].set_title(f"DNS model - β factor: rmse = {np.mean(NS_rmse):.4f}")
    axs[0].set_title(f"DNS model")
    axs[0].legend(["Level", "Slope", "Curvature"])

    axs[1].plot(gamma[:, 0], lw=2, color='blue')
    axs[1].plot(gamma[:, 1], lw=2, color='green')
    axs[1].plot(gamma[:, 2], lw=2, color='red')

    xticks = range(0, len(niz_datuma), 50)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels([niz_datuma_str[i] for i in xticks], rotation=45, ha='right')
    #
    # axs[1].set_title(f"RNS model - γ factor : rmse = {np.mean(RNS_rmse):.4f}")
    axs[1].set_title(f"RDNS model")
    axs[1].legend(["Short Rate", "Slope", "Curvature"])

    plt.show()


def plot_yield_curve_two_graphs(maturity, yield_data, y1, y2, string1, string2):
    t_idx = -1
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(maturity, yield_data[t_idx, :], 'o-', label='Actual Yield Curve')
    axs[0].plot(maturity, y1[t_idx, :], 'x--', label=string1)

    axs[0].set_title('Yield Curve using ' + string1)
    axs[0].set_xlabel('Maturity')
    axs[0].set_ylabel('Yield')
    axs[0].legend()

    axs[0].set_xticks(maturity)
    axs[0].set_xticklabels([f"{m}M" for m in maturity])

    axs[1].plot(maturity, yield_data[t_idx, :], 'o-', label='Actual Yield Curve')
    axs[1].plot(maturity, y2[t_idx, :], 'x--', label=string2)

    axs[1].set_title('Yield Curve using ' + string2)
    axs[1].set_xlabel('Maturity')
    axs[1].set_ylabel('Yield')
    axs[1].legend()

    axs[1].set_xticks(maturity)
    axs[1].set_xticklabels([f"{m}M" for m in maturity])

    plt.tight_layout()
    plt.show()


def plot_yield_curve_one_graph(maturity, yield_data, y1, y2, string1, string2, NS_rmse, RNS_rmse):
    t_idx = 185
    plt.figure(figsize=(10, 5))

    plt.plot(maturity, yield_data[t_idx, :], 'o-', label='Actual Yield Curve')
    plt.plot(maturity, y1[t_idx, :], 'x--', label=string1)
    plt.plot(maturity, y2[t_idx, :], 'm--', label=string2)
    plt.title('Yield Curve using ' + string1 + " & " + string2)
    plt.xlabel('Maturity')
    plt.ylabel('Yield')

    plt.xticks(maturity, [f"{m}M" for m in maturity])

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_beta_pred(beta, beta_pred):
    nobs, ncol = beta.shape

    components = ['level', 'Slope', 'Curvature']

    plt.figure(figsize=(14, 8))

    for i in range(ncol):
        plt.subplot(3, 1, i + 1)
        plt.plot(beta[:, i], label="beta")
        plt.plot(beta_pred[:, i], label="beta_pred", linestyle='--')
        plt.title('OLS esitmations vs ')
        plt.xlabel('Time')
        plt.ylabel(components[i])
        plt.legend()

    plt.tight_layout()
    plt.show()


def NS_factor_loading(la, m):
    H = np.column_stack([np.ones(len(m)),
                         (1 - np.exp(-la * m)) / (la * m),
                         ((1 - np.exp(-la * m)) / (la * m)) - np.exp(-la * m)])

    return H


def RNS_factor_loading(la, m):
    A = np.array([[1, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    H = NS_factor_loading(la, m)
    G = np.dot(H, np.linalg.inv(A))

    return G, A


"""
yield= H · βt

For each observation, we determine the β parameter using the OLS method
β= inv(HT· H)· (HT· Y) 
        """


def NS_ols(lambda_const, yield_data, maturity):
    H = NS_factor_loading(lambda_const, maturity)

    nobs, ncol = yield_data.shape
    rmse = np.zeros(nobs)
    beta = np.zeros((nobs, 3))
    yield_ns = np.zeros_like(yield_data)

    for t in range(nobs):
        beta_t = np.linalg.solve(np.dot(H.T, H), np.dot(H.T, yield_data[t]))
        yield_ns_t = np.dot(H, beta_t)

        beta[t, :] = beta_t
        rmse[t] = np.sqrt(np.mean((yield_data[t] - yield_ns_t) ** 2))
        yield_ns[t, :] = yield_ns_t

    return {'beta': beta, 'rmse': rmse, 'yield_ns': yield_ns}


def RNS_ols(yield_data, G):
    nobs, ncol = yield_data.shape
    rmse = np.zeros(nobs)
    gamma = np.zeros((nobs, 3))
    yield_rns = np.zeros_like(yield_data)

    for t in range(nobs):
        gamma_t = np.linalg.solve(np.dot(G.T, G), np.dot(G.T, yield_data[t]))
        yield_rns_t = np.dot(G, gamma_t)

        gamma[t, :] = gamma_t
        rmse[t] = np.sqrt(np.mean((yield_data[t] - yield_rns_t) ** 2))
        yield_rns[t, :] = yield_rns_t

    return {'gamma': gamma, 'rmse': rmse, 'yield_rns': yield_rns}


"""
Using the AR(1) model to determine the parameters µ, F, vt
βt = µ + F · βt−1 + vt

For each of the three beta factors(level, slope, curvature), we separately calculate the parameters µ, F, vt
        """


def OLS_beta_pred(beta):
    X = beta[:-1]
    y = beta[1:]

    X_with_intercept = sm.add_constant(X)

    model = sm.OLS(y, X_with_intercept)
    results = model.fit()

    params = results.params
    mu = params[0]
    F = params[1:].T
    v_t = results.resid

    return mu, F, v_t


def calculate_beta_pred_ols(mi, F, beta_first, v_t, nobs):
    beta_pred = np.zeros((nobs, 3))
    beta_pred[0] = beta_first

    for t in range(1, nobs):
        beta_pred[t] = mi + np.dot(F, beta_pred[t - 1]) + v_t[t - 1]

    return beta_pred


def NS_pred(num_pred, H, beta_last_date, mi, F, v_t):
    beta_pred_rns = np.zeros((num_pred, 3))
    beta_pred_rns[0] = beta_last_date

    for t in range(1, num_pred):
        beta_pred_rns[t] = mi + np.dot(F, beta_pred_rns[t - 1]) + v_t[:, t - 1]

    yield_rns_pred = np.zeros((num_pred, 16))
    for t in range(num_pred):
        yield_rns_pred[t] = np.dot(H, beta_pred_rns[t])
    return yield_rns_pred


def predict6m_more_rotatedARIMAS(maturity, yield_data_test, nobs, ncol, gamma, M, N, beta, mi_ns, F_ns, v_t_ns,
                                 lambda_const):

    H = NS_factor_loading(lambda_const, maturity)
    G, A = RNS_factor_loading_tau_s(lambda_const, maturity, (6 / 12))

    beta_pred_6_111 = np.zeros((6, ncol))
    beta_pred_6_012 = np.zeros((6, ncol))
    beta_pred_6_212 = np.zeros((6, ncol))

    for j in range(3):
        series = beta[:, j]

        model = SARIMAX(series, order=(1, 1, 1))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=6)
        beta_pred_6_111[:, j] = forecast.predicted_mean

        model = SARIMAX(series, order=(0, 1, 2))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=6)
        beta_pred_6_012[:, j] = forecast.predicted_mean

        model = SARIMAX(series, order=(2, 1, 2))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=6)
        beta_pred_6_212[:, j] = forecast.predicted_mean

    yield_pred_NSarima111 = np.zeros((6, 16))
    yield_pred_NSarima012 = np.zeros((6, 16))
    yield_pred_NSarima212 = np.zeros((6, 16))

    for i in range(6):
        yield_pred_NSarima111[i] = np.dot(H, beta_pred_6_111[i])
        yield_pred_NSarima012[i] = np.dot(H, beta_pred_6_012[i])
        yield_pred_NSarima212[i] = np.dot(H, beta_pred_6_212[i])

    gamma_pred_macro_111 = np.zeros((6, ncol))
    gamma_pred_111 = np.zeros((6, ncol))

    gamma_pred_macro_012 = np.zeros((6, ncol))
    gamma_pred_012 = np.zeros((6, ncol))

    gamma_pred_macro_212 = np.zeros((6, ncol))
    gamma_pred_212 = np.zeros((6, ncol))

    # --- Forecast za M (CPI) i N (IP) ---
    model_M = SARIMAX(M, order=(1, 1, 1))
    result_M = model_M.fit(disp=False)
    M_future = result_M.get_forecast(steps=6).predicted_mean

    model_N = SARIMAX(N, order=(1, 1, 1))
    result_N = model_N.fit(disp=False)
    N_future = result_N.get_forecast(steps=6).predicted_mean

    exog_future = np.column_stack([M_future, N_future])

    #######################################

    gamma_df = pd.DataFrame(gamma, columns=['g1', 'g2', 'g3'])
    exog_df = pd.DataFrame({'M': M, 'N': N})

    # VARIMA(1,1)
    model_varima = VARMAX(gamma_df, exog=exog_df, order=(1, 1))
    result = model_varima.fit(disp=False)
    print(result.summary())
    gamma_pred_6_varima_11 = result.forecast(steps=6, exog=exog_future).values

    # VARIMA(0,2)
    model_varima = VARMAX(gamma_df, exog=exog_df, order=(0, 2))
    result2 = model_varima.fit(disp=False)
    print(result2.summary())
    gamma_pred_6_varima_02 = result2.forecast(steps=6, exog=exog_future).values

    # VARIMA(2,2)
    model_varima = VARMAX(gamma_df, exog=exog_df, order=(2, 2))
    result3 = model_varima.fit(disp=False)
    print(result3.summary())
    gamma_pred_6_varima_22 = result3.forecast(steps=6, exog=exog_future).values

    # VARIMA(1,1) bez exog
    model_varima_no = VARMAX(gamma_df,  order=(1, 1))
    result_no4 = model_varima_no.fit(disp=False)
    print(result_no4.summary())
    gamma_pred_6_varima_11_no = result_no4.forecast(steps=6).values

    # VARIMA(0,2) bez exog
    model_varima_no = VARMAX(gamma_df,  order=(0, 2))
    result_no5 = model_varima_no.fit(disp=False)
    print(result_no5.summary())
    gamma_pred_6_varima_02_no = result_no5.forecast(steps=6).values

    # VARIMA(2,2) bez exog
    model_varima_no = VARMAX(gamma_df,  order=(2, 2))
    result_no = model_varima_no.fit(disp=False)
    print(result_no.summary())
    gamma_pred_6_varima_22_no = result_no.forecast(steps=6).values

    yield_macro_pred_varima_11 = np.zeros((6, len(maturity)))
    yield_macro_pred_varima_02 = np.zeros((6, len(maturity)))
    yield_macro_pred_varima_22 = np.zeros((6, len(maturity)))

    yield_macro_pred_varima_11_no = np.zeros((6, len(maturity)))
    yield_macro_pred_varima_02_no= np.zeros((6, len(maturity)))
    yield_macro_pred_varima_22_no = np.zeros((6, len(maturity)))

    for i in range(6):
        yield_macro_pred_varima_11[i] = np.dot(G, gamma_pred_6_varima_11[i])
        yield_macro_pred_varima_02[i] = np.dot(G, gamma_pred_6_varima_02[i])
        yield_macro_pred_varima_22[i] = np.dot(G, gamma_pred_6_varima_22[i])

        yield_macro_pred_varima_11_no[i] = np.dot(G, gamma_pred_6_varima_11_no[i])
        yield_macro_pred_varima_02[i] = np.dot(G, gamma_pred_6_varima_02_no[i])
        yield_macro_pred_varima_22_no[i] = np.dot(G, gamma_pred_6_varima_22_no[i])

    #######################################

    for j in range(3):
        series = gamma[:, j]
        exog = np.column_stack([M, N])

        # --- ARIMA(1,1,1) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(1, 1, 1))
        result_macro = model_macro.fit(disp=False)
        print(f"ARIMA(1,1,1) + macro fit za gamma_{j}: AIC={result_macro.aic:.2f}")

        forecast_macro = result_macro.get_forecast(steps=6, exog=exog_future)
        gamma_pred_macro_111[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(1,1,1) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(1, 1, 1))
        result_plain = model_plain.fit(disp=False)
        print(f"ARIMA(1,1,1) fit za gamma_{j}: AIC={result_plain.aic:.2f}")

        forecast_plain = result_plain.get_forecast(steps=6)
        gamma_pred_111[:, j] = forecast_plain.predicted_mean

        # --- ARIMA(0,1,2) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(0, 1, 2))
        result_macro = model_macro.fit(disp=False)
        print(f"ARIMA(0,1,2) + macro fit za gamma_{j}: AIC={result_macro.aic:.2f}")

        forecast_macro = result_macro.get_forecast(steps=6, exog=exog_future)
        gamma_pred_macro_012[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(0,1,2) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(0, 1, 2))
        result_plain = model_plain.fit(disp=False)
        print(f"ARIMA(0,1,2) fit za gamma_{j}: AIC={result_plain.aic:.2f}")

        forecast_plain = result_plain.get_forecast(steps=6)
        gamma_pred_012[:, j] = forecast_plain.predicted_mean

        # --- ARIMA(2,1,2) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(2, 1, 2))
        result_macro = model_macro.fit(disp=False)
        print(f"ARIMA(2,1,2) + macro fit za gamma_{j}: AIC={result_macro.aic:.2f}")

        forecast_macro = result_macro.get_forecast(steps=6, exog=exog_future)
        gamma_pred_macro_212[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(2,1,2) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(2, 1, 2))
        result_plain = model_plain.fit(disp=False)
        print(f"ARIMA(2,1,2) fit za gamma_{j}: AIC={result_plain.aic:.2f}")

        forecast_plain = result_plain.get_forecast(steps=6)
        gamma_pred_212[:, j] = forecast_plain.predicted_mean


    yield_macro_pred_111 = np.zeros((6, len(maturity)))
    yield_pred_111 = np.zeros((6, len(maturity)))

    yield_macro_pred_012 = np.zeros((6, len(maturity)))
    yield_pred_012 = np.zeros((6, len(maturity)))

    yield_macro_pred_212 = np.zeros((6, len(maturity)))
    yield_pred_212 = np.zeros((6, len(maturity)))

    for i in range(6):
        yield_macro_pred_111[i] = np.dot(G, gamma_pred_macro_111[i])
        yield_pred_111[i] = np.dot(G, gamma_pred_111[i])

        yield_macro_pred_012[i] = np.dot(G, gamma_pred_macro_012[i])
        yield_pred_012[i] = np.dot(G, gamma_pred_012[i])

        yield_macro_pred_212[i] = np.dot(G, gamma_pred_macro_212[i])
        yield_pred_212[i] = np.dot(G, gamma_pred_212[i])


    actual = yield_data_test[5, :]
    # rf= [ 4.56673538,  4.69549735,  4.74484817,  4.73879337,  4.68004115,
    #     4.58312968,  4.46587617,  4.35309596,  4.17497218,  4.05860113,
    #     3.92227865,  3.83101259,  3.7983235 ,  3.72411055,  3.65100613,
    #     3.65330342 ]
    #
    # rf = [
    #     4.779540873,
    #     4.859412993,
    #     4.882211756,
    #     4.856096166,
    #     4.780044571,
    #     4.671339161,
    #     4.549487114,
    #     4.434113769,
    #     4.252857067,
    #     4.137806515,
    #     4.00662223,
    #     3.911488378,
    #     3.86876606,
    #     3.799136218,
    #     3.734770862,
    #     3.732675581
    # ]
    # rf = [4.99561507, 5.06495963, 5.0707968, 5.03038839, 4.93880381, 4.81235231, 4.6731227, 4.5412988 , 4.3303954 , 4.1934665,  4.0404181,  3.93086468, 3.87155868, 3.79475015, 3.72454063 ,3.7134532]

    # sa grid search

    rf = [
        4.779540873,
        4.859412993,
        4.882211756,
        4.856096166,
        4.780044571,
        4.671339161,
        4.549487114,
        4.434113769,
        4.252857067,
        4.137806515,
        4.00662223,
        3.911488378,
        3.86876606,
        3.799136218,
        3.734770862,
        3.732675581
    ]

    # xgb = [
    #                               4.868977,
    #                               4.979370,
    #                               5.011387,
    #                              4.994840,
    #                              4.933556,
    #                              4.842418,
    #                              4.738781,
    #                             4.640507,
    #                              4.485223,
    #                              4.368695,
    #                             4.214110,
    #                              4.082749,
    #                              3.997613,
    #                              3.900828,
    #                             3.773554,
    #                             3.735138
    # ]


    xgb = [
                                4.889503,
                                  5.002182,
                                  5.034397,
                                 5.017213,
                                 4.952790,
                                 4.856702,
                                 4.747952,
                                4.644750,
                                 4.481904,
                                 4.362081,
                                 4.205288,
                                4.073077,
                                 3.988425,
                                 3.887949,
                                3.755211,
                                3.715348

    ]

    rmse_rdns_macro_012 = np.sqrt(mean_squared_error(actual, yield_macro_pred_012[5, :]))
    rmse_rdns_012 = np.sqrt(mean_squared_error(actual, yield_pred_012[5, :]))
    rmse_dns_012 = np.sqrt(mean_squared_error(actual, yield_pred_NSarima012[5, :]))
    rmse_varima_012 = np.sqrt(mean_squared_error(actual, yield_macro_pred_varima_02[5, :]))
    rmse_varima_012_no = np.sqrt(mean_squared_error(actual, yield_macro_pred_varima_02_no[5, :]))
    rmse_yield_macro_pred_varima_11_no = np.sqrt(mean_squared_error(actual, yield_macro_pred_varima_11_no[5,:]))
    rmse_xgb  =  np.sqrt(mean_squared_error(actual, xgb))
    rmse_rf = np.sqrt(mean_squared_error(actual, rf))
    print("=== RMSE - 6-Month Ahead Forecast ===")
    print(f"RDNS (with macro, ARIMA(0,1,2)): {rmse_rdns_macro_012:.6f}")
    print(f"RDNS (ARIMA(0,1,2)):             {rmse_rdns_012:.6f}")
    print(f"DNS  (ARIMA(0,1,2)):             {rmse_dns_012:.6f}")
    print(f"DNS  (VARIMA + macro(0,1,2)):             {rmse_varima_012:.6f}")
    print(f"DNS  (VARIMA :             {rmse_varima_012_no:.6f}")
    print(f"RF 6 meseci :             {rmse_rf:.6f}")
    print("XGB:", rmse_xgb)
    print(f"VARIMA 11: RDNS: ",rmse_yield_macro_pred_varima_11_no)

    plt.figure(figsize=(10, 5))

    plt.plot(maturity, yield_data_test[5, :],color='black', linewidth=1.5, label="Actual")

    # plt.plot(maturity, yield_macro_pred_012[5, :], 'x--', color='blue', label="RDNS (with macro, ARIMA(0,1,2))")
    plt.plot(maturity, yield_pred_012[5, :], 'x--',color='red', label="RDNS (ARIMA(0,1,2))")
    # plt.plot(maturity, yield_pred_NSarima012[5, :], 'x--',color='green', label="DNS (ARIMA(0,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_02[5, :], 'x--', label="RDNS (VARIMA(0,1,2))")


    # plt.plot(maturity, yield_macro_pred_111[5, :], 'x--',color='blue', label="RDNS (with macro, ARIMA(1,1,1))")
    # plt.plot(maturity, yield_pred_111[5, :], 'x--', color='cornflowerblue',label="RDNS (ARIMA(1,1,1))")
    # plt.plot(maturity, yield_pred_NSarima111[5, :], 'x--', color='green',label="DNS (ARIMA(1,1,1))")
    # plt.plot(maturity, yield_macro_pred_varima_11[5, :], 'x--',color='red', label="RDNS (with macro, VARIMA(1,1,1))")
    plt.plot(maturity, yield_macro_pred_varima_11_no[5, :], 'x--', color='orange', label="RDNS (VARIMA(1,1,1))")


    # plt.plot(maturity, yield_macro_pred_212[5, :], 'x--', color='blue', label="RDNS (with macro, ARIMA(2,1,2))")
    # plt.plot(maturity, yield_pred_212[5, :], 'x--', color='cornflowerblue',label="RDNS (ARIMA(2,1,2))")
    plt.plot(maturity, yield_pred_NSarima212[5, :], 'x--', color='green',label="DNS (ARIMA(2,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_22[5, :], 'x--', color='red', label="RDNS (with macro, VARIMA(2,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_22_no[5, :], 'x--', color='orange', label="RDNS (VARIMA(2,1,2))")


    plt.plot(maturity, rf, 'x--', color='cornflowerblue',label="Random Forest")
    plt.plot(maturity, xgb, 'x--', color='purple',label="Gradient Boosting")



    # plt.title("6-Month Ahead Yield Curve Forecast: RDNS, DNS and Random Forest vs Actual (December 2023.)")
    plt.title("6-Month Ahead Yield Curve Forecast: RDNS and DNS models vs Actual yield curve (December 2023.)")
    plt.xlabel('Maturity (Months)')
    plt.ylabel('Yield (%)')
    plt.xticks(maturity, [f"{m}M" for m in maturity])
    plt.legend()
    plt.tight_layout()
    plt.show()




def predict1m_more_rotatedARIMAS(maturity, yield_data_test, nobs, ncol, gamma, M, N, beta, mi_ns, F_ns, v_t_ns,
                                 lambda_const):

    H = NS_factor_loading(lambda_const, maturity)
    G, A = RNS_factor_loading_tau_s(lambda_const, maturity, (1 / 12))

    beta_pred_1_111 = np.zeros((1, ncol))
    beta_pred_1_012 = np.zeros((1, ncol))
    beta_pred_1_212 = np.zeros((1, ncol))

    for j in range(3):
        series = beta[:, j]

        model = SARIMAX(series, order=(1, 1, 1))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=1)
        beta_pred_1_111[:, j] = forecast.predicted_mean

        model = SARIMAX(series, order=(0, 1, 2))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=1)
        beta_pred_1_012[:, j] = forecast.predicted_mean

        model = SARIMAX(series, order=(2, 1, 2))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=1)
        beta_pred_1_212[:, j] = forecast.predicted_mean

    yield_pred_NSarima111 = np.zeros((1, 16))
    yield_pred_NSarima012 = np.zeros((1, 16))
    yield_pred_NSarima212 = np.zeros((1, 16))

    for i in range(1):
        yield_pred_NSarima111[i] = np.dot(H, beta_pred_1_111[i])
        yield_pred_NSarima012[i] = np.dot(H, beta_pred_1_012[i])
        yield_pred_NSarima212[i] = np.dot(H, beta_pred_1_212[i])

    # --- Forecast za M (CPI) i N (IP) ---
    model_M = SARIMAX(M, order=(1, 1, 1))
    result_M = model_M.fit(disp=False)
    M_future = result_M.get_forecast(steps=1).predicted_mean

    model_N = SARIMAX(N, order=(1, 1, 1))
    result_N = model_N.fit(disp=False)
    N_future = result_N.get_forecast(steps=1).predicted_mean

    exog_future = np.column_stack([M_future, N_future])

    #######################################

    # VARIMA(1,1) SA EXOG
    gamma_df = pd.DataFrame(gamma, columns=['g1', 'g2', 'g3'])
    exog_df = pd.DataFrame({'M': M, 'N': N})

    model_varima11 = VARMAX(gamma_df, exog=exog_df, order=(1, 1))
    result11 = model_varima11.fit(disp=False)
    print(result11.summary())
    gamma_pred_1_varima_11 = result11.forecast(steps=1, exog=exog_future).values

    # VARIMA(0,2) SA EXOG
    model_varima02 = VARMAX(gamma_df, exog=exog_df, order=(0, 2))
    result02 = model_varima02.fit(disp=False)
    print(result02.summary())
    gamma_pred_1_varima_02 = result02.forecast(steps=1, exog=exog_future).values

    # VARIMA(2,2) SA EXOG
    model_varima22 = VARMAX(gamma_df, exog=exog_df, order=(2, 2))
    result22 = model_varima22.fit(disp=False)
    print(result22.summary())
    gamma_pred_1_varima_22 = result22.forecast(steps=1, exog=exog_future).values

    # VARIMA(1,1) BEZ EXOG
    gamma_df = pd.DataFrame(gamma, columns=['g1', 'g2', 'g3'])

    model_varima11_no = VARMAX(gamma_df, order=(1, 1))
    result11_no = model_varima11_no.fit(disp=False)
    print(result11_no.summary())
    gamma_pred_1_varima_11_no = result11_no.forecast(steps=1).values

    # VARIMA(0,2) BEZ EXOG
    model_varima02_no = VARMAX(gamma_df, order=(0, 2))
    result02_no = model_varima02_no.fit(disp=False)
    print(result02_no.summary())
    gamma_pred_1_varima_02_no = result02_no.forecast(steps=1).values

    # VARIMA(2,2) BEZ EXOG
    model_varima22_no = VARMAX(gamma_df, order=(2, 2))
    result22_no = model_varima22_no.fit(disp=False)
    print(result22_no.summary())
    gamma_pred_1_varima_22_no = result22_no.forecast(steps=1).values

    yield_macro_pred_varima_11 = np.zeros((1, len(maturity)))
    yield_macro_pred_varima_02 = np.zeros((1, len(maturity)))
    yield_macro_pred_varima_22 = np.zeros((1, len(maturity)))

    yield_macro_pred_varima_11_no = np.zeros((1, len(maturity)))
    yield_macro_pred_varima_02_no = np.zeros((1, len(maturity)))
    yield_macro_pred_varima_22_no = np.zeros((1, len(maturity)))

    for i in range(1):
        yield_macro_pred_varima_11[i] = np.dot(G, gamma_pred_1_varima_11[i])
        yield_macro_pred_varima_02[i] = np.dot(G, gamma_pred_1_varima_02[i])
        yield_macro_pred_varima_22[i] = np.dot(G, gamma_pred_1_varima_22[i])


        yield_macro_pred_varima_11_no[i] = np.dot(G, gamma_pred_1_varima_11_no[i])
        yield_macro_pred_varima_02_no[i] = np.dot(G, gamma_pred_1_varima_02_no[i])
        yield_macro_pred_varima_22_no[i] = np.dot(G, gamma_pred_1_varima_22_no[i])

    #######################################

    gamma_pred_macro_111 = np.zeros((1, ncol))
    gamma_pred_111 = np.zeros((1, ncol))

    gamma_pred_macro_012 = np.zeros((1, ncol))
    gamma_pred_012 = np.zeros((1, ncol))

    gamma_pred_macro_212 = np.zeros((1, ncol))
    gamma_pred_212 = np.zeros((1, ncol))

    for j in range(3):
        series = gamma[:, j]
        exog = np.column_stack([M, N])

        # --- ARIMA(1,1,1) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(1, 1, 1))
        result_macro = model_macro.fit(disp=False)
        print(f"ARIMA(1,1,1) + macro fit za gamma_{j}: AIC={result_macro.aic:.2f}")

        forecast_macro = result_macro.get_forecast(steps=1, exog=exog_future)
        gamma_pred_macro_111[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(1,1,1) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(1, 1, 1))
        result_plain = model_plain.fit(disp=False)
        print(f"ARIMA(1,1,1) fit za gamma_{j}: AIC={result_plain.aic:.2f}")

        forecast_plain = result_plain.get_forecast(steps=1)
        gamma_pred_111[:, j] = forecast_plain.predicted_mean

        # --- ARIMA(0,1,2) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(0, 1, 2))
        result_macro = model_macro.fit(disp=False)
        print(f"ARIMA(0,1,2) + macro fit za gamma_{j}: AIC={result_macro.aic:.2f}")

        forecast_macro = result_macro.get_forecast(steps=1, exog=exog_future)
        gamma_pred_macro_012[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(0,1,2) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(0, 1, 2))
        result_plain = model_plain.fit(disp=False)
        print(f"ARIMA(0,1,2) fit za gamma_{j}: AIC={result_plain.aic:.2f}")

        forecast_plain = result_plain.get_forecast(steps=1)
        gamma_pred_012[:, j] = forecast_plain.predicted_mean

        # --- ARIMA(2,1,2) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(2, 1, 2))
        result_macro = model_macro.fit(disp=False)
        print(f"ARIMA(2,1,2) + macro fit za gamma_{j}: AIC={result_macro.aic:.2f}")

        forecast_macro = result_macro.get_forecast(steps=1, exog=exog_future)
        gamma_pred_macro_212[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(2,1,2) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(2, 1, 2))
        result_plain = model_plain.fit(disp=False)
        print(f"ARIMA(2,1,2) fit za gamma_{j}: AIC={result_plain.aic:.2f}")

        forecast_plain = result_plain.get_forecast(steps=1)
        gamma_pred_212[:, j] = forecast_plain.predicted_mean

    yield_macro_pred_111 = np.zeros((1, len(maturity)))
    yield_pred_111 = np.zeros((1, len(maturity)))

    yield_macro_pred_012 = np.zeros((1, len(maturity)))
    yield_pred_012 = np.zeros((1, len(maturity)))

    yield_macro_pred_212 = np.zeros((1, len(maturity)))
    yield_pred_212 = np.zeros((1, len(maturity)))

    for i in range(1):
        yield_macro_pred_111[i] = np.dot(G, gamma_pred_macro_111[i])
        yield_pred_111[i] = np.dot(G, gamma_pred_111[i])

        yield_macro_pred_012[i] = np.dot(G, gamma_pred_macro_012[i])
        yield_pred_012[i] = np.dot(G, gamma_pred_012[i])

        yield_macro_pred_212[i] = np.dot(G, gamma_pred_macro_212[i])
        yield_pred_212[i] = np.dot(G, gamma_pred_212[i])


    actual = yield_data_test[0, :]
    # rf = [4.81285622758086, 4.89442261694163, 4.9183626962914, 4.89795320962241, 4.83699664089617, 4.75031138,
    #       4.6520152, 4.55914328, 4.4164746,
    #       4.32303625, 4.21945028, 4.14578196, 4.11135559, 4.06991674, 4.03534654,
    #       4.04598]

    # bey grid serch
    # rf = [
    #     4.878029545,
    #     4.927690644,
    #     4.932292374,
    #     4.897124082,
    #     4.822882968,
    #     4.725309557,
    #     4.619574331,
    #     4.521338804,
    #     4.371175811,
    #     4.274405873,
    #     4.168934881,
    #     4.093469087,
    #     4.057703868,
    #     4.015116248,
    #     3.980693332,
    #     3.988498957,
    # ]

    # rf = [4.92818822,4.99297566, 4.98840899 ,4.93837431 ,4.83738596, 4.70447461, 4.56389815, 4.4336309 , 4.22859274 ,4.09900771 ,3.95629027 ,3.85559591, 3.80683611 ,3.73557165, 3.67428615 ,3.66637929]
    # sa grid search

    rf = [
        4.878029545,
        4.927690644,
        4.932292374,
        4.897124082,
        4.822882968,
        4.725309557,
        4.619574331,
        4.521338804,
        4.371175811,
        4.274405873,
        4.168934881,
        4.093469087,
        4.057703868,
        4.015116248,
        3.980693332,
        3.988498957
    ]

    # xgb = [
    #                              4.939376,
    #                              5.040331,
    #                              5.067470,
    #                             5.049196,
    #                             4.988652,
    #                             4.898736,
    #                             4.796299,
    #                             4.699748,
    #                             4.547868,
    #                             4.429415,
    #                            4.268754,
    #                             4.128718,
    #                             4.033520,
    #                             3.929795,
    #                            3.787867,
    #                            3.740452 ]

    xgb = [
                                 4.980000,
                                5.080909,
                                5.107139,
                                5.087980,
                                5.024670,
                                4.930241,
                                4.823075,
                                4.722037,
                                4.563195,
                                4.440800,
                               4.276016,
                               4.132995,
                                4.036328,
                               3.927398,
                               3.776308,
                               3.725283
    ]

    rmse_rdns_macro_012 = np.sqrt(mean_squared_error(actual, yield_macro_pred_012[0, :]))
    rmse_rdns_012 = np.sqrt(mean_squared_error(actual, yield_pred_012[0, :]))
    rmse_dns_012 = np.sqrt(mean_squared_error(actual, yield_pred_NSarima012[0, :]))
    rmse_varima_012 = np.sqrt(mean_squared_error(actual, yield_macro_pred_varima_02[0, :]))
    rmse_varima_012_no = np.sqrt(mean_squared_error(actual, yield_macro_pred_varima_02_no[0, :]))
    rmse_varima_11_no = np.sqrt(mean_squared_error(actual, yield_macro_pred_varima_11_no[0, :]))
    rmse_xgb = np.sqrt(mean_squared_error(actual, xgb))
    rmse_rf = np.sqrt(mean_squared_error(actual, rf))

    print("=== RMSE - 1-Month Ahead Forecast ===")
    print(f"RDNS (with macro, ARIMA(0,1,2)): {rmse_rdns_macro_012:.6f}")
    print(f"RDNS (ARIMA(0,1,2)):             {rmse_rdns_012:.6f}")
    print(f"DNS  (ARIMA(0,1,2)):             {rmse_dns_012:.6f}")
    print(f"DNS  (VARIMA + macro(0,1,2)):             {rmse_varima_012:.6f}")
    print(f"DNS  (VARIMA + macro(0,1,2)):             {rmse_varima_012_no:.6f}")
    print(f"VARIMA(1,1,1) RDNS " , rmse_varima_11_no )
    print(f"Gradient Boosting: ", rmse_xgb)
    print(f"RF 1m :             {rmse_rf:.6f}")


    plt.figure(figsize=(10, 5))
    plt.plot(maturity, yield_data_test[0, :],color='black', linewidth=2, label="Actual")
    # plt.plot(maturity, yield_macro_pred_111[0, :],  'x--', color='blue', label="RDNS (with macro, ARIMA(1,1,1))")
    # plt.plot(maturity, yield_pred_111[0, :],  'x--', color='cornflowerblue', label="RDNS (ARIMA(1,1,1))")
    # plt.plot(maturity, yield_pred_NSarima111[0, :], 'x--', color='green', label="DNS (ARIMA(1,1,1))")
    # plt.plot(maturity, yield_macro_pred_varima_11[0, :],  'x--', color='red', label="RDNS (with macro, VARIMA(1,1,1))")
    plt.plot(maturity, yield_macro_pred_varima_11_no[0, :], 'x--', color='orange', label="RDNS (VARIMA(1,1,1))")

    # plt.plot(maturity, yield_data_test[0, :], label="Actual")
    # plt.plot(maturity, yield_macro_pred_212[0, :], 'x--', color='blue', label="RDNS (with macro, ARIMA(2,1,2))")
    plt.plot(maturity, yield_pred_212[0, :], 'x--',color='cornflowerblue', label="RDNS (ARIMA(2,1,2))")
    # plt.plot(maturity, yield_pred_NSarima212[0, :], 'x--',color='green', label="DNS (ARIMA(2,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_22[0, :], 'x--',color='red', label="RDNS (with macro, VARIMA(2,2))")
    # plt.plot(maturity, yield_macro_pred_varima_22_no[0, :], 'x--', color='orange', label="RDNS (VARIMA(2,2))")


    # plt.plot(maturity, yield_data_test[0, :], label="Actual")
    # plt.plot(maturity, yield_macro_pred_012[0, :], 'x--', label="RDNS (with macro, ARIMA(0,1,2))")
    plt.plot(maturity, yield_pred_012[0, :], 'x--',color='green', label="RDNS (ARIMA(0,1,2))")
    # plt.plot(maturity, yield_pred_NSarima012[0, :], 'x--', label="DNS (ARIMA(0,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_02[0, :], 'x--', label="RDNS (VARIMA(0,1,2))")


    plt.plot(maturity, rf, 'x--',color='red', label="Random Forest")
    plt.plot(maturity, xgb, 'x--',color='purple', label="Gradient Boosting")



    plt.title("1-Month Ahead Yield Curve Forecast: RDNS and Random Forest vs Actual curve (July 2023.)")
    plt.xlabel('Maturity (Months)')
    plt.ylabel('Yield (%)')
    plt.xticks(maturity, [f"{m}M" for m in maturity])
    plt.legend()
    plt.tight_layout()
    plt.show()






def NS_ols_stable(lambda_const, yield_data, maturity, alpha=1e-5):
    """
    Stabilna verzija NS_ols sa regularizacijom.

    Parameters:
    -----------
    lambda_const : float
        Nelson-Siegel lambda parameter
    yield_data : np.array
        Yield data matrix (n_periods x n_maturities)
    maturity : np.array
        Maturity vector
    alpha : float
        Regularization parameter (ridge regression)

    Returns:
    --------
    dict: beta, rmse, yield_ns
    """
    H = NS_factor_loading(lambda_const, maturity)
    n_factors = H.shape[1]

    # Regularizaciona matrica
    ridge_matrix = alpha * np.eye(n_factors)

    nobs = yield_data.shape[0]
    rmse = np.zeros(nobs)
    beta = np.zeros((nobs, n_factors))
    yield_ns = np.zeros_like(yield_data)

    for t in range(nobs):
        # Regularizovana procena
        XTX = np.dot(H.T, H) + ridge_matrix
        XTy = np.dot(H.T, yield_data[t])

        # Koristi stabilnije rešenje
        beta_t = np.linalg.lstsq(XTX, XTy, rcond=None)[0]

        # Alternativno: np.linalg.solve (brže, ali manje stabilno)
        # beta_t = np.linalg.solve(XTX, XTy)

        yield_ns_t = np.dot(H, beta_t)

        beta[t, :] = beta_t
        rmse[t] = np.sqrt(np.mean((yield_data[t] - yield_ns_t) ** 2))
        yield_ns[t, :] = yield_ns_t

    return {'beta': beta, 'rmse': rmse, 'yield_ns': yield_ns}


def rmse_volatility(rmse):
    window = 6
    volatility = pd.Series(rmse).rolling(window).std()
    top_periods = volatility.nlargest(20)
    # print(top_periods)

    # print("**************")
    # print("Promene izmedju susednih dana: ")
    # diff = np.abs(np.diff(rmse))
    #
    # top_k = 10
    # volatile_inx = np.argsort(diff)[-top_k:]
    #
    # print("Najvece promene izmedju tacaka:")
    # print(volatile_inx)


def test_for_stationarity(ts):
    adf_result = adfuller(ts)
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")

    if adf_result[1] < 0.05:
        print("➡️ Serija je STACIONARNA (odbacuje se H0)")
    else:
        print("➡️ Serija NIJE stacionarna (ne odbacuje se H0)")

    kpss_result = kpss(ts, regression='c', nlags="auto")

    print("\nKPSS Statistic:", kpss_result[0])
    print("p-value:", kpss_result[1])
    print("Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value}")

    if kpss_result[1] < 0.05:
        print("➡️ Serija NIJE stacionarna (odbacuje se H0)")
    else:
        print("➡️ Serija je STACIONARNA (ne odbacuje se H0)")


def main():
    tau_s = (3 / 12)
    data, yield_data, maturity, CPI, IP = data_loading_fp('C:/Users/andjela.djurovic/Desktop/srednji.xlsx')
    data_test, yield_data_test, maturity, CPI_test, IP_test = data_loading_lw_test(
        'C:/Users/andjela.djurovic/Desktop/test6s.xlsx')

    # lambdaaa
    lambda_grid = np.linspace(0.01, 2, 100)  # OVO JE TAČNO
    avg_rmse = []

    for lamb in lambda_grid:
        result = NS_ols_stable(lamb, yield_data, maturity, alpha=1e-5)
        avg_rmse.append(np.mean(result['rmse']))

    best_lambda = lambda_grid[np.argmin(avg_rmse)]
    print(f"Optimal lambda: {best_lambda:.4f}")

    lambda_const = best_lambda

    # yt(T) = H * beta
    out_NS = NS_ols(lambda_const, yield_data, maturity)
    NS_rmse = out_NS['rmse']
    beta = out_NS['beta']

    yield_ns = out_NS['yield_ns']
    nobs, ncol = beta.shape

    if tau_s == 0:
        G, A = RNS_factor_loading(lambda_const, maturity)
    else:
        G, A = RNS_factor_loading_tau_s(lambda_const, maturity, tau_s=(3 / 12))

    out_RNS = RNS_ols(yield_data, G)
    gamma = out_RNS['gamma']
    yield_ols_RNS = out_RNS['yield_rns']
    RNS_rmse = out_RNS['rmse']

    # stats
    factor_stats(beta, gamma)

    corr_beta = np.corrcoef(beta.T)  # Korelaciona matrica 3x3
    print("Correlation matrix of beta factors:")
    print(corr_beta)

    corr_gamma = np.corrcoef(gamma.T)  # Korelaciona matrica 3x3
    print("Correlation matrix of gamma factors:")
    print(corr_gamma)

    # Proverite da li su yields previše korelisani
    corr_yields = np.corrcoef(yield_data.T)
    print("\nCondition number of yield correlation matrix:", np.linalg.cond(corr_yields))

    #end stats

    # plot_beta_factors_NS_RNS(beta, NS_rmse, gamma, RNS_rmse, date_list)
    plot_yield_curve_two_graphs(maturity, yield_data, yield_ns, yield_ols_RNS, "DNS model ", "RDNS model")
    plot_yield_curve_one_graph(maturity, yield_data, yield_ns, yield_ols_RNS, "DNS model ", "RDNS model", NS_rmse, RNS_rmse)

    rmse_volatility(RNS_rmse)


    dates = pd.date_range(start='2007-01-01', periods=len(beta), freq='MS')

    df = pd.DataFrame(beta, columns=['level', 'slope', 'curvature'], index=dates)
    df.index.name = 'date'

    # df.to_csv('betas.csv')
    #
    # test_for_stationarity(CPI)
    # rmse_volatility(RNS_rmse)
    # beta_graf(beta)
    # test_for_stationarity(np.diff(beta[:, 2]))
    # diff beta
    # beta_graf(beta[:, 2])

    mi_ns, F_ns, v_t_ns = OLS_beta_pred(beta)
    beta_pred = calculate_beta_pred_ols(mi_ns, F_ns, beta[0], v_t_ns, nobs)
    plot_beta_pred(beta, beta_pred)

    M = CPI.squeeze().to_numpy()
    N = IP.squeeze().to_numpy()

    predict1m_more_rotatedARIMAS(maturity, yield_data_test, nobs, ncol, gamma, M, N, beta, mi_ns, F_ns, v_t_ns, lambda_const)
    predict6m_more_rotatedARIMAS(maturity, yield_data_test, nobs, ncol, gamma, M, N, beta, mi_ns, F_ns, v_t_ns, lambda_const)



if __name__ == "__main__":
    main()