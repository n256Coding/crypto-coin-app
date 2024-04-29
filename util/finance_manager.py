from constant import LOSS, NEUTRAL, PROFIT
import pandas as pd


def calculate_expected_return(coin_data_df: pd.DataFrame, selected_coin_for_forecast: str, 
                          forecasted_dataset: pd.DataFrame, investing_amount_in_currency: float):
    
    today_price = coin_data_df[selected_coin_for_forecast].iloc[-1]
    forecasted_price = forecasted_dataset.iloc[-1].values[0]

    crypto_coins_today = (1 / today_price) * investing_amount_in_currency

    amount_in_currency_after_trading = crypto_coins_today * forecasted_price

    profit_loss = amount_in_currency_after_trading - investing_amount_in_currency
    if profit_loss > 0:
        indicator = PROFIT
    elif profit_loss < 0:
        indicator = LOSS
    else:
        indicator = NEUTRAL

    return f'{amount_in_currency_after_trading:.3f}', f'{abs(profit_loss):.3f}', indicator

