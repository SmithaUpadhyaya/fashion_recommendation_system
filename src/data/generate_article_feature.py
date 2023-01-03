from src.data.features.item.item_avg_median_sale_price import calculate_avg_median_sale_price_item
from src.data.features.item.item_convert_color_rgb import transform_item_color_rgb
from src.data.features.item.item_prev_days_popularity import calculate_item_prev_day_sales
import logs.logger as log
import argparse
import sys

if __name__ == '__main__':

    #try:

    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "config/config.yaml")
    
    parsed_args = args.parse_args()

    log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)
    #CONFIGURATION_PATH = parsed_args.config

    log.write_log(f'Started creating features for article...', log.logging.DEBUG)

    print('Feature to calculate avg median sales price of article started.')
    calculate_avg_median_sale_price_item(parsed_args.config)

    print('Feature to transform color of article in to RGB format started.')
    transform_item_color_rgb(parsed_args.config)

    print('Feature to calculate previous days sales of item started.')
    calculate_item_prev_day_sales(parsed_args.config)

    print('Generating features for article completed.')

    #except Exception as e:
    #    raise RecommendationException(e, sys) 