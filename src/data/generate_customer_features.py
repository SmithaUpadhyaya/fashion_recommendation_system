from src.data.features.user.user_purchase_elapsed_days_from_prev_tran import user_purchase_elapsed_days_from_prev_tran
from src.data.features.user.pct_type_ctb import calculate_percentage_of_product_type_user
import logs.logger as log
import argparse
import sys

if __name__ == '__main__':

#    try:

    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "config/config.yaml")
    
    parsed_args = args.parse_args()

    log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)
    #CONFIGURATION_PATH = parsed_args.config

    log.write_log(f'Started creating features for customer...', log.logging.DEBUG)

    print('Calculate user preference on product type started.')
    calculate_percentage_of_product_type_user(parsed_args.config)

    print('Calculate user purchase elapsed days from there previous transaction started.')
    user_purchase_elapsed_days_from_prev_tran(parsed_args.config)

    print('Generating features for customer completed.')

#    except Exception as e:
#        raise RecommendationException(e, sys) 



