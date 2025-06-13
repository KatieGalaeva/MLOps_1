import pandas as pd
import logging
from catboost import CatBoostClassifier
from catboost import Pool


logger = logging.getLogger(__name__)
logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/catboost_model.cbm')

# Define optimal threshold
model_th = 0.5
logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, path_to_file, return_raw=False):
     # Return proba for positive class
    cat_features = [
        'merch', 'cat_id', 'name_1', 'name_2', 'gender', 'street', 'one_city', 'us_state', 'jobs', 'year',	'month',	'day',	'hour',	'minute'	,'day_of_week'
    ]
    for col in cat_features:
        if col in dt.columns:
            dt[col] = dt[col].astype(str)

    pool = Pool(dt, cat_features=cat_features) 
    probs = model.predict_proba(pool)[:, 1]


    raw_df = pd.read_csv(path_to_file)

     # Make submission dataframe
    submission = pd.DataFrame({
        'index': raw_df.index,
        'prediction': (probs > model_th).astype(int)
    })

    logger.info('Prediction complete for file: %s', path_to_file)

    if return_raw:
        return probs, submission
    else:
        return submission
