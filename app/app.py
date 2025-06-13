import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import seaborn as sns
import logging
from catboost import CatBoostClassifier
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.model_features = [
            'amount',
            'bearing_degree_1',
            'cat_id',
            'gender',
            'hav_dist_1',
            'jobs',
            'lat',
            'lon',
            'merch',
            'name_1',
            'name_2',
            'one_city',
            'population_city',
            'street',
            'us_state',
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'day_of_week'
        ]

        self.numeric_feat = [
            'amount',
            'lat',
            'lon',
            'population_city',
            'merchant_lat',
            'merchant_lon',
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'day_of_week',
            'bearing_degree_1',
            'hav_dist_1'
        ]
        self.train = load_train_data()
        self.model = CatBoostClassifier()
        self.model.load_model('./models/catboost_model.cbm') 
        logger.info('Service initialized')


    def save_feature_importances(self, output_path):
        try:
            with open("./models/feature_importances.json") as f:
                importance_dict = json.load(f)

            top_5 = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

            with open(output_path, 'w') as f:
                json.dump(top_5, f, indent=4)

            logger.info("Top-5 feature importances saved to: %s", output_path)

        except Exception as e:
            logger.error("Failed to save feature importances: %s", e)



    def save_prediction_density_plot(self, predictions, output_path):
        try:
            plt.figure(figsize=(8, 5))
            plt.title("Prediction Score Distribution")
            sns.kdeplot(predictions, fill=True)
            plt.xlabel("Prediction Score")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info("Prediction density plot saved to: %s", output_path)

        except Exception as e:
            logger.error("Failed to save prediction density plot: %s", e)

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df, self.model_features, self.numeric_feat)

            logger.info('Making prediction')
            
            probs, submission = make_pred(processed_df, file_path, return_raw=True)
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            filename = os.path.basename(file_path)
            feature_importances_path = os.path.join(self.output_dir, f"feature_importance_{timestamp}_{filename}.json")
            self.save_feature_importances(feature_importances_path)

            # Save prediction density plot
            density_plot_path = os.path.join(self.output_dir, f"prediction_density_{timestamp}_{filename}.png")
            self.save_prediction_density_plot(probs, density_plot_path)

            logger.info('All outputs saved for file: %s', file_path)


        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()