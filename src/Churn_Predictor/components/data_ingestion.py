import os
import zipfile
from src.Churn_Predictor.entity.config_entity import DataIngestionConfig
from src.Churn_Predictor import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """Download datasets from kaggle using kaggle API"""
        if not os.path.exists(self.config.local_data_file):
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()

                logger.info(f"Downloading dataset: {self.config.dataset_slug}")
                api.dataset_download_files(
                    self.config.dataset_slug, 
                    path=self.config.root_dir, 
                    unzip=False
                )
                
                # Rename the downloaded file
                downloaded_file = os.path.join(
                    self.config.root_dir, 
                    f"{self.config.dataset_slug.split('/')[-1]}.zip"
                )
                
                if os.path.exists(downloaded_file):
                    os.rename(downloaded_file, self.config.local_data_file)
                
                logger.info(f"Dataset downloaded successfully to: {self.config.local_data_file}")

            except Exception as e:
                logger.error(f"Error downloading dataset: {e}")
                raise e
        else:
            logger.info(f"Dataset already exists at: {self.config.local_data_file}")
    
    def extract_zip_file(self):  # ← Fixed indentation (was nested inside download_data)
        """Extract the downloaded zip file"""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Dataset extracted successfully to: {unzip_path}")
        except Exception as e:
            logger.error(f"Error extracting dataset: {e}")
            raise e
    
    def fetch_and_prepare_data(self):
        """Main method: download and extract data"""
        self.download_data()
        self.extract_zip_file()