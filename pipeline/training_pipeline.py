from src.data_preprocessing import DataPreprocessor
from src.model_trainer import ModelTraining

if __name__=="__main__":
    processor = DataPreprocessor("artifacts/raw/data.csv" , "artifacts/processed")
    processor.run()

    trainer = ModelTraining("artifacts/processed/" , "artifacts/models/")
    trainer.run()