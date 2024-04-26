import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import warnings
warnings.filterwarnings('ignore')
def encoding(df):
        print("encoding called")
        mapping_airline = {'SpiceJet':0, 'AirAsia':1, 'Vistara':2, 'GO_FIRST':3, 'Indigo':4, 'Air_India':5}
        mapping_city = {'Delhi':0, 'Mumbai':1, 'Bangalore':2, 'Kolkata':3, 'Hyderabad':4, 'Chennai':5}
        mapping_time = {'Evening':0, 'Early_Morning':1, 'Morning':2, 'Afternoon':3, 'Night':4, 'Late_Night':5}
        mapping_stops = {'zero':0, 'one':1, 'two_or_more':2}
        mapping_class = {'Economy':0, 'Business':1}

        df['airline'] = df['airline'].map(mapping_airline)
        df['source_city'] = df['source_city'].map(mapping_city)
        df['destination_city'] = df['destination_city'].map(mapping_city)
        df['departure_time'] = df['departure_time'].map(mapping_time)
        df['arrival_time'] = df['arrival_time'].map(mapping_time)
        df['stops'] = df['stops'].map(mapping_stops)
        df['class'] = df['class'].map(mapping_class)

        return df

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_encoded=encoding(features)
            data_scaled=preprocessor.transform(data_encoded)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            print(e)
            raise CustomException(e,sys)
    


class CustomData:
    def __init__(  self,
        airline: str,
        source_city: str,
        destination_city,
        departure_time: str,
        arrival_time: str,
        stops: str,
        classes: str,
        duration: float,
        days_left: int):

        self.airline = airline

        self.source_city = source_city

        self.destination_city = destination_city

        self.arrival_time = arrival_time

        self.departure_time = departure_time

        self.stops = stops

        self.classes = classes

        self.duration = duration

        self.days_left = days_left

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "airline": [self.airline],
                "source_city": [self.source_city],
                "departure_time": [self.departure_time],
                "stops": [self.stops],
                "arrival_time": [self.arrival_time],
                "destination_city": [self.destination_city],
                "class": [self.classes],
                "duration": [self.duration],
                "days_left": [self.days_left]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

