import argparse
import math
import os

import h2o
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default=None)
parser.add_argument("--model_path", default=None)
parser.add_argument("--output_path", default=None)
ms = parser.parse_args()


if ms.input_path is None:
    raise ValueError('Please provide the path of your input data.')

if ms.model_path is None:
    raise ValueError('Please provide the path of the model file.')

if ms.output_path is None:
    raise ValueError('Please provide the path to save the results.')

if os.path.isdir(ms.input_path):
    raise IOError('Provided input path is a directory. \
Please provide the path to the data file.')

if os.path.isdir(ms.model_path):
    raise IOError('Provided model path is a directory. \
Please provide the path to the model file.')

if os.path.isdir(ms.output_path):
    raise IOError('Provided output path is a directory.\
Please specify the file path to store the results.')

# Initialize h2o
h2o.init()

# define functions to engineer the feature hour
def hour_x(hour):
    return math.sin(2 * math.pi * hour / 24)

def hour_y(hour):
    return math.cos(2 * math.pi * hour / 24)

def prepare_test_data(input_path=ms.input_path):
    '''Prepare the test dataset

    Parameters
    ----------
    input_path: str
        Path to the saved input data.

    Returns
    -------
    test: H2O.DataFrame
        Test dataset after the same feature engineering. 
    '''
# jason_path = '../../Data Science/data_to_predict.json'
# test = pd.read_json(jason_path, lines = True)
    test = pd.read_json(input_path, lines=True)
    print 'Using input data from %s' % input_path
    # Engineer the features of test data accordingly
    test.created_at = pd.to_datetime(test.created_at)
    test['weekday'] = test.created_at.map(lambda x: x.weekday())
    test['hour'] = test.created_at.map(lambda x: x.hour)
    test['hour_x'] = test.hour.map(hour_x)
    test['hour_y'] = test.hour.map(hour_y)
    test = h2o.H2OFrame(test)
    return test

def predict(model_path=ms.model_path,
            output_path=ms.output_path):
    '''Make predictions

    Parameters
    ----------
    model_path: str
        Path to the saved model file.
    output_path: str
        Path to save the results.

    Returns
    -------
    None
    '''
    # load the saved model
    aml_top = h2o.load_model(model_path)
    # make predictions
    test = prepare_test_data()
    pred = aml_top.predict(test)
    res = pd.DataFrame()
    res['delivery_id'] = test.as_data_frame()['delivery_id']
    res['predicted_delivery_seconds'] = pred.as_data_frame()['predict']
    res.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    predict()
    h2o.cluster().shutdown()
    print 'Saving results to %s' % ms.output_path
