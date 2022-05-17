import argparse
import logging
import os
import json

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd


if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')
    parser.add_argument('--model-name', type=str, default='catboost_model.dump')
    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target
    

    args, _ = parser.parse_known_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logging.info('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    logging.info('building training and testing datasets')
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]
        
    # define and train model
    model = CatBoostRegressor()
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), logging_level='Silent') 
    
    # print abs error
    logging.info('validating model')
    abs_err = np.abs(model.predict(X_test) - y_test)

    # print couple perf metrics
    for q in [10, 50, 90]:
        logging.info('AE-at-' + str(q) + 'th-percentile: '
              + str(np.percentile(a=abs_err, q=q)))
    
    # persist model
    path = os.path.join(args.model_dir, args.model_name)
    logging.info('saving to {}'.format(path))
    model.save_model(path)

    
# Model serving
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = CatBoostRegressor()
    model.load_model(os.path.join(model_dir, "catboost_model.dump"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request_body = json.loads(request_body)
        inpVar = request_body["Input"]
        return inpVar
    else:
        raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""
def output_fn(prediction, content_type):
    res = prediction[0]
    respJSON = {"Output": res}
    return respJSON
