# from experimentation import BasicPipeline, DataProcessorABC, ModelGeneratorABC, ModelPublisherABC

def main():
    from os import path
    import numpy as np
    import cv2
    import json
    import requests

    url = 'http://localhost:8500/invocations?stage=staging'
    headers = {
        'Content-Type': 'application/json'
    }

    image_data = cv2.imread(path.join('experimentation', 'base-data', 'test', 'angry', 'PrivateTest_88305.jpg'),
                            cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (48, 48))
    image_data = np.array(image_data).reshape((-1, 48, 48, 1))
    image_data = image_data / 255

    json_data = {
        'inputs': image_data.tolist()
    }

    print(json_data)

    response = requests.request('POST', headers=headers, url=url, json=json_data)
    if response.status_code != 200:
        print('FAAEEEENNNNNNNNN!')
        print(f'Status code: \t{response.status_code}')
        print(f'Response: \t{response.text}')
    else:
        result = json.loads(response.text)

        print('Tack så mycket!')
        print(f'Status code: \t{response.status_code}')
        print(f'Response: \t{result}')

    print(f'Hello, world!')


if __name__ == '__main__':
    # import os
    # import mlflow
    #
    # os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    # os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
    # os.environ['MLFLOW_TRACKING_URI'] = 'http://blazoned.nl:8999'
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://blazoned.nl:9000'
    # os.environ['MODEL_NAME'] = 'onnx emotion model'
    #
    # calibrated_model = mlflow.search_runs(experiment_names=['ONNX Emotion Recognition'],
    #                                       filter_string='run_name = "v0.0.0"').iloc[0]
    # model_run_id = calibrated_model.loc['run_id']
    # loaded_model = mlflow.pyfunc.load_model(f'runs:/{model_run_id}/model')
    # print(model_run_id)
    main()
