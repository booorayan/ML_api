# ML_api
REST API for a machine learning model that performs predictive classification of penguin species.
To run the app through uvicorn use the command:
  
 `uvicorn api:app --reload`

One can test the API in the Swagger UI at http://127.0.0.1:8000/docs

Using curl one can test prediction as below:

`curl -X 'POST' 'http://localhost:8000/predict/' 
 -H 'accept: application/json' -H 'Content-Type: application/json' 
 -d '{
 "culmen_length_mm": 49.9,
 "culmen_depth_mm": 16.1,
 "flipper_length_mm": 213.0,
 "body_mass_g": 5400.0,
 "sex": 1
}'`

The expected response/prediction is:

`{"prediction":"Gentoo","probability":0.99}`

## Todo:
1 [] Hyperparameter tuning to optimize the model
