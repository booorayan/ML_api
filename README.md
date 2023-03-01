# API for Random Forest Classifier Model
REST API for a machine learning model(random forest classifier) that performs simple predictive classification of penguin species.

This project employs FastAPI to build the API.

To run the server through uvicorn use the command:
  
 `uvicorn api:app --reload`

## Make Prediction
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

Here, the columns: culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g and sex are the independent variables.
The 'species' column is the dependent variable.

The expected response/prediction is:

`{"prediction":"Gentoo","probability":0.99}`

## Todo:
1 [] Hyperparameter tuning to optimize the model

2 [x] Test prediction feature
