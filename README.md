- The `artifacts` folder contains all the saved models according to timestamps (the folder name is the timestamp of when the model was uploaded to the gcs bucket).
- It also contains the `v1` and `v2` directories which contain their own stored models according to timestamp.

- The data folder contains the input data files splitted in train and eval datasets.
- The `v1` and `v2` folders within the data folder contain the splitted data for the v1 and v2 datasets given. 