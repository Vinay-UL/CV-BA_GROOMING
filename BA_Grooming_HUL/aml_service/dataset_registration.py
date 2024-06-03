import os, json, sys
from azureml.core import Workspace
from azureml.core import Run
from azureml.core import Experiment
from azureml.core.model import Model
from azureml.core import  Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)

# Get the default datastore
default_ds = ws.get_default_datastore()

with open("aml_config/dataset_registration_config.json") as f:
    config = json.load(f)

file_dataset=Dataset.File.from_files(path=(default_ds, config['datastore_name']))

description=["Modified rexona text class dataset"]
#description=["Shoe class dataset consists of with shoe and without shoe images"]

file_dataset = file_dataset.register(workspace=ws,
                                     name=config['dataset_register_name'],
                                     description=description,
                                     create_new_version=['True'])

dataset=Dataset.get_by_name(ws,'modified_perfume_rexona_test_images')
print(dataset)