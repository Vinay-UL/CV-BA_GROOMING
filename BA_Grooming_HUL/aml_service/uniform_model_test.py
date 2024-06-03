import os, json
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Dataset
from azureml.core.compute import RemoteCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import ComputeTarget, ComputeInstance
from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException

cli_auth = AzureCliAuthentication()
# Get workspace
ws = Workspace.from_config(auth=cli_auth)


# Attach Experiment
experiment_name = 'uniform_test_exp'
exp = Experiment(workspace=ws, name=experiment_name)
print(exp.name, exp.workspace.name, sep="\n")

with open("aml_config/dataset_config_uniform.json") as f:
    config2 = json.load(f)

dataset=Dataset.get_by_name(ws,config2['model_dataset_test'])


myenv='uniform_env'
try:
    # Check for existing environment
    myenv = Environment.get(workspace=ws, name=myenv)
    print('Found existing environment, use it.')
except:
    # If it doesn't already exist, create it
    try:
        # From a pip requirements file
        myenv = Environment.from_pip_requirements(name = "uniform_env",file_path = "./environment_setup/shoe_class_requirements.txt")
        # Register environment
        myenv.register(workspace=ws) 
    except Exception as ex:
        print(ex)

with open("aml_config/security_config.json") as f:
    config = json.load(f)
cluster_name = config["remote_vm_name"]
remote_vm_size=config["remote_vm_size"]

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size=remote_vm_size, max_nodes=1)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)

# Use when snapshot size exceeds more than 300MB
import azureml._restclient.snapshots_client
azureml._restclient.snapshots_client. SNAPSHOT_MAX_SIZE_BYTES =1000000000
src = ScriptRunConfig(
    source_directory="./code/scoring", script="uniform_test.py",
    arguments=[dataset.as_named_input('input').as_mount()],
    compute_target=cluster_name,
    environment=myenv
)
run = exp.submit(src)

# Shows output of the run on stdout.
run.wait_for_completion(show_output=True, wait_post_processing=True)


 
