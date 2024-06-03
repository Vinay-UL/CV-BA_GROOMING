from azureml.core.compute import ComputeTarget, ComputeInstance
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace

import json

cli_auth = AzureCliAuthentication()
print("cli_auth",cli_auth)
# Get workspace
ws = Workspace.from_config(path="./PackBenchmark_Azure_Devops/aml_config",auth=cli_auth)

with open("./PackBenchmark_Azure_Devops/aml_config/security_config.json") as f:
    config = json.load(f)

remote_vm_name = config["remote_vm_name"]
remote_vm_size = config["remote_vm_size"]

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=remote_vm_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size=remote_vm_size, max_nodes=1)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
