from azureml.core.compute import ComputeTarget, ComputeInstance
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace

import json

cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)

with open("aml_config/security_config.json") as f:
    config = json.load(f)

remote_vm_name = config["remote_vm_name"]
remote_vm_size = config["remote_vm_size"]
# STANDARD_D3_V2

try:
    instance = ComputeInstance(workspace=ws, name=remote_vm_name)
    print('Instance Found')
    print(instance.get_status().state)
    if instance.get_status().state == 'Running':
        instance.stop(wait_for_completion=True, show_output=True)
    elif instance.get_status().state == 'Stopped':
        print("Instance is already stopped")


except ComputeTargetException:
    print("Instance is not Present")
