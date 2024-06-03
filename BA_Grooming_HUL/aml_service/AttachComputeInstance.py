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
    print('Found existing instance, using it.')
    if instance.get_status().state =='Stopped':
        print("Instance is stopped.. Starting it now")
        instance.start(wait_for_completion=True, show_output=True)


except ComputeTargetException:
    print("Creating New Training Instance")
    compute_config = ComputeInstance.provisioning_configuration(
        vm_size=remote_vm_size,
        ssh_public_access=False,
        # vnet_resourcegroup_name='<my-resource-group>',
        # vnet_name='<my-vnet-name>',
        # subnet_name='default',
        # admin_user_ssh_public_key='<my-sshkey>'
    )
    instance = ComputeInstance.create(ws, remote_vm_name, compute_config)
    instance.wait_for_completion(show_output=True)