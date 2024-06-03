"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
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
with open("aml_config/config.json") as f:
    config1 = json.load(f)

# Read the New VM Config
with open("aml_config/security_config.json") as f:
    config = json.load(f)
remote_vm_name = config["remote_vm_name"]
remote_vm_size = config["remote_vm_size"]

with open("aml_config/dataset_config_uniform.json") as f:
    config2 = json.load(f)

# Attach Experiment
experiment_name = 'uniform_exp'
exp = Experiment(workspace=ws, name=experiment_name)
print(exp.name, exp.workspace.name, sep="\n")


dataset=Dataset.get_by_name(ws,config2["name"])

#instance = ComputeInstance(workspace=ws, name=remote_vm_name)

# myenv = Environment.from_pip_requirements(name = "myenv",
#                                           file_path = "./environment_setup/requirements.txt")
#myenv = Environment.get(ws, name='customize_curated')

# myenv = Environment.from_conda_specification(name = "uniform_env",
#                                              file_path = "./environment_setup/uniform_requirements.txt")
# Editing a run configuration property on-fly.
#run_config_user_managed = RunConfiguration()
#run_config_user_managed.environment.python.user_managed_dependencies = True

# Editing a run configuration property on-fly.
# run_config_system_managed = RunConfiguration()
# # Use a new conda environment that is to be created from the conda_dependencies.yml file
# run_config_system_managed.environment.python.user_managed_dependencies = False
# # Automatically create the conda environment before the run
# run_config_system_managed.prepare_environment = True


# From a pip requirements file
# myenv = Environment.from_pip_requirements(name = "uniform_env",
#                                           file_path = "./environment_setup/shoe_class_requirements.txt")
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

cluster_name = remote_vm_name

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


src = ScriptRunConfig(
    source_directory="./code", script="training/train_uniform_classification.py",
    arguments=[dataset.as_named_input('input').as_mount()],
    compute_target=cluster_name,
    environment=myenv
)
run = exp.submit(src)

# Shows output of the run on stdout.
run.wait_for_completion(show_output=True, wait_post_processing=True)

# Raise exception if run fails
if run.get_status() == "Failed":
    raise Exception(
        "Training on local env failed with following run status: {} and logs: \n {}".format(
            run.get_status(), run.get_details_with_logs()
        )
    )

# Writing the run id to /aml_config/run_id.json
run_id = {}
run_id["run_id"] = run.id
run_id["experiment_name"] = run.experiment.name
with open("aml_config/run_id_uniform.json", "w") as outfile:
    json.dump(run_id, outfile)

os.chdir("./code/scoring")
print('Directory files:',os.listdir())

model_local_dir = "model"
os.makedirs(model_local_dir, exist_ok=True)

# model_local_dir = "shoeClassification_efficientNetB7_baseFreezed700"
# os.makedirs(model_local_dir, exist_ok=True)
# print('directory created with name: ',model_local_dir)
# Download Model to Project root directory

#download_folder = 'efficientNetB7_baseFreezed_Sigmoid'
model_name = "efficientNetB7_baseFreezed650_Sigmoid"

run.download_files(model_name, "./model/")


print("Downloaded model {} to Project root directory".format(model_name))       
