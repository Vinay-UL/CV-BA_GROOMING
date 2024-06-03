import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.model import InferenceConfig
from azureml.core.image import ContainerImage
from azureml.core.compute import AksCompute
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.authentication import AzureCliAuthentication

cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)
print('Printing the model details')
# # Get the Image to deploy details
# try:
#     with open("aml_config/model_uniform.json") as f:
#         config = json.load(f)
# except:
#     print("No model, thus no deployment on AKS")
#     # raise Exception('No new model to register as production model perform better')
#     sys.exit(0)

# print(config)

os.chdir("./code/scoring")

image_config = ContainerImage.image_configuration(execution_script ='uniform_score.py', 
                                    runtime = "python", conda_file = 'conda_dependencies_uniform.yml',
                                    dependencies =['model'])

image = ContainerImage.create(name = "uniform-classification-image",
                              models = [],
                              image_config = image_config,
                              workspace = ws)
image.wait_for_creation(show_output = True) 

aks_target = AksCompute(ws, 'azure-devops')

aks_config = AksWebservice.deploy_configuration()

aks_service_name ='uniform-deploy-service'
aks_service = Webservice.deploy_from_image(workspace = ws, 
                                           name = aks_service_name,
                                           image =image,
                                           deployment_config = aks_config,
                                           deployment_target = aks_target,
                                           overwrite=True)
aks_service.wait_for_deployment(show_output = True)

aks_service.get_keys()

# Enable Azure Monitoring
aks_service.update(enable_app_insights=True)
print('AppInsights enabled!')

print(
    "Deployed AKS Webservice: {} \nWebservice Uri: {}".format(
        aks_service.name, aks_service.scoring_uri
    )
)
os.chdir('..')
os.chdir('..')
# service=Webservice(name ='aciws0622', workspace =ws)
# Writing the ACI details to /aml_config/aci_webservice.json
aks_webservice = {}
aks_webservice["aks_name"] = aks_service.name
aks_webservice["aks_url"] = aks_service.scoring_uri
with open("aml_config/uniform_aks_webservice.json", "w") as outfile:
    json.dump(aks_webservice, outfile)