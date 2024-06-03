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
import numpy,base64,requests
import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.authentication import AzureCliAuthentication

cli_auth = AzureCliAuthentication()
# Get workspace
ws = Workspace.from_config(auth=cli_auth)

import time
time.sleep(300)

# os.chdir('./blush_detection_folder')
# Get the AKS Details
try:
    with open("./aml_config/uniform_aks_webservice.json") as f:
        config = json.load(f)
except:
    print("No new model, thus no deployment on AKS")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)

service_name = config["aks_name"]
# Get the hosted web service
service = Webservice(workspace=ws, name=service_name)
print("service loaded successfully")

with open('./data/5a5fa4ff-e901-4300-8af4-4f9572fe98d5.JPG', 'rb') as f:
    encoded_img = base64.b64encode(f.read())
test = json.dumps({'image' : encoded_img.decode()})
print("test image for passed as input is loaded successfully")

key1, Key2 = service.get_keys()
url=config['aks_url']
print(key1)
print(url)

try:
    headers = {'Content-Type':'application/json',
           'Authorization': 'Bearer ' + key1}
    response=requests.post(url,data=test,headers = headers)
    print('Output Result: ',response.json())
#    print(service.get_logs())
except Exception as e:
    result = str(e)
    print(result)
    raise Exception("AKS service is not working as expected")

# Delete aks after test
service.delete()
