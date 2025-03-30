import azure.functions as func
import logging
import json
import base64
import requests

from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
import os

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SearchIndex,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    VectorSearchAlgorithmConfiguration
)

from azure.storage.blob import BlobServiceClient

def load_api_key(file_path):
    try:
        with open(file_path, "r") as file:
            api_key = file.read().strip()  
        return api_key
    except FileNotFoundError:
        raise Exception(f"The file '{file_path}' containing the API key was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the API key: {e}")
    
cwd = os.getcwd()
# os.chdir('Azure Function')
# print("Current" + cwd)
# # Set endpoints for Azure services
# AZURE_SEARCH_SERVICE: str = load_api_key(cwd+'\endpoints\AZURE_SEARCH_SERVICE.txt')
# AZURE_OPENAI_ACCOUNT: str = load_api_key(cwd+'\endpoints\AZURE_OPENAI_ACCOUNT.txt')
# AZURE_AI_MULTISERVICE_ACCOUNT: str = load_api_key(cwd+'\endpoints\AZURE_AI_MULTISERVICE_ACCOUNT.txt')
# storage_connection_string = load_api_key(cwd+'\endpoints\storage_connection_string.txt')
# vision_endpoint = load_api_key(cwd+'\endpoints\vision_endpoint.txt')

# # Set API keys for Azure services
# azure_ai_multiservice_key = load_api_key(cwd+'\keys\azure_ai_multiservice_key.txt')
# azure_openai_key = load_api_key(cwd+'\keys\azure_openai_key.txt')
# key_azure_ai_search_service =  load_api_key(cwd+'\keys\key_azure_ai_search_service.txt')
# vision_api_key = load_api_key(cwd+'\keys\vision_api_key.txt')

# Set endpoints for Azure services
AZURE_SEARCH_SERVICE: str = ""
AZURE_OPENAI_ACCOUNT: str = ""
AZURE_AI_MULTISERVICE_ACCOUNT: str = ""
storage_connection_string = ""
vision_endpoint = ""

# Set API keys for Azure services
azure_ai_multiservice_key = ""
azure_openai_key = ""
key_azure_ai_search_service =  ""
vision_api_key = ""

index_name = "b2t2-project-vector-advance"
container_name = "b2t2projectcontainer"

def find_similar_images(image_input):
    """
    Processes an image (URL or binary data), generates a vector, performs a vector search,
    and returns matching image file names from Azure Blob Storage.
    """
    
    # Read image data
    if isinstance(image_input, str) and image_input.startswith("http"):
        response = requests.get(image_input)
        if response.status_code == 200:
            image_data = response.content
        else:
            raise ValueError(f"Failed to fetch image. HTTP Status: {response.status_code}")
    elif isinstance(image_input, bytes):
        image_data = image_input
    else:
        return []

    # Generate image vector using Azure Computer Vision
    url = f"{vision_endpoint}/computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
    headers = {
        "Ocp-Apim-Subscription-Key": vision_api_key,
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(url, headers=headers, data=image_data)

    if response.status_code != 200:
        raise Exception(f"ERROR {response.status_code}: {response.text}")

    image_vector = response.json().get("vector")

    # Perform vector search in Azure AI Search
    search_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE, credential=AzureKeyCredential(key_azure_ai_search_service), index_name=index_name)
    vector_query = VectorizedQuery(vector=image_vector, k_nearest_neighbors=3, fields="image_vector")
    results = search_client.search(search_text="", vector_queries=[vector_query], select=["title"], top=10)

    SCORE_THRESHOLD = 0.8
    filtered_results = [result for result in results if result["@search.score"] >= SCORE_THRESHOLD][:2]

    if not filtered_results:
        return []

    # Retrieve matching files from Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    if not container_client.exists():
        raise Exception("Azure Blob Storage container does not exist")

    matching_images = []
    for result in filtered_results:
        file_name = result["title"]
        blob_client = container_client.get_blob_client(file_name)
        download_stream = blob_client.download_blob()
        matching_images.append(download_stream.readall())

    return matching_images

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    
    
    try:
        req_body = req.get_json()
        image_input = req_body.get("image_input")

        if not image_input:
            return func.HttpResponse("Missing 'image_input' parameter", status_code=400)

        # Call find_similar_images function
        binary_images = find_similar_images(image_input)

        # Convert binary data to base64
        encoded_images = [base64.b64encode(img).decode("utf-8") for img in binary_images]

        return func.HttpResponse(
            json.dumps({"images": encoded_images}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)



    # name = req.params.get('name')
    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    # if name:
    #     return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    # else:
    #     return func.HttpResponse(
    #          "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
    #          status_code=200
    #     )