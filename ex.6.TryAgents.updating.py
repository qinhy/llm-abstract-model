
import requests
from LLMAbstractModel.utils import ClassificationTemplate, StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
debug = True

## add French Address Search function ( this )
@descriptions('Reverse geocode coordinates to an address', lon='longitude', lat='latitude')
class ReverseGeocodeFunction(Model4LLMs.Function):
    def __call__(self, lon: float, lat: float):
        # Construct the URL with the longitude and latitude parameters
        url = f"https://api-adresse.data.gouv.fr/reverse/?lon={lon}&lat={lat}"        
        # Perform the HTTP GET request
        response = requests.get(url)        
        # Check if the request was successful
        if response.status_code == 200:
            # Return the JSON data from the response
            return response.json()
        else:
            # Handle the error case
            return {'error': f"Request failed with status code {response.status_code}"}

# Define the French address agent
french_address_llm = store.add_new_chatgpt4omini(vendor_id='auto', system_prompt='''
You are familiar with France and speak English.
You will answer questions by providing information.
If you want to use an address searching by coordinates, please only reply with the following text:
```json
{"lon":2.37,"lat":48.357}
```
''')

# Add functions for reverse geocoding and address extraction
french_address_search_function = store.add_new_function(ReverseGeocodeFunction())
coordinates_extract = store.add_new_function(RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True))

# Workflow for querying French address agent and handling responses
def french_address_llm_workflow(question='I am in France and My GPS shows (47.665176, 3.353434), where am I?'):
    debugprint = lambda msg:print(f'--> [french_address_llm_workflow]: {msg}') if debug else lambda:None
    query = question
    while True:
        debugprint(f'Asking french_address_llm with: [{dict(question=query)}]')
        response = french_address_llm(query)
        coord_or_query = coordinates_extract(response)
        
        # If the response contains coordinates, perform a reverse geocode search
        if isinstance(coord_or_query, dict) and "lon" in coord_or_query and "lat" in coord_or_query:
            debugprint(f'Searching address with coordinates: [{coord_or_query}]')
            query = french_address_search_function(**coord_or_query)
            query = f'\n## Question\n{question}\n## Information\n```\n{query}\n```\n'
        else:
            # Return the final response if no coordinates were found
            answer = coord_or_query
            debugprint(f'Final answer: [{dict(answer=answer)}]')
            return answer

# try to build in Workflow class
french_address_workflow = store.add_new_workflow(
    tasks={
        french_address_llm.get_id()    : ['question'],
        coordinates_extract.get_id()   : [french_address_llm.get_id()],
        french_address_search_function.get_id()   : [coordinates_extract.get_id()],
    })

# Define the triage agent
triage_llm = store.add_new_chatgpt4omini(
    vendor_id='auto',
    system_prompt='''
You are a professional guide who can connect the asker to the correct agent.
## Available Agents:
- french_address_llm_workflow: Familiar with France and speaks English.

## How to connect the agent:
```agent
french_address_llm_workflow
```
'''
)

# Function to extract the agent from triage agent's response
agent_extract = store.add_new_function(RegxExtractor(regx=r"```agent\s*(.*)\s*\n```"))

# Workflow for handling triage queries and routing to the appropriate agent
def triage_llm_workflow(question='I am in France and My GPS shows (47.665176, 3.353434), where am I?'):
    debugprint = lambda msg:print(f'--> [triage_llm_workflow]: {msg}') if debug else lambda:None

    debugprint(f'Asking triage_llm with: [{dict(question=question)}]')    
    while True:
        # Get the response from triage_llm and extract the agent name
        response = triage_llm(question)
        agent_name = agent_extract(response)
        if agent_name not in ['french_address_llm_workflow']:
            continue
        debugprint(f'Switching to agent: [{agent_name}]')
        break    
    # Dynamically call the extracted agent
    return eval(agent_name)(question)

# Example usage
# answer = triage_llm_workflow('I am in France and My GPS shows (47.665176, 3.353434), where am I?')

# print(f'\n\n######## Answer ########\n\n{answer}')

# try to build in Workflow class
# coord_or_query_condition = store.add_new_obj(ClassificationTemplate())
# res = coord_or_query_condition(True,
#                          [lambda q:isinstance(q, dict) and "lon" in q and "lat" in q ,
#                           lambda q:True
#                           ])

# valid_agent = store.add_new_obj(ClassificationTemplate())
# res = valid_agent(target='french_address_llm_workflow',
#                   condition_funcs=[lambda q:q in ['french_address_llm_workflow'],])
# # Define the workflow with tasks
# triage_workflow = store.add_new_workflow(
#     tasks={
#         triage_llm.get_id()            : ['question'],
#         agent_extract.get_id()         : [triage_llm.get_id()],
#         eval(agent_extract)(question)  : [agent_extract.get_id(),'question'],
#     })
# print(res)