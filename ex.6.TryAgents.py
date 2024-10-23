
import requests
from LLMAbstractModel.utils import ClassificationTemplate, StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
debug = True

## add French Address Search function
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
''',
id='ChatGPT4oMini:french_address_llm')

# Add functions for reverse geocoding and address extraction
french_address_search_function = store.add_new_obj(ReverseGeocodeFunction(),
                                                   id='ReverseGeocodeFunction:french_address_search_function')
first_json_extract = store.add_new_obj(RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True),
                                       id='RegxExtractor:first_json_extract')

# Workflow function for querying French address agent and handling responses
@descriptions('Workflow function of querying French address agent', question='The question to ask the LLM')
class FrenchAddressAgent(Model4LLMs.Function):
    french_address_llm_id:str
    first_json_extract_id:str
    french_address_search_function_id:str
    
    def __call__(self,question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [french_address_agent]: {msg}') if debug else lambda:None
        query = question
        french_address_llm = self.get_controller().storage().find(self.french_address_llm_id)
        first_json_extract = self.get_controller().storage().find(self.first_json_extract_id)
        french_address_search_function = self.get_controller().storage().find(self.french_address_search_function_id)
        
        while True:
            debugprint(f'Asking french_address_llm with: [{dict(question=query)}]')
            response = french_address_llm(query)
            coord_or_query = first_json_extract(response)
            
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

# Define the triage agent
triage_llm = store.add_new_chatgpt4omini(
    vendor_id='auto',
    system_prompt='''
You are a professional guide who can connect the asker to the correct agent.
## Available Agents:
- french_address_agent: Familiar with France and speaks English.

## How to connect the agent:
```agent
french_address_agent
```
''',
id='ChatGPT4oMini:triage_llm')

# Function to extract the agent from triage agent's response
agent_extract = store.add_new_obj(RegxExtractor(regx=r"```agent\s*(.*)\s*\n```"),id='RegxExtractor:agent_extract')

# Workflow for handling triage queries and routing to the appropriate agent
@descriptions('Workflow function of triage queries and routing to the appropriate agent', question='The question to ask the LLM')
class TriageAgent(Model4LLMs.Function):
    triage_llm_id:str
    agent_extract_id:str
    french_address_agent_id:str
    
    def __call__(self,
                 question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [triage_agent]: {msg}') if debug else lambda:None

        triage_llm = self.get_controller().storage().find(self.triage_llm_id)
        agent_extract = self.get_controller().storage().find(self.agent_extract_id)
        french_address_agent = self.get_controller().storage().find(self.french_address_agent_id)

        debugprint(f'Asking triage_llm with: [{dict(question=question)}]')    
        while True:
            # Get the response from triage_llm and extract the agent name
            response = triage_llm(question)
            agent_name = agent_extract(response)
            if agent_name  == 'french_address_agent':
                debugprint(f'Switching to agent: [{agent_name}]')
                return french_address_agent(question,debug=debug)


store.add_new_obj(FrenchAddressAgent(
                                    french_address_llm_id='ChatGPT4oMini:french_address_llm',
                                    first_json_extract_id='RegxExtractor:first_json_extract',
                                    french_address_search_function_id='ReverseGeocodeFunction:french_address_search_function'),
                                id='FrenchAddressAgent:french_address_agent')

store.add_new_obj(TriageAgent(
                triage_llm_id='ChatGPT4oMini:triage_llm',
                agent_extract_id='RegxExtractor:agent_extract',
                french_address_agent_id='FrenchAddressAgent:french_address_agent'),
            id='TriageAgent:triage_agent')

data = store.dumps()
store.clean()
store.loads(data)

# Example usage
answer = store.find('TriageAgent:triage_agent')(
                question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                debug=True)


print(f'\n\n######## Answer ########\n\n{answer}')
