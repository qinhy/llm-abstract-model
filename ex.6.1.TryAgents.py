
import json
import requests
from LLMAbstractModel.utils import RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')


## add French Address Search function
@descriptions('French reverse geocode coordinates to an address', lon='longitude', lat='latitude')
class FrenchReverseGeocodeFunction(Model4LLMs.Function):

    def __call__(self, lon: float, lat: float, debug=False):
        debugprint = lambda msg:print(f'--> [french_address_search_function]: {msg}') if debug else lambda:None
        # Construct the URL with the longitude and latitude parameters
        url = f"https://api-adresse.data.gouv.fr/reverse/?lon={lon}&lat={lat}"        
        # Perform the HTTP GET request
        debugprint(f'Searching address with: [{dict(url=url)}]')
        response = requests.get(url)        
        # Check if the request was successful
        if response.status_code == 200:
            # Return the JSON data from the response
            response = response.json()
        else:
            # Handle the error case
            response = {'error': f"Request failed with status code {response.status_code}"}
        
        debugprint(f'Got response: [{dict(response=response)}]')
        return response


# Workflow function for querying French address agent and handling responses
@descriptions('Workflow function of querying French address agent', question='The question to ask the LLM')
class FrenchAddressAgent(Model4LLMs.Function):
    french_address_llm_id:str
    first_json_extract:RegxExtractor = RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True)
    french_address_search_function_id:str
    french_address_system_prompt:str='''
You are familiar with France and speak English.
You will answer questions by providing information.
If you want to use an address searching by coordinates,
please only reply with the following json in markdown format:
```json
{"lon":2.37,"lat":48.357}
```
'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):        
        self.get_controller().update(french_address_llm_id = llm_obj.get_id())

    def __call__(self,question='My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [french_address_agent]: {msg}') if debug else lambda:None
        query = question
        store = self.get_controller().storage()
        french_address_llm:Model4LLMs.AbstractLLM = store.find(self.french_address_llm_id)
        french_address_llm.system_prompt = self.french_address_system_prompt
        french_address_search_function = store.find(self.french_address_search_function_id)
        
        while True:
            debugprint(f'Asking french_address_llm with: [{dict(question=query)}]')
            response = french_address_llm(query)
            debugprint(f'french_address_llm response with: [{dict(response=response)}]')
            coord_or_query = self.first_json_extract(response)
            if '"lon"' in coord_or_query and '"lat"' in coord_or_query:
                coord_or_query = json.loads(coord_or_query)
            
            # If the response contains coordinates, perform a reverse geocode search
            if isinstance(coord_or_query, dict) and "lon" in coord_or_query and "lat" in coord_or_query:
                debugprint(f'[french_address_search_function]: Searching address with coordinates: [{coord_or_query}]')
                query = french_address_search_function(**coord_or_query,debug=debug)
                query = f'\n## Question\n{question}\n## Information\n```\n{query}\n```\n'
            else:
                # Return the final response if no coordinates were found
                answer = coord_or_query
                debugprint(f'Final answer: [{dict(answer=answer)}]')
                return answer

# Workflow for handling triage queries and routing to the appropriate agent
@descriptions('Workflow function of triage queries and routing to the appropriate agent', question='The question to ask the LLM')
class TriageAgent(Model4LLMs.Function):
    triage_llm_id:str
    french_address_agent_id:str
    agent_extract:RegxExtractor = RegxExtractor(regx=r"```agent\s*(.*)\s*\n```")
    triage_system_prompt:str='''
You are a professional guide who can connect the asker to the correct agent.
Additionally, you are a skilled leader who can respond to questions using the agents' answer.
## Available Agents:
- french_address_agent: Familiar with France and speaks English.

## Just reply like following to connect the agent:
```agent
french_address_agent
```
'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):        
        self.get_controller().update(french_address_llm_id = llm_obj.get_id())
    
    def __call__(self,
                 question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [triage_agent]: {msg}') if debug else lambda:None

        store = self.get_controller().storage()
        triage_llm:Model4LLMs.AbstractLLM = store.find(self.triage_llm_id)
        triage_llm.system_prompt = self.triage_system_prompt
        french_address_agent = store.find(self.french_address_agent_id)

        debugprint(f'Asking triage_llm with: [{dict(question=question)}]')    
        while True:
            # Get the response from triage_llm and extract the agent name
            response = triage_llm(question)
            agent_name = self.agent_extract(response)
            debugprint(f'agent_extract : [{dict(response=response)}]')
            if 'french_address_agent' in agent_name:
                agent_name = 'french_address_agent'
                debugprint(f'Switching to agent: [{agent_name}]')
                answer = french_address_agent(question,debug=debug)
                return answer

# Initialize the French address search function that converts coordinates to addresses
french_address_search_function = store.add_new_obj(FrenchReverseGeocodeFunction())

# Create a LLM instance for handling French address queries
french_address_llm = store.add_new(Model4LLMs.ChatGPT41Nano)(vendor_id='auto')

# Create a LLM instance for the triage system that routes queries
triage_llm = store.add_new(Model4LLMs.ChatGPT41Nano)(vendor_id='auto')

# Initialize the French address agent with its required components
store.add_new_obj(FrenchAddressAgent(   
                                french_address_llm_id=french_address_llm.get_id(),
                                french_address_search_function_id=french_address_search_function.get_id()
                            ),id='FrenchAddressAgent:french_address_agent')

# Initialize the triage agent that coordinates between different specialized agents
store.add_new_obj(TriageAgent(
                triage_llm_id=triage_llm.get_id(),
                french_address_agent_id='FrenchAddressAgent:french_address_agent'
                ),id='TriageAgent:triage_agent')

# Serialize the store state for showing reusability
data = store.dumps()
# clean all
store.clean()
# Deserialize the json data to restore the store state
store.loads(data)

# Test the system with a sample GPS coordinate query
answer = store.find_all('*triage_agent*')[0](
                question='My GPS shows (47.665, 3.353), I need a detailed address.',
                debug=True)

# Display the final response
print(f'\n\n######## Answer ########\n\n{answer}')
