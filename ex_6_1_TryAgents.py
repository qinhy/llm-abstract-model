
import json
from typing import Optional
from pydantic import BaseModel, Field
import requests
from LLMAbstractModel.utils import RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs

def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')

# def add_tool(t:Type[MermaidWorkflowFunction]):        
#     mcp.add_tool(t, 
#     description=t.model_fields['description'].default)

class FrenchReverseGeocodeFunction(Model4LLMs.MermaidWorkflowFunction):
    description:str = "French reverse geocode coordinates to an address"

    class Arguments(BaseModel):
        lon: float = Field(..., description="Longitude")
        lat: float = Field(..., description="Latitude")

    class Returness(BaseModel):
        address: Optional[dict] = Field(None, description="Geocoded address result from French government API")
        success: bool = Field(..., description="True if API call was successful")
        error: Optional[str] = Field(None, description="Error message, if any")

    args: Optional[Arguments] = None
    rets: Optional[Returness] = None
    debug: bool = False

    def __call__(self):
        def debugprint(msg):
            if self.debug:
                print(f'--> [FrenchReverseGeocodeFunction]: {msg}')

        lon, lat = self.args.lon, self.args.lat
        url = f"https://api-adresse.data.gouv.fr/reverse/?lon={lon}&lat={lat}"
        debugprint(f"Querying URL: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            debugprint(f"Received data: {data}")
            self.rets = self.Returness(address=data, success=True)
        except Exception as e:
            error_msg = f"Request failed: {e}"
            debugprint(error_msg)
            self.rets = self.Returness(address=None, success=False, error=error_msg)

        return self.rets


# Workflow function for querying French address agent and handling responses
class FrenchAddressAgent(Model4LLMs.MermaidWorkflowFunction):
    description:str = Field('Workflow function of querying French address agent')
    # question:str = Field(...,description='The question to ask the LLM')
    
    french_address_llm_id:str
    first_json_extract:RegxExtractor = RegxExtractor(para=dict(regx=r"```json\s*(.*)\s*\n```", is_json=True))
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
        self.controller.update(french_address_llm_id = llm_obj.get_id())

    def __call__(self,question='My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [french_address_agent]: {msg}') if debug else lambda:None
        query = question
        store = self.controller.storage()
        french_address_llm:Model4LLMs.AbstractLLM = store.find(self.french_address_llm_id)
        french_address_llm.system_prompt = self.french_address_system_prompt
        french_address_search_function:FrenchReverseGeocodeFunction = store.find(self.french_address_search_function_id)
        
        while True:
            debugprint(f'Asking french_address_llm with: [{dict(question=query)}]')
            response = french_address_llm(query)
            debugprint(f'french_address_llm response with: [{dict(response=response)}]')
            coord_or_query = self.first_json_extract(response)
            if '"lon"' in coord_or_query and '"lat"' in coord_or_query:
                coord_or_query = json.loads(coord_or_query)
            
            # If the response contains coordinates, perform a reverse geocode search
            if isinstance(coord_or_query, dict):
                french_address_search_function.args = FrenchReverseGeocodeFunction.Arguments.model_validate(coord_or_query)
                french_address_search_function.debug=debug
                debugprint(f'[french_address_search_function]: Searching address with coordinates: [{coord_or_query}]')
                query = french_address_search_function()
                query = f'\n## Question\n{question}\n## Information\n```\n{query}\n```\n'
            else:
                # Return the final response if no coordinates were found
                answer = coord_or_query
                debugprint(f'Final answer: [{dict(answer=answer)}]')
                return answer

# Workflow for handling triage queries and routing to the appropriate agent
class TriageAgent(Model4LLMs.MermaidWorkflowFunction):
    description:str = Field('Workflow function of triage queries and routing to the appropriate agent')

    triage_llm_id:str
    french_address_agent_id:str
    agent_extract:RegxExtractor = RegxExtractor(para=dict(regx=r"```agent\s*(.*)\s*\n```"))
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
        self.controller.update(french_address_llm_id = llm_obj.get_id())
    
    def __call__(self,
                 question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [triage_agent]: {msg}') if debug else lambda:None

        store = self.controller.storage()
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
french_address_agent = store.add_new_obj(FrenchAddressAgent(   
                                french_address_llm_id=french_address_llm.get_id(),
                                french_address_search_function_id=french_address_search_function.get_id()
                            ),id='french_address_agent')

# Initialize the triage agent that coordinates between different specialized agents
store.add_new_obj(TriageAgent(
                triage_llm_id=triage_llm.get_id(),
                french_address_agent_id='french_address_agent'
                ),id='triage_agent')

# Serialize the store state for showing reusability
data = store.dumps()
# clean all
store.clean()
# Deserialize the json data to restore the store state
store.loads(data)

# Test the system with a sample GPS coordinate query
answer = store.find_all('*triage_agent')[0](
                question='My GPS shows (47.665, 3.353), I need a detailed address.',
                debug=True)

# Display the final response
print(f'\n\n######## Answer ########\n\n{answer}')
