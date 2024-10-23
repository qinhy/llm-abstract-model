
import requests
from LLMAbstractModel.LLMsModel import LLMsStore, Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
store = LLMsStore()


##########################
## ex.4.CustomStores.py ##
##########################
class FibonacciObj(Model4LLMs.AbstractObj):
    n:int=1

### registeration magic
store.add_new_obj(FibonacciObj()).get_controller().delete()


############################
## ex.5.CustomWorkflow.py ##
############################
@descriptions('Add two numbers', x='first number', y='second number')
class AddFunction(Model4LLMs.Function):
    def __call__(self, x: int, y: int):
        return x + y

@descriptions('Multiply a number by 2', x='number to multiply')
class MultiplyFunction(Model4LLMs.Function):
    def __call__(self, x: int):
        return x * 2

@descriptions('Subtract second number from first', x='first number', y='second number')
class SubtractFunction(Model4LLMs.Function):
    def __call__(self, x: int, y: int):
        return x - y

@descriptions('Return a constant value of 5')
class ConstantFiveFunction(Model4LLMs.Function):
    def __call__(self):
        return 5

@descriptions('Return a constant value of 3')
class ConstantThreeFunction(Model4LLMs.Function):
    def __call__(self):
        return 3
    
### registeration magic
store.add_new_function(AddFunction()).get_controller().delete()
store.add_new_function(MultiplyFunction()).get_controller().delete()
store.add_new_function(ConstantThreeFunction()).get_controller().delete()
store.add_new_function(ConstantFiveFunction()).get_controller().delete()


#######################
## ex.6.TryAgents.py ##
#######################
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
            
### registeration magic
store.add_new_obj(ReverseGeocodeFunction()).get_controller().delete()
store.add_new_obj(FrenchAddressAgent(french_address_llm_id='',
                                    first_json_extract_id='',
                                    french_address_search_function_id='')).get_controller().delete()
store.add_new_obj(TriageAgent(triage_llm_id='',agent_extract_id='',
            french_address_agent_id='')).get_controller().delete()