import {
    LLMsStore,
    Model4LLMs,
} from "./LLMsModel";
import { RegxExtractor, StringTemplate } from "./utils";
import axios from 'axios';

const store = new LLMsStore();
const vendor = store.addNewOpenAIVendor(process.env.OPENAI_API_KEY);
const debug = true;
class FrenchReverseGeocodeFunction extends Model4LLMs.Function {
    async acall(lon: number, lat: number): Promise<any> {
        // Construct the URL with the longitude and latitude parameters
        const url = `https://api-adresse.data.gouv.fr/reverse/?lon=${lon}&lat=${lat}`;

        try {
            // Perform the HTTP GET request
            const response = await axios.get(url);
            // Check if the request was successful
            return response.data;
        } catch (error: any) {
            // Handle the error case
            return {
                error: `Request failed with status code ${error.response?.status || "unknown"}`
            };
        }
    }
}
// Add functions for reverse geocoding and address extraction
const frenchAddressSearchFunction = store.add_new_obj(new FrenchReverseGeocodeFunction());

class FrenchAddressAgent extends Model4LLMs.Function {
    french_address_llm_id: string = '';
    first_json_extract: RegxExtractor = new RegxExtractor('```json\\n([\\s\\S]*?)\\n```',true);
    french_address_search_function_id: string = '';
    french_address_system_prompt: string = `
You are familiar with France and speak English.
You will answer questions by providing information.
If you want to use an address searching by coordinates, please only reply with the following text:
\`\`\`json
{"lon":2.37,"lat":48.357}
\`\`\`
`;

constructor(
    french_address_llm_id: string,
    french_address_search_function_id: string
) {
    super();
    this.french_address_llm_id = french_address_llm_id;
    this.french_address_search_function_id = french_address_search_function_id;
}
    get_controller() {
        // Mocked controller implementation
        return {
            storage: () => store,
        };
    }

    change_llm(llmObj: Model4LLMs.AbstractLLM): void {
        this.french_address_llm_id = llmObj.get_id();
        this.get_controller().storage().add_new_obj(this);
    }

    async acall(
        question: string = 'I am in France and My GPS shows (47.665176, 3.353434), where am I?',
        debug: boolean = false
    ): Promise<any> {
        const debugprint = (msg: string) => {
            if (debug) console.log(`--> [french_address_agent]: ${msg}`);
        };

        let query = question;
        const controller = this.get_controller();
        const frenchAddressLLM : Model4LLMs.AbstractLLM = controller.storage().find(this.french_address_llm_id);
        this.first_json_extract = new RegxExtractor(this.first_json_extract.regx,this.first_json_extract.isJson);
        frenchAddressLLM.system_prompt = this.french_address_system_prompt;

        const frenchAddressSearchFunction:FrenchReverseGeocodeFunction = controller.storage().find(this.french_address_search_function_id);

        while (true) {
            debugprint(`Asking french_address_llm with: [${JSON.stringify({ question: query })}]`);
            const response = await frenchAddressLLM.acall(query);
            const coordOrQuery = this.first_json_extract.call(response,false);
            if (coordOrQuery && coordOrQuery.lon && coordOrQuery.lat) {
                debugprint(`[french_address_search_function]: Searching address with coordinates: [${JSON.stringify(coordOrQuery)}]`);
                const searchResponse = await frenchAddressSearchFunction.acall(coordOrQuery.lon, coordOrQuery.lat);
                query = `
## Question
${question}
## Information
\`\`\`
${JSON.stringify(searchResponse)}
\`\`\``;
            } else {
                const answer = coordOrQuery;
                debugprint(`Final answer: [${JSON.stringify({ answer })}]`);
                return answer;
            }
        }
    }
}


class TriageAgent extends Model4LLMs.Function {
    triage_llm_id: string;
    french_address_agent_id: string;
    agent_extract: RegxExtractor = new RegxExtractor('```agent\\n([\\s\\S]*?)\\n```');
    triage_system_prompt: string = `
You are a professional guide who can connect the asker to the correct agent.
Additionally, you are a skilled leader who can respond to questions using the agents' answer.
## Available Agents:
- french_address_agent: Familiar with France and speaks English.

## How to connect the agent:
\`\`\`agent
french_address_agent
\`\`\`
`;

    constructor(
        triage_llm_id: string,
        french_address_agent_id: string
    ) {
        super();
        this.triage_llm_id = triage_llm_id;
        this.french_address_agent_id = french_address_agent_id;
    }

    get_controller() {
        return {
            storage: () => store,
        };
    }

    async acall(
        question: string = 'I am in France and My GPS shows (47.665176, 3.353434), where am I?',
        debug: boolean = false
    ): Promise<any> {
        const debugprint = (msg: string) => {
            if (debug) console.log(`--> [triage_agent]: ${msg}`);
        };

        const controller = this.get_controller();
        const triageLLM : Model4LLMs.AbstractLLM= controller.storage().find(this.triage_llm_id);
        const frenchAddressAgent : FrenchAddressAgent = controller.storage().find(this.french_address_agent_id);
        this.agent_extract = new RegxExtractor(this.agent_extract.regx,this.agent_extract.isJson)
        triageLLM.system_prompt = this.triage_system_prompt;

        debugprint(`Asking triage_llm with: [${JSON.stringify({ question })}]`);

        while (true) {
            const response = await triageLLM.acall(question); // Replace with actual LLM call
            const agentName = this.agent_extract.call(response);

            debugprint(`agent_extract : [${agentName}]`);

            if (agentName === 'french_address_agent') {
                debugprint(`Switching to agent: [${agentName}]`);
                const answer = await frenchAddressAgent.acall(question, debug);
                return await triageLLM.acall(`## User Question\n${question}\n## ${agentName} Answer\n${answer}\n`);
            }
        }
    }
}

const frenchAddressLLM = store.addNewChatGPT4oMini('auto');
const triageLLM = store.addNewChatGPT4oMini('auto');

store.add_new_obj(
    new FrenchAddressAgent(
        frenchAddressLLM.get_id(),
        frenchAddressSearchFunction.get_id()),
    'FrenchAddressAgent:french_address_agent'
);
store.add_new_obj(
    new TriageAgent(
        triageLLM.get_id(),
        'FrenchAddressAgent:french_address_agent'),
    'TriageAgent:triage_agent'
);

// Serialize and reload the store
const data = store.dumps();
store.clean();
store.loads(data);

const agent:TriageAgent = store.find('TriageAgent:triage_agent');
console.log(`\n\n######## Answer ########\n\n${
    await agent.acall(
        'I am in France and My GPS shows (47.665176, 3.353434), where am I?',
        true
    )
}`);


