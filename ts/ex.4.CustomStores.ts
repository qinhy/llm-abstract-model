import { LLMsStore, Model4LLMs } from './LLMsModel';

function myprint(expression: string): void {
    console.log("##", expression, ":\n", eval(expression), "\n");
}

const store = new LLMsStore();

const systemPrompt = "You are smart assistant";

const vendor = store.addNewOpenAIVendor(process.env.OPENAI_API_KEY);
const chatGPT4oMini = store.addNewChatGPT4oMini(vendor.get_id());
chatGPT4oMini.system_prompt = systemPrompt;

// Uncomment to add other vendors
// const vendor = store.add_new_ollama_vendor();
// const gemma2 = store.add_new_gemma2({ vendor_id: vendor.get_id(), system_prompt: systemPrompt });
// const phi3 = store.add_new_phi3({ vendor_id: vendor.get_id(), system_prompt: systemPrompt });
// const llama32 = store.add_new_llama({ vendor_id: vendor.get_id(), system_prompt: systemPrompt });

// Add custom function
class FibonacciFunction extends Model4LLMs.AbstractObj {
    __call__(n: number): number {
        function fibonacci(n: number): number {
            if (n <= 1) return n;
            return fibonacci(n - 1) + fibonacci(n - 2);
        }
        return fibonacci(n);
    }
}

const get_fibo = store.add_new_obj(new FibonacciFunction());
myprint('get_fibo.model_dump_json_dict()');
// -> {'rank': [0], 'create_time': ..., 'update_time': ..., 'status': '', 'metadata': {}, 'name': 'FibonacciFunction', ...}

myprint('store.find_all("FibonacciFunction:*")')
// -> 13

// Add custom Obj
class FibonacciObj extends Model4LLMs.AbstractObj {
    n: number;

    constructor(n: number) {
        super();
        this.n = n;
    }
}

const fb = store.add_new_obj(new FibonacciObj(7));
myprint('store.find_all("FibonacciObj:*")[0].model_dump_json_dict()');
// -> {'rank': [0], 'create_time': ..., 'update_time': ..., 'status': '', 'metadata': {}, 'n': 7}

myprint("store.dumps()");
// -> JSON representation of the store contents

// Test web requests
store.clean();
const req = store.addNewRequest("http://localhost:8000/tasks/status/some-task-id","GET");


myprint("req.model_dump_json_dict()");
// -> {'rank': [0], 'create_time': ..., 'update_time': ..., 'status': '', 'metadata': {}, 'name': 'RequestsFunction', ...}

console.log(await req.acall());
// -> {'task_id': 'some-task-id', 'status': 'STARTED', 'result': {'message': 'Task is started'}}
