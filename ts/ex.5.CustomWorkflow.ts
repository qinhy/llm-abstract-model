import {
    LLMsStore,
    Model4LLMs,
} from "./LLMsModel";
import { RegxExtractor, StringTemplate } from "./utils";


function myprint(expression: string): void {
    console.log("##", expression, ":\n", eval(expression), "\n");
}

const store = new LLMsStore();

class AddFunction extends Model4LLMs.Function {
    call(x: number, y: number): number {
        return x + y;
    }
}

class MultiplyFunction extends Model4LLMs.Function {
    call(x: number): number {
        return x * 2;
    }
}

class SubtractFunction extends Model4LLMs.Function {
    call(x: number, y: number): number {
        return x - y;
    }
}

class ConstantFiveFunction extends Model4LLMs.Function {
    call(): number {
        return 5;
    }
}

class ConstantThreeFunction extends Model4LLMs.Function {
    call(): number {
        return 3;
    }
}

// Add functions to the store
const taskAdd = store.addNewFunction(new AddFunction());
const taskMultiply = store.addNewFunction(new MultiplyFunction());
const subtractFunction = store.addNewFunction(new SubtractFunction());
const taskConstantThree = store.addNewFunction(new ConstantThreeFunction());
const taskConstantFive = store.addNewFunction(new ConstantFiveFunction());

// Create a new WorkFlow instance
let workflow: Model4LLMs.WorkFlow = store.add_new_obj(
    new Model4LLMs.WorkFlow({
        tasks: {
            [taskAdd.get_id()]: [
                taskMultiply.get_id(),
                subtractFunction.get_id(),
            ],
            [taskMultiply.get_id()]: [taskConstantFive.get_id()],
            [subtractFunction.get_id()]: [taskConstantFive.get_id(), taskConstantThree.get_id()],
            [taskConstantThree.get_id()]: [],
            [taskConstantFive.get_id()]: [],
        },
    })
);

// Run the workflow
console.log('## await workflow.acall()\n',
    await workflow.acall());
// -> 13

// Retrieve and print the result of each task
myprint("JSON.stringify(workflow.model_dump_json_dict(), null, 2)");
// -> JSON representation of the workflow

// Clean up the store
store.clean();


// ###############
const system_prompt = `
You are an expert in English translation.
I will provide you with the text. Please translate it.
You should reply with translations only, without any additional information.
## Your Reply Format Example
\`\`\`translation
...
\`\`\`
`.trim();

const vendor = store.addNewOpenAIVendor(process.env.OPENAI_API_KEY);
const llm = store.addNewChatGPT4oMini('auto', system_prompt);

// Create template and extractor tasks
const input_template = store.add_new_obj(
    new StringTemplate(`
\`\`\`text
{}
\`\`\`
`.trim())
);

const extract_result = store.add_new_obj(
    new RegxExtractor('```translation\\n([\\s\\S]*?)\\n```')
);

// Define the workflow with tasks
workflow = store.addNewWorkFlow(
    {
        [input_template.get_id()]: ['input'],
        [llm.get_id()]: [input_template.get_id()],
        [extract_result.get_id()]: [llm.get_id()]
    }
);

// Input example
console.log('## await workflow.acall({ input: [["こんにちは！はじめてのチェーン作りです！"]] })\n',
    await workflow.acall({ input: [["こんにちは！はじめてのチェーン作りです！"]] }));
console.log('## workflow.results\n',workflow.results);

// -> Hello! This is my first time making a chain!

workflow.get_controller().delete();

// Also support sequential list input
const sequential_workflow = store.addNewWorkFlow(
    [input_template.get_id(),
    llm.get_id(),
    extract_result.get_id()]
);

// Reuse the workflow by setting new input
console.log('## await sequential_workflow.acall(["常識とは、18歳までに身に付けた偏見のコレクションである。"])\n',
    await sequential_workflow.acall(["常識とは、18歳までに身に付けた偏見のコレクションである。"]));
// -> Common sense is a collection of prejudices acquired by the age of 18.

// Save and load workflow
const data = store.dumps();
store.clean();
store.loads(data);
const reloaded_workflow:Model4LLMs.WorkFlow = store.find_all('WorkFlow:*')[0];

console.log('## await reloaded_workflow.acall(["为政以德，譬如北辰，居其所而众星共之。"])\n',
    await reloaded_workflow.acall(["为政以德，譬如北辰，居其所而众星共之。"]));
// -> Governing with virtue is like the North Star, which occupies its position while all the other stars revolve around it.
