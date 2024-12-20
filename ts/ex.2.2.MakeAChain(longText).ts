import { StringTemplate, RegxExtractor } from './utils';
import { LLMsStore } from './LLMsModel';
import { TextFile } from './file';

// Initialize the LLMsStore
const store = new LLMsStore();

// Define the system prompt
const systemPrompt = `I will provide pieces of the text along with prior summarizations.
Your task is to read each new text snippet and add new summarizations accordingly.
You should reply summarizations only, without any additional information.

## Your Reply Format Example
\`\`\`summarization
- This text shows ...
\`\`\``;

// Add vendors and LLM models
const openaiVendor = store.addNewOpenAIVendor(process.env.OPENAI_API_KEY);
const chatGPT4oMini = store.addNewChatGPT4oMini(openaiVendor.get_id());
chatGPT4oMini.system_prompt = systemPrompt;

// const ollamaVendor = store.addNewOllamaVendor();
// const gemma2 = store.addNewGemma2(ollamaVendor.get_id());
// gemma2.system_prompt = systemPrompt;

// const phi3 = store.addNewPhi3(ollamaVendor.get_id());
// phi3.system_prompt = systemPrompt;

// const llama32 = store.addNewLlama(ollamaVendor.get_id());
// llama32.system_prompt = systemPrompt;

// Add a message template
const msgTemplate = store.add_new_obj(
    new StringTemplate(`Please reply summarizations in {}, and should not over {} words.
## Text Snippet
\`\`\`text
{}
\`\`\`
## Previous Summarizations
\`\`\`summarization
{}
\`\`\``)
);

// Add a regex extractor
const resExt = store.add_new_obj(
    new RegxExtractor('```summarization\\n([\\s\\S]*?)\\n```')
);

// Test summarization function
async function* testSummary(
    llm = chatGPT4oMini,
    filePath: string,
    limitWords: number = 1000,
    chunkLines: number = 100,
    overlapLines: number = 30
) {
    let preSummarization: string | null = null;
    const textFile = new TextFile(filePath, chunkLines, overlapLines);

    for (const chunk of textFile) {
        const msg = msgTemplate.call(['Japanese', limitWords, chunk.join('\n'), preSummarization]);
        const output = await llm.acall(msg);
        const result = resExt.call(output);

        preSummarization = result;
        yield result;
    }
}

// Custom chain example
function compose(...funcs: Function[]): Function {
    return (...args: any[]) =>
      funcs.reduce((acc, func) => ( func(acc) ), args);
  }
  
const chainList = [msgTemplate, chatGPT4oMini, resExt];
const chain = compose(
    msgTemplate.call.bind(msgTemplate),
    chatGPT4oMini.acall.bind(chatGPT4oMini),
    resExt.acall.bind(resExt)
);

// Example usage
let preSummarization = '';
const textFile = new TextFile('../The Adventures of Sherlock Holmes.txt', 100, 30);

for (const chunk of textFile) {
    // const summarization = await chain(['Japanese', 100, chunk.join('\n'), preSummarization]);
    // preSummarization = summarization;
    console.log(await resExt.acall(chatGPT4oMini.acall(msgTemplate.call(['Japanese', 100, chunk.join('\n'), preSummarization]))));
    break; // Remove this break to process the entire file
}

// Batch processing for multiple files
// async function processFiles(files: string[]) {
//     for (const file of files) {
//         const summarizer = testSummary(chatGPT4oMini, file);
//         let combinedSummarization = '';
        
//         for await (const summarization of summarizer) {
//             combinedSummarization += summarization;

//             // Save to individual file
//             const outputFile = file.replace('file', 'output');
//             const fs = require('fs');
//             fs.appendFileSync(outputFile, summarization);
//         }

//         // Save combined summarizations
//         const allInOneFile = 'allinone.txt';
//         const fs = require('fs');
//         fs.appendFileSync(allInOneFile, `## ${file}\n${combinedSummarization}`);
//     }
// }
