import { StringTemplate, RegxExtractor } from './utils';
import { LLMsStore, Model4LLMs } from './LLMsModel';

// Initialize the store
const store = new LLMsStore();

// System prompt
const systemPrompt = `You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
## Your Reply Format Example
\`\`\`translation
...
\`\`\``;

// Add vendor and model
const vendor = store.addNewOpenAIVendor(process.env.OPENAI_API_KEY);
const llm = store.addNewChatGPT4oMini('auto');
llm.system_prompt = systemPrompt;

console.log('############# Make a message template');

// Add a string template function
const translateTemplate:StringTemplate = store.add_new_obj(
  new StringTemplate(`\`\`\`text\n{}\n\`\`\``)
);
console.log(translateTemplate.call(['こんにちは！はじめてのチェーン作りです！']));
// -> ...

console.log(
  await llm.call(
    translateTemplate.call(['こんにちは！はじめてのチェーン作りです！'])
  )
);
// -> ```translation
// -> Hello! This is my first time making a chain!
// -> ```

console.log(
  '############# Make a "translation" extractor, strings between "```translation" and "```"'
);

// Add a regex extractor
const getResult = store.add_new_obj(
  new RegxExtractor(/```translation\s*(.*?)\s*```/)
);

// Chain process
console.log(
  getResult.call(
    await llm.call(
      translateTemplate.call(['こんにちは！はじめてのチェーン作りです！'])
    )
  )
);
// -> Hello! This is my first time making a chain!

console.log('############# Make the chain more graceful and simple');

// Compose function to chain multiple functions
function compose(...funcs: Function[]): Function {
  return (...args: any[]) =>
    funcs.reduce((acc, func) => (Array.isArray(acc) ? func(...acc) : func(acc)), args);
}

function sayA(x: string): string {
  return `${x}a`;
}
function sayB(x: string): string {
  return `${x}b`;
}
function sayC(x: string): string {
  return `${x}c`;
}

const sayABC = compose(sayA, sayB, sayC);
console.log('Test chain up:', sayABC(''));

// Translation chain
const translatorChain = [translateTemplate, llm, getResult];
const translator = compose(...translatorChain);

console.log(translator('こんにちは！はじめてのチェーン作りです！'));
// -> Hello! This is my first time making a chain!

console.log(translator('常識とは、18歳までに身に付けた偏見のコレクションである。'));
// -> Common sense is a collection of prejudices acquired by the age of 18.

console.log(translator('为政以德，譬如北辰，居其所而众星共之。'));
// -> Governing with virtue is like the North Star, which remains in its place while all the other stars revolve around it.

console.log('############# Save/Load chain JSON');
const chainDump = LLMsStore.chainDumps(translatorChain);
console.log(chainDump);

const loadedChain = store.chainLoads(chainDump);
console.log(loadedChain);
const loadedTranslator = compose(...loadedChain);

console.log(loadedTranslator('こんにちは！はじめてのチェーン作りです！'));
// -> Hello! It's my first time making a chain!

console.log('############# Additional template usage');

// Add new LLM
const newLLM = store.addNewChatGPT4oMini('auto');

// Use raw JSON template
const jsonTemplate = store.addNewFunction(
  new StringTemplate(`[
    {"role":"system","content":"You are an expert in translation text.I will provide text. Please translate it.\\nYou should reply translations only, without any additional information.\\n\\n## Your Reply Format Example\\n\\\`\\\`\\\`translation\\n...\\n\\\`\\\`\\\`"},
    {"role":"user","content":"\\nPlease translate the text into {}.\\n\\\`\\\`\\\`text\\n{}\\n\\\`\\\`\\\`"}
]`)
);

console.log(
  jsonTemplate.call(['to English', 'こんにちは！はじめてのチェーン作りです！'])
);
// -> [
// ->     {"role":"system","content":"You are an expert in translation text.I will provide text. Please translate it.\nYou should reply translations only, without any additional information.\n\n## -> Your Reply Format Example\n```translation\n...\n```"},
// ->     {"role":"user","content":"\nPlease translate the text into English.\n```text\nこんにちは！はじめてのチェーン作りです！\n```"}
// -> ]

const message = JSON.parse(
  jsonTemplate.call(['to English', 'こんにちは！はじめてのチェーン作りです！'])
);
console.log(message);

console.log(await newLLM.call(message));
// -> ```translation
// -> Hello! This is my first time making a chain!
// -> ```

const advancedTranslatorChain = [jsonTemplate, JSON.parse, newLLM, getResult];
const advancedTranslator = compose(...advancedTranslatorChain);

console.log(
  advancedTranslator('to Chinese', 'こんにちは！はじめてのチェーン作りです！')
);
// -> 你好！这是第一次制作链条！
console.log(
  advancedTranslator('to Japanese', 'Hello! This is my first time making a chain!')
);
// -> こんにちは！これは私の初めてのチェーン作りです！
