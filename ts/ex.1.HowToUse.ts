import { LLMsStore } from './LLMsModel';
import { Model4LLMs } from './LLMsModel';

const store = new LLMsStore();

// Retrieve API key from environment variables or use 'null' as default
const openaiApiKey = process.env.OPENAI_API_KEY || 'null';
const vendor = store.addNewOpenAIVendor(openaiApiKey);

// Add a new chat model (ChatGPT4Omini)
const chatgpt4omini = store.addNewChatGPT4oMini(vendor.get_id());

// Uncomment to add other models (e.g., ChatGPTO1Mini)
// const o1mini = store.addNewChatGPTO1Mini(vendor.getId());

// Add a text embedding model
const textEmbedding = store.add_new_obj(new Model4LLMs.TextEmbedding3Small());

// Uncomment if you have XAI API setup
// const xaiApiKey = process.env.XAI_API_KEY || 'null';
// const xaiVendor = store.addNewXaiVendor(xaiApiKey);
// const grok = store.addNewGrok(xaiVendor.getId());

// Uncomment if you have Ollama setup
// const ollamaVendor = store.addNewOllamaVendor();
// const gemma2 = store.addNewGemma2(ollamaVendor.getId());
// const phi3 = store.addNewPhi3(ollamaVendor.getId());
// const llama32 = store.addNewLlama(ollamaVendor.getId());

// Use the chat model
console.log(chatgpt4omini.call('hi! What is your name?'));
// -> Hello! I’m called Assistant. How can I help you today?

// Send multiple messages to the chat model
console.log(
    chatgpt4omini.call([
        { role: 'system', content: 'You are a highly skilled professional English translator.' },
        { role: 'user', content: '"こんにちは！"' }
    ])
);
// -> "Hello!"

// Use the text embedding model
console.log(textEmbedding.call('hi! What is your name?').slice(0, 10), '...');
// -> [0.0118862, -0.0006172658, -0.008183353, 0.02926386, -0.03759078, -0.031130238, -0.02367668 ...]
