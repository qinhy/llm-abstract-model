<template>
    <div class="container mx-auto p-6 bg-gray-50">
      <h1 class="text-2xl font-bold text-center text-gray-800 mb-8">LLMs Interaction GUI</h1>
      
      <!-- Vendor Details -->
      <div class="vendor-section bg-white shadow-lg p-6 rounded-lg mb-8">
        <h2 class="text-xl font-semibold text-gray-700 mb-4">Vendor Configuration</h2>
        <div class="mb-4">
          <label for="apiKey" class="block text-gray-600 text-sm font-medium mb-2">OpenAI API Key:</label>
          <InputText id="apiKey" v-model="openaiApiKey" placeholder="Enter text for OpenAI API Key..." class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
        <p class="text-sm text-gray-600">
          <strong>Vendor ID:</strong> {{ vendorId }}
        </p>
      </div>
      
      <!-- ChatGPT4oMini Interaction -->
      <div class="interaction-section bg-white shadow-lg p-6 rounded-lg mb-8">
        <h2 class="text-xl font-semibold text-gray-700 mb-4">ChatGPT4oMini</h2>
        <div class="mb-4">
          <InputText v-model="chatInput" placeholder="Type a message..." class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
        <Button label="Send" @click="sendChat" class="p-button bg-blue-500 hover:bg-blue-600 text-white font-semibold px-4 py-2 rounded-lg shadow" />
        <div class="response mt-4">
          <p class="text-sm text-gray-600">
            <strong>Response:</strong> {{ chatResponse }}
          </p>
        </div>
      </div>
      
      <!-- Text Embedding Interaction -->
      <div class="embedding-section bg-white shadow-lg p-6 rounded-lg">
        <h2 class="text-xl font-semibold text-gray-700 mb-4">Text Embedding</h2>
        <div class="mb-4">
          <InputText v-model="embeddingInput" placeholder="Enter text for embedding..." class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
        <Button label="Get Embedding" @click="getEmbedding" class="p-button bg-blue-500 hover:bg-blue-600 text-white font-semibold px-4 py-2 rounded-lg shadow" />
        <div class="response mt-4">
          <p class="text-sm text-gray-600">
            <strong>Embedding:</strong> {{ embeddingResponse }}
          </p>
        </div>
      </div>
    </div>
  </template>
  
  
  <script>
  import { ref, onMounted, computed } from "vue";
  import { Model4LLMs, LLMsStore } from '../libs/LLMsModel';
  
  export default {
    setup() {
      // State variables
      const store = new LLMsStore();
      const openaiApiKey = computed({
        get:()=>{return store.find_all('OpenAIVendor:*')[0].api_key},
        set:(val)=>{store.find_all('OpenAIVendor:*')[0].get_controller().update({api_key:val})},
      })
      const vendor = store.addNewOpenAIVendor('null');
      const vendorId = vendor.get_id();
      const chatgpt4omini = store.addNewChatGPT4oMini(vendorId);
      const textEmbedding = store.add_new_obj(new Model4LLMs.TextEmbedding3Small());
  
      const chatInput = ref("");
      const chatResponse = ref("");
      const embeddingInput = ref("");
      const embeddingResponse = ref("");
  
      // Functions
      const sendChat = async () => {
        if (!chatInput.value) return;
        chatResponse.value = '';
        // const response = await chatgpt4omini.acall(chatInput.value);
        // chatResponse.value = response;
        
        for await (const chunk of chatgpt4omini.scall(chatInput.value)) {
            chatResponse.value += chunk;
        }
      };
  
      const getEmbedding = async () => {
        if (!embeddingInput.value) return;
        const response = await textEmbedding.acall(embeddingInput.value);
        embeddingResponse.value = response.slice(0, 10).join(", ") + " ...";
      };
  
      return {
        openaiApiKey,
        vendorId,
        chatInput,
        chatResponse,
        embeddingInput,
        embeddingResponse,
        sendChat,
        getEmbedding,
      };
    },
  };
  </script>