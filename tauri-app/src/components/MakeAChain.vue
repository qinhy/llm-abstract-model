<template>
  <div style="width: 100vw; height: 60vh">
    <Button label="Run" icon="pi pi-play" @click="runGraph" />
    <BaklavaEditor :view-model="baklava" />
    <div v-if="config_obj_id" class="vendor-section bg-white shadow-lg p-6 rounded-lg mb-8">

      <div v-if="config_obj_id.split(':')[0]=='StringTemplate'">{{ config_obj_id }}</div>

      <div v-if="config_obj_id.split(':')[0]=='RegxExtractor'">{{ config_obj_id }}</div>

      <div v-if="config_obj_id.split(':')[0]=='ChatGPT4oMini'">        
        <!-- <h2 class="text-xl font-semibold text-gray-700 mb-4">Vendor Configuration</h2> -->
        <div class="mb-4">
          <label for="apiKey" class="block text-gray-600 text-sm font-medium mb-2">OpenAI API Key:</label>
          <InputText id="apiKey" v-model="openaiApiKey" placeholder="Enter text for OpenAI API Key..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
        <p class="text-sm text-gray-600">
          <strong>Vendor ID:</strong> {{ vendorId }}
        </p>
      </div>
    </div>
    <div style="margin-top: 1px;">
    </div>
  </div>
</template>

<script lang="ts">
import { BaklavaEditor, EditorComponent, useBaklava, DependencyEngine, applyResult, SelectInterface, ButtonInterface } from "baklavajs";
import "@baklavajs/themes/dist/syrup-dark.css";
import { CodeBlockExtract, PromptInput, PromptOutput } from "./Nodes/CustomNodes";
import { defineNode, NodeInterface, TextInputInterface, TextareaInputInterface, TextInterface } from "baklavajs";
import { Model4LLMs, LLMsStore } from '../libs/LLMsModel';
import { computed, onMounted, ref, watch } from "vue";
import { RegxExtractor, StringTemplate } from "../libs/utils";

export default {
  components: {
    BaklavaEditor
  },
  setup() {
    // set up interactive GUI
    const baklava = useBaklava();
    window.baklava = baklava;
    baklava.editor.registerNodeType(PromptInput);
    baklava.editor.registerNodeType(PromptOutput);
    const engine = new DependencyEngine(baklava.editor);
    const token = Symbol();
    engine.events.afterRun.subscribe(token, (result) => {
      engine.pause();
      applyResult(result, baklava.editor);
      engine.resume();
    });


    const store = new LLMsStore();
    const vendorId = computed({
      get: () => { return store.find_all('OpenAIVendor:*')[0].get_id() },
    })
    const save = () => {
      store.set("Graph", baklava.editor.save());
      localStorage.setItem('MakeAChain', store.dumps());
    }

    const load = () => {
      store.clean();
      store.add_new_obj(new StringTemplate('')).get_controller().delete();
      store.add_new_obj(new RegxExtractor('')).get_controller().delete();
      store.loads(localStorage.getItem('MakeAChain'));
      loadUI();
      baklava.editor.load(store.get("Graph"))
    }

    const loadUI = () => {
      store.find_all('*:*').filter(obj => obj.acall || obj.call)
        .map(obj => obj.get_id().split(':')[0])
        .forEach(name => {
          const ids = store.keys(`${name}:*`).map(n => n.split(':')[1]);
          // console.log(ids);

          baklava.editor.registerNodeType(defineNode({
            type: name,
            inputs: {
              ID: () => new SelectInterface("ID", ids[0], ids).setPort(false),
              source: () => new TextInputInterface("Text", ""),
            },
            outputs: { output: () => new NodeInterface<string>("Output", "null"), },
            async calculate({ source, ID },) {
              const obj = store.find(`${name}:${ID}`);
              if(name=='StringTemplate' && !Array.isArray(source)){
                source = [source]
              }      
              if (!source) throw Error(`Invalid input of [${source}] to ${name}`)
              if (obj.call) return { output: obj.call(source) };
              if (obj.acall) return { output: await obj.acall(source) };
            },
          }));
        })
    }

    const openaiApiKey = computed({
      get: () => { return store.find_all('OpenAIVendor:*')[0].api_key },
      set: (val) => { store.find_all('OpenAIVendor:*')[0].get_controller().update({ api_key: val }) },
    })

    if (localStorage.getItem('MakeAChain')) { load(); }
    else {
      const vendor = store.addNewOpenAIVendor('null');
      const systemPrompt = `You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
## Your Reply Format Example
\`\`\`translation
...
\`\`\``;
      store.addNewChatGPT4oMini("auto", systemPrompt);
      store.add_new_obj(new StringTemplate(`\`\`\`text\n{}\n\`\`\``));
      store.add_new_obj(new RegxExtractor('```translation\\s*(.*?)\\s*```'));
      loadUI();
    }

    // engine.start();
    // Function to execute the graph
    const runGraph = () => {
      engine.runOnce({ test: 'test' });
    };

    setInterval(() => {
      save();
      // console.log("Auto-saved at", new Date().toLocaleTimeString());
    }, 2000);

    // Watch for selected nodes
    const config_obj_id = ref('');
    onMounted(() => {
      watch(
        () => baklava.displayedGraph.selectedNodes,
        (newValue) => {
          if (!baklava.editor) return;
          if (newValue.length==0) return;
          const node = newValue[0];
          if(node.inputs.ID?.value){
            const obj = store.find_all(`*:${node.inputs.ID.value}`)[0];
            config_obj_id.value = obj.get_id();
          }
          else{
            config_obj_id.value = '';
          }
        }
      );
    });


    return { baklava, runGraph, openaiApiKey, vendorId, config_obj_id };
  }
}
</script>

<!-- <script>
import { ref, computed } from "vue";
import { StringTemplate, RegxExtractor } from '../libs/utils';
import { LLMsStore, Model4LLMs } from '../libs/LLMsModel';

export default {
  setup() {
    // Initialize store and add vendor/model
    const store = new LLMsStore();
    const openaiApiKey = computed({
      get: () => store.find_all('OpenAIVendor:*')[0]?.api_key || "",
      set: (val) => {
        const vendor = store.find_all('OpenAIVendor:*')[0];
        if (vendor) vendor.get_controller().update({ api_key: val });
      },
    });
    const vendor = store.addNewOpenAIVendor(null);
    const vendorId = vendor.get_id();

    const systemPrompt = `You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
## Your Reply Format Example
\`\`\`translation
...
\`\`\``;

    const llm = store.addNewChatGPT4oMini('auto');
    llm.system_prompt = systemPrompt;

    const translateTemplate = store.add_new_obj(
      new StringTemplate(`\`\`\`text\n{}\n\`\`\``)
    );
    const getResult = store.add_new_obj(
      new RegxExtractor('```translation\\s*(.*?)\\s*```')
    );

    // State variables
    const translationInput = ref("");
    const rawResponse = ref("");
    const extractedTranslation = ref("");

    // Functions
    const translateText = async () => {
      if (!translationInput.value) return;
      rawResponse.value = "";
      extractedTranslation.value = "";
      try {
        // Apply string template
        const templateOutput = translateTemplate.call([translationInput.value]);

        // Get LLM response
        const llmResponse = await llm.acall(templateOutput);

        rawResponse.value = llmResponse;

        // Extract the translation
        const extracted = getResult.call(llmResponse);
        extractedTranslation.value = extracted;
      } catch (error) {
        rawResponse.value = "Error in processing the request.";
        extractedTranslation.value = "";
        console.error(error);
      }
    };

    return {
      openaiApiKey,
      vendorId,
      translationInput,
      rawResponse,
      extractedTranslation,
      translateText,
    };
  },
};
</script> -->
