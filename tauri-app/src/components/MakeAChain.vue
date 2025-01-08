<template>
  <div :style="{width: '100vw', height: config_obj_id?'50vh':'100vh'}">
    <Button label="Run" icon="pi pi-play" @click="runGraph" />
    <BaklavaEditor :view-model="baklava" />
    <div v-if="config_obj_id" class="vendor-section bg-white shadow-lg p-6 rounded-lg mb-8">

      <div v-if="config_obj_id.split(':')[0]=='StringTemplate'">
        
      <div class="mb-4">
        <label for="StringTemplate" class="block text-gray-600 text-sm font-medium mb-2">String Template:</label>
        <InputText id="StringTemplate" v-model="stringTemplate" placeholder="Enter text for String Template..."
          class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
      </div>
      </div>

      <div v-if="config_obj_id.split(':')[0]=='RegxExtractor'">
        
        <div class="mb-4">
          <label for="RegxExtractor" class="block text-gray-600 text-sm font-medium mb-2">Regx Extractor:</label>
          <InputText id="RegxExtractor" v-model="regxExtractor" placeholder="Enter text for Regx Extractor..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
      </div>

      <div v-if="config_obj_id.split(':')[0]=='ChatGPT4oMini'">        
        <p class="text-sm text-gray-600">
          <strong>Vendor ID:</strong> {{ vendorId }}
        </p>
        <div class="mb-4">
          <label for="apiKey" class="block text-gray-600 text-sm font-medium mb-2">OpenAI API Key:</label>
          <InputText id="apiKey" v-model="openaiApiKey" placeholder="Enter text for OpenAI API Key..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
        <div class="mb-4">
          <label for="SystemPrompt" class="block text-gray-600 text-sm font-medium mb-2">System Prompt:</label>
          <Textarea id="SystemPrompt" v-model="systemPrompt" placeholder="Enter text for System Prompt ..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
      </div>
    </div>
    <div style="margin-top: 1px;">
    </div>
  </div>
</template>

<script lang="ts">
import { BaklavaEditor, useBaklava, DependencyEngine, applyResult } from "baklavajs";
import "@baklavajs/themes/dist/syrup-dark.css";
import { PromptInput, PromptOutput } from "./Nodes/CustomNodes";
import { defineNode, NodeInterface, TextInputInterface } from "baklavajs";
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
    // window.baklava = baklava;
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
      set: (val) => { return val; },
    })
    const save = () => {
      store.set("Graph", baklava.editor.save());
      localStorage.setItem('MakeAChain', store.dumps());
    }
      
    const load = () => {
      loadUI();
      store.clean();
      store.loads(localStorage.getItem('MakeAChain')??'{}');
      const g = store.get("Graph");
      if(g)baklava.editor.load({
        graph: g.graph,
        graphTemplates: g.graphTemplates
      });
    }

    store.addNewChatGPT4oMini().get_controller().delete();
    store.add_new_obj(new StringTemplate('')).get_controller().delete();
    store.add_new_obj(new RegxExtractor('')).get_controller().delete();

    const loadUI = () => {
      const objs = [
        new Model4LLMs.ChatGPT4oMini(),new StringTemplate(''),
        new RegxExtractor('')];
      objs.filter(obj => obj.hasOwnProperty('acall') || obj.hasOwnProperty('call'))
        .forEach(obj => {
          const name = obj.constructor.name;
          baklava.editor.registerNodeType(defineNode({
            type: name,
            inputs: {
              // ID: () => new SelectInterface("ID", ids[0], ids).setPort(false),
              source: () => new TextInputInterface("Text", ""),
            },
            outputs: { output: () => new NodeInterface<string>("Output", "null"), },
            onCreate(){
              Object.assign(this, { build: ()=>{
                if(this.hasOwnProperty('llm_obj'))return;
                const obj = store.find(this.title);
                if(obj){Object.assign(this, {'llm_obj':obj});}
                else{
                  const newobj = store.add_new_obj(new store.MODEL_CLASS_GROUP[name]());
                  Object.assign(this, {'llm_obj':newobj});                  
                  this.title = newobj.get_id();
                }
              } });
            },
            onDestroy(){
              this.llm_obj?.get_controller().delete();
            },
            async calculate({ source },) {
              this.build();
              const obj = this.llm_obj;//store.find(`${name}:${ID}`);
              if(name=='StringTemplate' && !Array.isArray(source)){
                source = [source]
              }      
              if (!source) throw Error(`Invalid input of [${source}] to ${name}`)
              if (obj.call) return { output: obj.call(source) };
              if (obj.acall) return { output: await obj.acall(source) };
            },
          }));
        });
    }    

    const openaiApiKey = computed({
      get: () => { return (store.find_all('OpenAIVendor:*')[0] as unknown as Model4LLMs.OpenAIVendor).api_key },
      set: (val) => { store.find_all('OpenAIVendor:*')[0].get_controller().update({ api_key: val }) },
    })

    const systemPrompt = computed({
      get: () => { return store.find(config_obj_id.value).system_prompt },
      set: (val) => { store.find(config_obj_id.value).get_controller().update({ system_prompt: val }) },
    })

    const stringTemplate = computed({
      get: () => { return store.find(config_obj_id.value).string },
      set: (val) => { store.find(config_obj_id.value).get_controller().update({ string: val }) },
    })

    const regxExtractor = computed({
      get: () => { return store.find(config_obj_id.value).regx },
      set: (val) => { store.find(config_obj_id.value).get_controller().update({ regx: val }) },
    })

    

    if (localStorage.getItem('MakeAChain')) { load(); }
    else {
      store.addNewOpenAIVendor('null');
//       const systemPrompt = `You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
// ## Your Reply Format Example
// \`\`\`translation
// ...
// \`\`\``;
      // store.addNewChatGPT4oMini("auto", systemPrompt);
      // store.add_new_obj(new StringTemplate(`\`\`\`text\n{}\n\`\`\``));
      // store.add_new_obj(new RegxExtractor('```translation\\s*(.*?)\\s*```'));
      loadUI();
      // baklava.editor.load({"graph":{"id":"f209b6fa-5498-4552-9c9a-c9fecb54b676","nodes":[{"type":"ChatGPT4oMini","id":"8404b184-09c9-4c15-bd54-bf3cefb616a8","title":"ChatGPT4oMini","inputs":{"ID":{"id":"a22af7e5-4777-4bf9-9872-c551fb515439","value":"906a-08ca-928b-e0a5-582c"},"source":{"id":"9dac03dd-656e-4b50-9df1-11159d39a794","value":""}},"outputs":{"output":{"id":"8f62d00f-ba10-473d-85f8-f1ae77cfa44e","value":"null"}},"position":{"x":312.7999683283151,"y":-128.6429774846618},"width":200,"twoColumn":false},{"type":"StringTemplate","id":"531b27b3-a36a-4ae6-a00f-f4644cd8ad9a","title":"StringTemplate","inputs":{"ID":{"id":"6a672382-f309-4fd0-a14b-63f0a0fdfbbd","value":"ea35-2a82-8ce4-b08c-c10f"},"source":{"id":"8288a42f-c629-42b5-aaf2-4d303409b438","value":""}},"outputs":{"output":{"id":"ae5d6995-c193-4cfb-808b-f84770ada3fa","value":"null"}},"position":{"x":89.28288142171132,"y":-130.34984751044357},"width":200,"twoColumn":false},{"type":"PromptInput","id":"18d9def2-8749-4bd8-960b-b67b23477795","title":"PromptInput","inputs":{"source":{"id":"277a3ab1-91fe-4d09-8376-b5271014da09","value":"Hi!"}},"outputs":{"output":{"id":"1a762dc5-f248-4cc9-935b-cafecdfbe2dd","value":"null"}},"position":{"x":-136.09209498739915,"y":-190.33197351695105},"width":200,"twoColumn":false},{"type":"RegxExtractor","id":"b0d0d5d2-fb9c-4cc8-9dcf-56bc673d1c37","title":"RegxExtractor","inputs":{"ID":{"id":"e2c3dbb0-5d9e-4f23-b9a2-f8ed9ebdde89","value":"c49c-5921-db8e-dc28-3ee0"},"source":{"id":"059bbba6-dac4-4d30-bed7-c28b7f00d9c3","value":""}},"outputs":{"output":{"id":"36627195-68fd-4a31-97f9-be2a6cb12de0","value":"null"}},"position":{"x":536.8131917198019,"y":-138.81769319486864},"width":200,"twoColumn":false},{"type":"PromptOutput","id":"6209f68a-542a-44ed-9db8-dc8e8ff92ac5","title":"Display","inputs":{"value":{"id":"b3ab4747-a870-4f84-a264-6d7e9e53924a","value":null}},"outputs":{"display":{"id":"b7d1ae1e-3351-4bfc-8903-22fa9d69d22f","value":"null"}},"position":{"x":761.5711123624714,"y":-97.56476680161879},"width":200,"twoColumn":false}],"connections":[{"id":"69a26221-ca04-4bb3-84e7-e5f996ba78bd","from":"ae5d6995-c193-4cfb-808b-f84770ada3fa","to":"9dac03dd-656e-4b50-9df1-11159d39a794"},{"id":"f304e713-8e6b-4849-b70f-0426fa1f9276","from":"1a762dc5-f248-4cc9-935b-cafecdfbe2dd","to":"8288a42f-c629-42b5-aaf2-4d303409b438"},{"id":"195f740b-f2dc-4d4a-9160-c7b61cbadbf8","from":"8f62d00f-ba10-473d-85f8-f1ae77cfa44e","to":"059bbba6-dac4-4d30-bed7-c28b7f00d9c3"},{"id":"d80c9a20-88ac-49fe-a167-e9f27210a75a","from":"36627195-68fd-4a31-97f9-be2a6cb12de0","to":"b3ab4747-a870-4f84-a264-6d7e9e53924a"}],"inputs":[],"outputs":[],"panning":{"x":1017.8907740695233,"y":306.57506062265844},"scaling":0.5903646427685914},"graphTemplates":[]})
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
          if (!baklava.editor){config_obj_id.value = ''; return;}
          if (newValue.length==0){config_obj_id.value = ''; return;}
          const node = newValue[0];
          const obj = store.find(node.title);
          
          if(obj){
            config_obj_id.value = obj.get_id();
            node.llm_obj = obj;
          }
          else if(node.hasOwnProperty('build')){
            node.build();
          }
          else{
            config_obj_id.value = '';
          }
        }
      );
    });


    return { baklava, runGraph, openaiApiKey, systemPrompt, stringTemplate, regxExtractor, vendorId, config_obj_id };
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
