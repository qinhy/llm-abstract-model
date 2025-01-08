<template>
  <Toast/>
  <div :style="{width: '100vw', height: '100vh'}">
    <Button label="Run" icon="pi pi-play" @click="runGraph" style="position: absolute; right: 50px;top: 0px; z-index: 10000;"/>
    <BaklavaEditor :view-model="baklava" />
    <div v-if="config_obj_id" style="position: absolute; right: 0px; bottom: -30px; z-index: 10000;"
            class="vendor-section bg-white shadow-lg p-6 rounded-lg mb-8">

      <div v-if="config_obj_id.split(':')[0]=='StringTemplate'">
        
      <div class="mb-4">
        <label for="StringTemplate" class="block text-gray-600 text-sm font-medium mb-2">String Template:</label>
        <InputText id="StringTemplate" v-model="stringTemplate" placeholder="String Template..."
          class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
      </div>
      </div>

      <div v-if="config_obj_id.split(':')[0]=='RegxExtractor'">
        
        <div class="mb-4">
          <label for="RegxExtractor" class="block text-gray-600 text-sm font-medium mb-2">Regx Extractor:</label>
          <InputText id="RegxExtractor" v-model="regxExtractor" placeholder="Regx Extractor..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
      </div>

      <div v-if="config_obj_id.split(':')[0]=='ChatGPT4oMini'">        
        <p class="text-sm text-gray-600">
          <strong>Vendor ID:</strong> {{ vendorId }}
        </p>
        <div class="mb-4">
          <label for="apiKey" class="block text-gray-600 text-sm font-medium mb-2">OpenAI API Key:</label>
          <InputText id="apiKey" v-model="openaiApiKey" placeholder="OpenAI API Key..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
        <div class="mb-4">
          <label for="SystemPrompt" class="block text-gray-600 text-sm font-medium mb-2">System Prompt:</label>
          <Textarea id="SystemPrompt" v-model="systemPrompt" placeholder="System Prompt ..."
            class="p-inputtext w-full border border-gray-300 rounded-md shadow-sm focus:ring focus:ring-blue-200" />
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { BaklavaEditor, useBaklava, DependencyEngine, applyResult, TextInterface } from "baklavajs";
import "@baklavajs/themes/dist/syrup-dark.css";
import { PromptInput, PromptOutput, TimeSleepNode } from "./Nodes/CustomNodes";
import { defineNode, NodeInterface, TextInputInterface } from "baklavajs";
import { Model4LLMs, LLMsStore } from '../libs/LLMsModel';
import { computed, onMounted, ref, watch } from "vue";
import { RegxExtractor, StringTemplate } from "../libs/utils";
import { useToast } from 'primevue/usetoast';

export default {
  components: {
    BaklavaEditor
  },
  setup() {
    const store_name = 'CustomWorkflow';
    const llm_objs = [new Model4LLMs.ChatGPT4oMini(),new StringTemplate(''),new RegxExtractor('')];
    const node_types: Record<string, any> = {"TimeSleepNode":TimeSleepNode,
                                "PromptInput":PromptInput, "Display":PromptOutput};
    const store = new LLMsStore();
    const initial_nodes = ['PromptInput',
                          new StringTemplate(`\`\`\`text\n{}\n\`\`\``),
                          new RegxExtractor('```translation\\s*(.*?)\\s*```'),
                          'Display']

    const openaiApiKey = computed({
      get: () => { return (store.find_all('OpenAIVendor:*')[0] as unknown as Model4LLMs.OpenAIVendor).api_key },
      set: (val) => { store.find_all('OpenAIVendor:*')[0].get_controller().update({ api_key: val }) },
    })

    const systemPrompt = computed({
      get: () => { return (store.find(config_obj_id.value) as unknown as Model4LLMs.AbstractLLM).system_prompt },
      set: (val) => { (store.find(config_obj_id.value) as unknown as Model4LLMs.AbstractLLM)
                        .get_controller().update({ system_prompt: val }) },
    })

    const stringTemplate = computed({
      get: () => { return (store.find(config_obj_id.value) as unknown as StringTemplate).string },
      set: (val) => { (store.find(config_obj_id.value) as unknown as Model4LLMs.AbstractObj)
                        .get_controller().update({ string: val }) },
    })

    const regxExtractor = computed({
      get: () => { return (store.find(config_obj_id.value) as unknown as RegxExtractor).regx },
      set: (val) => { (store.find(config_obj_id.value) as unknown as Model4LLMs.AbstractObj)
                        .get_controller().update({ regx: val }) },
    })


    const toast = useToast();
    // set up interactive GUI
    const baklava = useBaklava();
    // window.baklava = baklava;
    const engine = new DependencyEngine(baklava.editor);
    const token = Symbol();

    engine.events.afterRun.subscribe(token, (result) => {
      engine.pause();
      applyResult(result, baklava.editor);
      engine.resume();
    });

    const vendorId = computed({
      get: () => { return store.find_all('OpenAIVendor:*')[0].get_id() },
      set: (val) => { return val; },
    })
    const save = () => {
      store.set("Graph", baklava.editor.save());
      localStorage.setItem(store_name, store.dumps());
    }
      
    const load = () => {
      store.clean();
      store.loads(localStorage.getItem(store_name)??'{}');
      const g = store.get("Graph");
      if(g)baklava.editor.load({
        graph: g.graph,
        graphTemplates: g.graphTemplates
      });
    }
    Object.entries(node_types).forEach((v: any)=>{
      baklava.editor.registerNodeType(v[1]);
    });
    llm_objs.forEach(obj => {
      store.add_new_obj(obj).get_controller().delete();
      const name = obj.constructor.name;
      node_types[name] = defineNode({
        type: name,
        inputs: {
          source: () => new TextInputInterface("Text", ""),
        },
        outputs: { 
          info: () => new TextInterface("Info", "null").setPort(false),
          output: () => new NodeInterface<string>("Output", "null"), 
        },
        onCreate(){
          Object.assign(this, { build: ()=>{
            if(this.hasOwnProperty('llm_obj')){
              const this_obj: { [key: string]: any } = { ...this };
              this.title = this_obj.llm_obj.get_id();
              return;}
            var obj = store.find(this.title);
            if(obj){Object.assign(this, {'llm_obj':obj});}
            else{
              const ClassConstructor = store.get_class(name);
              obj = store.add_new_obj(new ClassConstructor());
              Object.assign(this, {'llm_obj':obj});
            }
            this.title = obj?obj.get_id():'null';
          } });
        },
        onDestroy(){
          const this_obj: { [key: string]: any } = { ...this };
          this_obj.llm_obj?.get_controller().delete();
        },
        async calculate({ source },) {
          var output = 'null';
          try {
            if (!source) throw Error(`Invalid input of [${source}] to ${name}`)
            const this_obj: { [key: string]: any } = { ...this };
            this_obj.build();
            const obj = this_obj.llm_obj;//store.find(`${name}:${ID}`);
            if(name=='StringTemplate' && !Array.isArray(source)){
            if (obj.call) output = await obj.call([source]);
            }
            else{
            if (obj.call) output = await obj.call(source);
            if (obj.acall) output = await obj.acall(source);
          }
          } catch (error) {
            output = 'Run Error';
            if(output.includes('Error')){                  
              toast.add({ severity: 'error', 
                summary: 'Run Error', detail: error?.toString(), life: 3000 });
            }
          } finally{
            if(!output.includes('Error')){
              toast.add({ severity: 'success', 
                summary: 'success', detail: `[${name}]: ${output}`, life: 3000 });
            }
            return { output,info:output };
          }
        },
      })
      baklava.editor.registerNodeType(node_types[name]);
    });

    const addNodeWithCoordinates = (nodeType:string, x:number=0, y:number=0)=>{
            const n = new node_types[nodeType]();
            baklava.editor.graph.addNode(n);
            n.position.x = x;
            n.position.y = y;
            return n;
    }
    const addNodes = (nodes:any[])=>{
      nodes = nodes.map((v,i) =>{
          if(typeof v === "string")return addNodeWithCoordinates(v,300+i*225,100);
          else{
            const n = addNodeWithCoordinates(v.class_name(),300+i*225,100);
            Object.assign(n, {'llm_obj':store.add_new_obj(v)});
            return n;
          }
        });
        nodes.forEach((n,i)=>{
        if(i==nodes.length-1)return;
        baklava.editor.graph.addConnection(
              n.outputs.output,
              nodes[i+1].inputs.source
          );
        })
    }

    if (localStorage.getItem(store_name)) { load(); }
    else {
      store.addNewOpenAIVendor('null');      
      addNodes(initial_nodes);
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
            Object.assign(node, {'llm_obj':obj});
          }
          else if(node.hasOwnProperty('build')){
              const node_obj: { [key: string]: any } = { ...node };
              node_obj.build();
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