<template>

  <div>
    <Button icon="pi pi-bars" @click="visible = true" style="position: absolute; right: 0px;top: 0px; z-index: 10000;"/>
    <Drawer v-model:visible="visible" header="Components">
      <div class="card flex justify-center">
        <div class="flex flex-wrap gap-4">

          <!-- <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient1" name="Chat" value="Chat" />
            <label for="ingredient1">Chat</label>
          </div> -->

          <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient3" name="HowToUse" value="HowToUse" />
            <label for="ingredient3">HowToUse</label>
          </div>
          
          <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient2" name="MakeAChain" value="MakeAChain" />
            <label for="ingredient2">MakeAChain</label>
          </div>
          
          <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient2" name="CustomWorkflow" value="CustomWorkflow" />
            <label for="ingredient2">CustomWorkflow</label>
          </div>

        </div>
      </div>
    </Drawer>

    <Chat v-if="ingredient == 'Chat'"></Chat>
    <CustomWorkflow v-if="ingredient == 'CustomWorkflow'"></CustomWorkflow>
    <MakeAChain v-if="ingredient == 'MakeAChain'"></MakeAChain>
    <HowToUse v-if="ingredient == 'HowToUse'"></HowToUse>

  </div>
</template>
<script lang="ts">
import CustomWorkflow from "./components/CustomWorkflow.vue";
import MakeAChain from "./components/MakeAChain.vue";
import HowToUse from "./components/HowToUse.vue"
import Chat from "./components/Chat.vue";
import { ref } from "vue";
// import { ref } from "vue";
export default {
  components: {
    MakeAChain, Chat, HowToUse, CustomWorkflow
  },
  props: [],
  setup() {
    const visible = ref(false);
    const ingredient = ref('MakeAChain');
    return {
      visible, ingredient
    }
  }
}
</script>