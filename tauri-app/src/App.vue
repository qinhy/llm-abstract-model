<template>

  <div>
    <Button icon="pi pi-bars" @click="visible = true" class="absolute bottom-0 left-0" />
    <Drawer v-model:visible="visible" header="Components">
      <div class="card flex justify-center">
        <div class="flex flex-wrap gap-4">
          <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient1" name="Chat" value="Chat" />
            <label for="ingredient1">Chat</label>
          </div>
          
          <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient2" name="FileUpload" value="FileUpload" />
            <label for="ingredient2">FileUpload</label>
          </div>
          
          <div class="flex items-center gap-2">
            <RadioButton v-model="ingredient" inputId="ingredient3" name="HowToUse" value="HowToUse" />
            <label for="ingredient3">HowToUse</label>
          </div>

        </div>
      </div>
    </Drawer>

    <Chat v-if="ingredient == 'Chat'"></Chat>
    <FileUpload v-if="ingredient == 'FileUpload'"></FileUpload>
    <HowToUse v-if="ingredient == 'HowToUse'"></HowToUse>

  </div>
</template>
<script lang="ts">
import FileUpload from "./components/FileUpload.vue";
import HowToUse from "./components/HowToUse.vue"
import Chat from "./components/Chat.vue";
import { ref } from "vue";
// import { ref } from "vue";
export default {
  components: {
    FileUpload, Chat, HowToUse
  },
  props: [],
  setup() {
    const visible = ref(false);
    const ingredient = ref('HowToUse');
    return {
      visible, ingredient
    }
  }
}
</script>