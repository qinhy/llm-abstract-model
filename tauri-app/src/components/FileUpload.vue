<template>
    <div class="file-upload-demo">
      <h2>Upload a Text File</h2>
      <FileUpload
        mode="basic"
        name="demo"
        accept=".txt"
        choose-label="Upload File"
        @select="onFileSelect"
      />
      <div v-if="fileContent">
        <h3>File Content:</h3>
        <Textarea v-model="fileContent" disabled class="w-full"></Textarea>
      </div>
    </div>
  </template>
  
  <script>
  import { ref } from "vue";
  import FileUpload from "primevue/fileupload";
  
  export default {
    components: {
      FileUpload
    },
    setup() {
      const fileContent = ref("");
  
      const onFileSelect = (event) => {
        const file = event.files[0];
        const reader = new FileReader();
        reader.onload = () => {
          fileContent.value = reader.result;
        };
        reader.readAsText(file);
      };
  
      return {
        fileContent,
        onFileSelect,
      };
    },
  };
  </script>
  