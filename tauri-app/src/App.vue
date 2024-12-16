<template>
  <main class="card">
    <h1 class="text-4xl font-bold text-center text-gray-700 m-5">
      Welcome to Tauri + Vue
    </h1>

    <FileUpload></FileUpload>
  </main>
</template>
<script lang="ts">
import FileUpload from "./components/FileUpload.vue";
// import { ref } from "vue";
export default {
  components: {
    FileUpload
  },
  props: [],
  setup() {
  }
}
</script>