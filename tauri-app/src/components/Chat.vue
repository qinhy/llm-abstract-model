<template>
    <p-toast />
    <p class="text-xl font-bold text-center px-4 py-2">
        {{ selected_chat.title }}
    </p>
    <p-tabs value="1">
        <p-tablist>
            <p-tab v-for="(c, i) in ['Home', 'Chat', 'Config', 'Help']" :value="`${i}`">{{ c }}</p-tab>
        </p-tablist>
        <p-tabpanels>
            <p-tabpanel value="0">
                <div>
                    <p-select :options="storage.all_chat().map(c => [c.title, c.uuid])"
                        @change="(e) => { selected_chat = storage.get_chat(e.value[1]) }"
                        :placeholder="selected_chat.title" />
                </div>
                <br>
                <div class="space-x-2">
                    <p-button label="new"
                        @click="() => { selected_chat = storage.get_chat(storage.firstchat(null)) }"></p-button>
                    <p-button label="delete" severity="danger" @click="() => {
                        storage.del_chat(selected_chat.uuid);
                        const cs = storage.all_chat().map(c => c.uuid).pop();
                        if (cs) selected_chat = storage.get_chat(cs);
                    }"></p-button>
                </div>
            </p-tabpanel>

            <p-tabpanel value="1">
                <ul class="text-base">
                    <hr>
                    <li v-for="msg in [...storage.get_chat_msgs(selected_chat.uuid).msgs, ai_message]">
                        <div v-if="msg.content && msg.content.length > 0">
                            <div class="markdown-body">
                                <div v-if="msg.is_img">
                                    <p class="font-bold">{{ msg.role }} :</p><img
                                        :src="`data:image/jpeg;base64,${msg.content}`">
                                </div>
                                <div v-else v-html="markdown_config.render(`**${msg.role}** : ${msg.content}`)">
                                </div>
                            </div>
                            <div class="flex justify-end">
                                <i class="pi pi-replay hover:text-blue-500 hover:scale-110 transition duration-200"
                                    @click="history_repush(msg)"></i>
                                <i class="pi pi-trash pi pi-replay hover:text-blue-500 hover:scale-110 transition duration-200"
                                    @click="() => storage.del_msg(msg.uuid)"></i>
                                <span class="italic text-xs">{{ msg.timestamp }}</span>
                            </div>
                            <hr>
                        </div>
                    </li>
                </ul><br>
                <hr>
                <hr><br>

                <div class="flex items-center space-x-2 p-2 border border-gray-300 rounded-lg shadow-sm">
                    <p-inputtext v-model="user_message.content" class="w-full"
                        placeholder="Type something for AI..."></p-inputtext>
                    <p-button label="Send" @click="sendmsg"></p-button>
                </div>

                <div class="flex items-center space-x-2 p-2 border border-gray-300 rounded-lg shadow-sm">
                    <div>
                        <span class="font-medium">Image:</span>
                        <p-toggleswitch v-model="image_on" class="ml-2"></p-toggleswitch>
                    </div>
                    <div v-if="image_on">
                        <input type="file" name="test" onchange="previewFile(this)" accept="image/*">
                        <img id="preview">
                    </div>
                </div>

            </p-tabpanel>
            <p-tabpanel value="2">
                <div class="card"
                    v-for="config in [...storage.get_chat_configs('Global'), ...storage.get_chat_configs(selected_chat.uuid)]">
                    <label :for="config.name"> {{ config.what }}</label>
                    <div v-if="config.type == 'boolean'"><p-toggleswitch :inputId="config.name" v-model="config.val"
                            @click="command_darkmode" /></div>

                    <div v-else-if="config.type == 'string'"><p-inputtext :id="config.name" v-model="config.val" /></div>
                    <div v-else-if="config.type == 'number'"><p-inputnumber :id="config.name" v-model="config.val"
                            mode="decimal" showButtons :min="0" :max="100" /></div>
                </div>
            </p-tabpanel>
            <p-tabpanel value="3">
                <hr>
                <p class="font-bold">You Configs</p>
                <div class="card">
                    <p
                        v-for="conf in [...storage.get_chat_configs('Global'), ...storage.get_chat_configs(selected_chat.uuid)]">
                        {{ conf }}</p>
                </div>
                <hr>
                <hr>

                <p class="font-bold">Data send to Server</p>
                <div class="card">{{ openaibody() }}</div>
                <hr>
                <hr>

                <p class="font-bold">Back end data in JSON, for restore or backup</p>
                <div><p-textarea v-model="storage_str" rows="5" cols="40" /></div>
            </p-tabpanel>
        </p-tabpanels>
    </p-tabs>

</template>