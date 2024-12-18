<template>
    <Toast />
    <p class="text-xl font-bold text-center px-4 py-2">
        {{ selected_chat.title }}
    </p>
    <Tabs value="Home" scrollable>
        <TabList>
            <Tab v-for="i in ['Home', 'Chat', 'Config', 'Help', 'RSA crypt']" :value="i">{{ i }}</Tab>
        </TabList>
        <TabPanels>
            <TabPanel value="Home">
                <div>
                    <Select :options="storage.all_chat().map(c => [c.title, c.uuid])"
                        @change="(e) => { selected_chat = storage.get_chat(e.value[1]) }"
                        :placeholder="selected_chat.title" />
                </div>
                <br>
                <div class="space-x-2">
                    <Button label="new"
                        @click="() => { selected_chat = storage.get_chat(storage.firstchat(null)) }"></Button>
                    <Button label="delete" severity="danger" @click="() => {
                        storage.del_chat(selected_chat.uuid);
                        const cs = storage.all_chat().map(c => c.uuid).pop();
                        if (cs) selected_chat = storage.get_chat(cs);
                    }"></Button>
                </div>
            </TabPanel>

            <TabPanel value="Chat">
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
                                    @click="() => { user_message.content = ' '; storage.del_msg(msg.uuid); user_message.content = '' }"></i>
                                <span class="italic text-xs">{{ msg.timestamp }}</span>
                            </div>
                            <hr>
                        </div>
                    </li>
                </ul><br>
                <hr>
                <hr><br>

                <div class="flex items-center space-x-2 p-2 border border-gray-300 rounded-lg shadow-sm">
                    <InputText v-model="user_message.content" class="w-full" placeholder="Type something for AI...">
                    </InputText>
                    <Button label="Send" @click="sendmsg"></Button>
                </div>

                <div class="flex items-center space-x-2 p-2 border border-gray-300 rounded-lg shadow-sm">
                    <div>
                        <span class="font-medium">Image:</span>
                        <ToggleSwitch v-model="image_on" class="ml-2"></ToggleSwitch>
                    </div>
                    <div v-if="image_on">
                        <input type="file" name="test" onchange="previewFile(this)" accept="image/*">
                        <img id="preview">
                    </div>
                </div>

            </TabPanel>

            <TabPanel value="Config">
                <div class="card"
                    v-for="config in [...storage.get_chat_configs('Global'), ...storage.get_chat_configs(selected_chat.uuid)]">
                    <label :for="config.name"> {{ config.what }}</label>
                    <div v-if="config.type == 'boolean'">
                        <ToggleSwitch :inputId="config.name" v-model="config.val" @click="command_darkmode" />
                    </div>

                    <div v-else-if="config.type == 'string'">
                        <InputText :id="config.name" v-model="config.val" />
                    </div>
                    <div v-else-if="config.type == 'number'">
                        <InputNumber :id="config.name" v-model="config.val" mode="decimal" showButtons :min="0"
                            :max="100" />
                    </div>
                </div>
            </TabPanel>

            <TabPanel value="Help">
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

                <p class="font-bold">Back data in JSON, for restore or backup</p>
                <div><Textarea v-model="storage_str" rows="5" cols="35" /></div>
            </TabPanel>

            <TabPanel value="RSA crypt">
                <div>
                    <FileUpload mode="basic" name="demo" accept=".pem" choose-label="Open Key" @select="onFileSelect" />
                    <div>
                        <Textarea v-if="fileContent" v-model="fileContent" disabled class="w-full"></Textarea>
                        <Button v-if="fileContent.includes('PUBLIC')" label="Encrypt"
                            @click="() => encryptRSA(fileContent)"></Button>
                        <Button v-if="fileContent.includes('PRIVATE')" label="Decrypt"
                            @click="() => decryptRSA(fileContent)"></Button>

                        <div v-if="fileContent.includes('PUBLIC')" class="card">
                            <span>{{ storage.dumps() }}</span>
                        </div>
                        <Textarea v-if="filedecode && fileContent.includes('PRIVATE')" v-model="filedecode" disabled
                            class="w-full"></Textarea>
                    </div>
                </div>
            </TabPanel>
        </TabPanels>
    </Tabs>

    <Dialog v-model:visible="isloading" modal :closable="false" header="Loading" class="w-half center-dialog">
        <i class="ml-4 pi pi-spinner pi-spin" style="font-size: 2rem"></i>
    </Dialog>

</template>


<script>
import { ref, computed } from "vue";
import { SingletonKeyValueStorage } from "../libs/Storage";
import { useToast } from 'primevue/usetoast';
import { PEMFileReader, SimpleRSAChunkEncryptor } from '../libs/RSA';

export default {
    components: {
    },
    setup() {
        const toast = useToast();
        const storage = new SingletonKeyValueStorage();
        storage.tempTsBackend();
        {
            const gen_uuid = (prefix = '') => {
                return prefix + 'xxxx-xxxx-xxxx-xxxx-xxxx'.replace(/x/g, function () {
                    return Math.floor(Math.random() * 16).toString(16);
                });
            }
            const get_now = () => new Date().toISOString();

            storage.load_local_storage = () => {
                const data = localStorage.getItem('single-file-vue-chat');
                if (data) {
                    storage.loads(atob(data));
                    return true;
                }
                return false;
            }

            storage.save_local_storage = () => {
                localStorage.setItem('single-file-vue-chat', btoa(storage.dumps()));
            }
            storage.new_chat = () => {
                const uuid = `Chat:${gen_uuid()}`;
                storage.set(uuid, { title: `Title:${uuid}`, msg_uuids: [], timestamp: get_now(), uuid: uuid });
                return storage.get(uuid);
            };
            storage.del_chat = (uuid) => {
                storage.get_chat(uuid)?.msg_uuids.forEach(m => storage.del_msg(m));
                storage.keys(`^Config:${uuid}:*`).map(k => storage.delete(k));
                storage.delete(uuid);
            }
            storage.get_chat = (uuid) => storage.get(uuid);
            storage.get_chat_msgs = (chat_uuid) => { return { msgs: storage.all_msgs(chat_uuid) } }
            storage.all_chat = (uuid) => { return storage.keys('^Chat:*').map(k => storage.get_chat(k)) };

            storage.get_msg = (uuid) => storage.get(uuid);
            storage.add_msg = (chat_uuid, { role, content }) => {
                const uuid = `Message:${gen_uuid()}`;
                storage.get_chat(chat_uuid).msg_uuids.push(uuid);
                storage.set(uuid, { role: role, content: content, timestamp: get_now(), uuid: uuid, chat: chat_uuid });
            };
            storage.add_img = (chat_uuid, { role, content }) => {
                const uuid = `Message:${gen_uuid()}`;
                storage.get_chat(chat_uuid).msg_uuids.push(uuid);
                storage.set(uuid, { role: role, content: content, timestamp: get_now(), uuid: uuid, chat: chat_uuid, is_img: true });
            };
            storage.del_msg = (msg_uuid) => {
                console.log(msg_uuid);
                const msg = storage.get_msg(msg_uuid);
                const chat = storage.get_chat(msg.chat);
                chat.msg_uuids = chat.msg_uuids.filter(m_uuid => m_uuid != msg_uuid);
                storage.delete(msg_uuid);
            };
            storage.all_msgs = (chat_uuid) => {
                const chat = storage.get_chat(chat_uuid);
                return chat ? chat.msg_uuids.map(m_uuid => storage.get_msg(m_uuid)) : [];
            };

            storage.set_chat_config = (chat_uuid, name, val, what) => {
                storage.set(`Config:${chat_uuid}:${name}`, { name: name, val: val, what: what, type: typeof (val) });
                return storage.get_chat_config(chat_uuid, name);
            }
            storage.get_chat_config = (chat_uuid, name) => storage.get(`Config:${chat_uuid}:${name}`);
            storage.get_chat_configs = (chat_uuid) => storage.keys(`^Config:${chat_uuid}:*`).map(k => storage.get(k));

            storage.firstchat = (title = 'Hello!') => {
                const chat = storage.new_chat();
                if (title) chat.title = title;
                else chat.title = chat.uuid;
                storage.add_msg(chat.uuid, { role: 'user', content: 'hi' });
                storage.add_msg(chat.uuid, { role: 'assistant', content: "I'm ready to assist you. What do you need help with?" });

                storage.set_chat_config(chat.uuid, 'sysp', 'You are a helpful assistant.', 'System Prompt');
                storage.set_chat_config(chat.uuid, 'modelname', 'gpt-4o-mini', 'Model Name');
                storage.set_chat_config(chat.uuid, 'sendlast', 4, 'Send last messages');
                storage.set_chat_config(chat.uuid, 'url', 'https://api.openai.com/v1/chat/completions', 'API URL');
                return chat.uuid;
            }
        }

        if (!storage.load_local_storage()) {
            storage.set_chat_config('Global', 'apikey', 'sk-', 'API Key');
            storage.firstchat();
        }

        const commands = ref({});
        const selected_chat = ref(storage.all_chat()[0]);
        selected_chat.get = () => selected_chat.value;

        storage.set_chat_config('Global', 'darkmode', true, 'Darkmode');
        const command_darkmode = () => document.getElementsByTagName('html')[0].classList.toggle('app-dark');
        command_darkmode();

        class GreetUser {
            static dict() {
                const name = GreetUser.description().name;
                return { name: GreetUser.greetUser }
            }
            static description() {
                return {
                    name: "greetUser",
                    description: "Greet the user by name.",
                    parameters: {
                        type: "object",
                        properties: {
                            name: { type: "string", description: "Name of the user" }
                        },
                        required: ["name"]
                    }
                };
            }
            static greetUser(name) {
                return `Hello, ${name}! Welcome to our website.`;
            }
        }
        storage.set_chat_config('Global', 'functions', { ...GreetUser.dict() }, 'Functions');

        const get_config = (name) => storage.get_chat_config(selected_chat.get()?.uuid, name)?.val;
        const get_functions = () => {
            const g_functions = storage.get_chat_config('Global', 'functions').val
            return [];//Object.entries(g_functions).map(e => e[0]);
            // get_config('functions');
            // res = new Function('return {log:console.log,alert:alert}')();
        }

        const user_message = ref({ role: 'user', content: '' });
        const ai_message = ref({ role: 'assistant', content: '' });
        ai_message.get = () => ai_message.value;

        const markdown_config = { render(content) { return content } };
        // window.markdownit({
        //     linkify: true, xhtmlOut: true, html: true, breaks: true,
        // });

        const history_repush = (msg) => {
            storage.del_msg(msg.uuid)
            user_message.value.content = msg.content;
            sendmsg();
        }

        // const toast = PrimeVue.useToast();
        const showInfo = (msg = 'Info Content', life = 1000) => {
            toast.add({ severity: 'info', summary: 'Info', detail: msg, life: life });
        };
        const showError = (msg = 'Error Content', life = 5000) => {
            console.error(msg);
            toast.add({ severity: 'error', summary: 'Error', detail: msg, life: life });
        };

        const openaibody = (sysp = null) => {
            const body = { model: get_config('modelname'), stream: true }
            if (get_functions().length > 0) {
                body.functions = get_functions();
                body.function_call = "auto";
            }

            body.messages = [{ role: 'system', content: get_config('sysp') },
            ...storage.get_chat_msgs(selected_chat.get()?.uuid)?.msgs];
            if (!sysp) sysp = body.messages[0];

            body.messages = [sysp, ...body.messages.filter(m => m.timestamp).slice(-parseInt(get_config('sendlast')))]
                .map(m => {
                    return m.is_img ?
                        { role: m.role, content: [{ type: "image_url", image_url: { url: `data:image/jpeg;base64,${m.content}` } }] } :
                        { role: m.role, content: m.content }
                })
            return JSON.stringify(body);
        }

        const headers = () => {
            return {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${storage.get_chat_config('Global', 'apikey').val}`
            }
        };
        const openaichat = async (target_message = null, sysp = null, at_end = null) => {
            if (!target_message) target_message = ai_message;
            const decoder = new TextDecoder('utf-8');
            try {
                const response = await fetch(get_config('url'), {
                    headers: headers(), method: 'POST', body: openaibody(sysp)
                });

                if (!response.ok) { // Checks if the status code is outside of the 2xx range
                    switch (response.status) {
                        case 400: throw new Error('Bad Request: The server could not understand the request.');
                        case 401: throw new Error('Unauthorized: Please check your credentials.');
                        case 403: throw new Error('Forbidden: You do not have permission to access this resource.');
                        case 404: throw new Error('Not Found: The requested resource could not be found.');
                        case 429: throw new Error('Too Many Requests: You have reached the rate limit.');
                        case 500: throw new Error('Internal Server Error: The server encountered an unexpected condition.');
                        case 503: throw new Error('Service Unavailable: The server is currently unable to handle the request.');
                        default: throw new Error(`An error occurred: ${response.statusText}`);
                    }
                }
                const reader = response.body.getReader();
                target_message.get().content = '';
                const stream = new ReadableStream({
                    start(controller) {
                        function push() {
                            // Read from the stream
                            reader.read().then(({ done, value }) => {
                                // When no more data needs to be consumed, close the stream
                                if (done) {
                                    controller.close();
                                    return;
                                }
                                const text = decoder.decode(value);
                                const lines = text.split(/\n+/);
                                for (const line of lines) {
                                    const json_text = line.replace(/^data:\s*/, '');
                                    if (json_text === '[DONE]') {
                                        if (at_end) at_end();
                                        else if (target_message.get().content.length > 0) {
                                            storage.add_msg(selected_chat.get().uuid, target_message.get());
                                            target_message.get().content = '';
                                        }
                                        return;
                                    }
                                    if (json_text.length == 0) continue;
                                    const data = JSON.parse(json_text);
                                    const content = data.choices[0].delta.content;
                                    if (content) target_message.get().content += content;
                                }
                                // Enqueue the next data chunk into our target stream
                                controller.enqueue(value);
                                push();
                            }).catch(e => {
                                showError(`${e.message}`);
                                controller.error(e);
                            });
                        }
                        push();
                    }
                });
                return await new Response(stream, { headers: { "Content-Type": "text/plain" } }).text();
            } catch (e_2) {
                return showError(`Failed to fetch: ${e_2.message}`);
            }
        }

        const mk_title = () => {
            const tmp = {}; tmp.get = () => tmp;
            openaichat(tmp, { role: 'system', content: 'Please reply summarizaion of all in 5 words.' },
                () => { selected_chat.get().title = tmp.content })
        }

        const sendmsg = () => {
            if (image_on.value) {
                const preview = document.getElementById('preview');
                const base64Data = preview.src;
                const base64Only = base64Data.split(',')[1];
                storage.add_img(selected_chat.get().uuid, { role: 'user', content: base64Only });
                preview.src = "";
            }

            storage.add_msg(selected_chat.get().uuid, user_message.value);
            openaichat();

            if (image_on.value) {
                image_on.value = false;
            }
            user_message.value.content = '';

            if (selected_chat.get().msg_uuids.length > 10 && selected_chat.get().title.includes(selected_chat.get().uuid)) mk_title();

        };
        const storage_str = computed({
            get() { return storage.dumps(); },
            set(val) { storage.loads(val) }
        });
        const image_on = ref(false);

        const isloading = ref(false);

        const fileContent = ref("");
        const filedecode = ref("");

        if (!fileContent.value && localStorage.getItem('single-file-vue-chat-encrypt')) {
            filedecode.value = localStorage.getItem('single-file-vue-chat-encrypt');
        }

        const onFileSelect = (event) => {
            const file = event.files[0];
            const reader = new FileReader();
            reader.onload = () => {
                fileContent.value = reader.result;
                if (fileContent.value.includes('PUBLIC')) {
                    filedecode.value = storage.dumps();
                }
                if (fileContent.value.includes('PRIVATE')) {
                    filedecode.value = localStorage.getItem('single-file-vue-chat-encrypt');
                }
            };
            reader.readAsText(file);
        };
        const encryptRSA = (publicKeyString) => {
            isloading.value = true;
            setTimeout(() => {
                if (!publicKeyString) publicKeyString = localStorage.getItem('single-file-vue-chat-publickey');
                filedecode.value = storage.dumpRSAs(publicKeyString, true);
                localStorage.setItem('single-file-vue-chat-publickey', publicKeyString);
                localStorage.setItem('single-file-vue-chat-encrypt', filedecode.value);
                localStorage.setItem('single-file-vue-chat-raw',storage.dumps());
                isloading.value = false;
            }, 100);
        }
        const decryptRSA = (privateKeyString) => {
            isloading.value = true;
            setTimeout(() => {
                if (localStorage.getItem('single-file-vue-chat-encrypt')) {
                    filedecode.value = localStorage.getItem('single-file-vue-chat-encrypt');
                }
                storage.clean();
                storage.loadRSAs(filedecode.value, privateKeyString);
                filedecode.value = storage.dumps();
                fileContent.value = localStorage.getItem('single-file-vue-chat-publickey');
                isloading.value = false;
            }, 100);
        }


        return {
            isloading,
            storage, storage_str, selected_chat, markdown_config, image_on,
            user_message, ai_message,
            sendmsg, showInfo, history_repush, openaibody, get_config, command_darkmode,
            onFileSelect, fileContent, filedecode, encryptRSA, decryptRSA,
        };
    },
};
</script>
