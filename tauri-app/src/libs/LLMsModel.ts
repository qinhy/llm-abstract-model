
import axios, { AxiosResponse } from "axios";
// import { TopologicalSorter } from "graphlib";
import { Controller4Basic, Model4Basic, BasicStore } from "./BasicModel";

export namespace Controller4LLMs {
    export class AbstractObjController extends Controller4Basic.AbstractObjController { };

    export class AbstractLLMController extends AbstractObjController {

        getVendor(auto: boolean = false): any {
            const store = this.storage();
            const model = this.model as unknown as Model4LLMs.AbstractLLM;
            if (!auto) {
                const vendor = store.find(model.vendor_id);

                if (!vendor) {
                    throw new Error(`vendor of ${model.vendor_id} is not exists! Please change_vendor(...)`);
                }
                return vendor;
            } else {
                const modelType = model.constructor.name;
                if (["ChatGPT4o", "ChatGPT4oMini"].includes(modelType)) {
                    const vendors = store.find_all("OpenAIVendor:*");
                    if (vendors.length === 0) {
                        throw new Error("auto get vendor of OpenAIVendor:* is not exists! Please (add and) change_vendor(...)");
                    }
                    return vendors[0];
                } else if (modelType === "Grok") {
                    const vendors = store.find_all("XaiVendor:*");
                    if (vendors.length === 0) {
                        throw new Error("auto get vendor of XaiVendor:* is not exists! Please (add and) change_vendor(...)");
                    }
                    return vendors[0];
                } else if (["Gemma2", "Phi3", "Llama"].includes(modelType)) {
                    const vendors = store.find_all("OllamaVendor:*");
                    if (vendors.length === 0) {
                        throw new Error("auto get vendor of OllamaVendor:* is not exists! Please (add and) change_vendor(...)");
                    }
                    return vendors[0];
                }
                throw new Error(`not support vendor of ${model.vendor_id}`);
            }
        }

        changeVendor(vendorId: string): any {
            const vendor = this.storage().find(vendorId);
            if (!vendor) {
                throw new Error(`vendor of ${vendorId} is not exists! Please do add new vendor`);
            }
            this.update({ vendor_id: vendor.get_id() });
            return vendor;
        }
    };

    export class OpenAIChatGPTController extends AbstractLLMController { }
    export class ChatGPT4oController extends AbstractLLMController { }
    export class ChatGPT4oMiniController extends AbstractLLMController { }
    export class ChatGPTO1Controller extends AbstractLLMController { }
    export class ChatGPTO1MiniController extends AbstractLLMController { }
    export class GrokController extends AbstractLLMController { }
    export class Gemma2Controller extends AbstractLLMController { }
    export class Phi3Controller extends AbstractLLMController { }
    export class LlamaController extends AbstractLLMController { }
    export class OpenAIVendorController extends AbstractLLMController { }

    export class AbstractEmbeddingController extends AbstractObjController {

        getVendor(auto: boolean = false): any {
            const store = this.storage();
            const model = this.model as unknown as Model4LLMs.AbstractEmbedding;
            if (!auto) {
                const vendor = store.find(model.vendor_id);
                if (!vendor) {
                    throw new Error(`vendor of ${model.vendor_id} is not exists! Please change_vendor(...)`);
                }
                return vendor;
            } else {
                if (model.constructor === Model4LLMs.TextEmbedding3Small) {
                    const vendors = store.find_all("OpenAIVendor:*");
                    if (vendors.length === 0) {
                        throw new Error("auto get vendor of OpenAIVendor:* is not exists! Please (add and) change_vendor(...)");
                    }
                    return vendors[0];
                }
                throw new Error(`not support vendor of ${model.vendor_id}`);
            }
        }
    };

    export class TextEmbedding3SmallController extends AbstractEmbeddingController { }

    export class WorkFlowController extends AbstractObjController {

        private _topologicalSort(graph: { [key: string]: string[] }): string[] {
            const inDegree: { [key: string]: number } = {};
            const zeroInDegree: string[] = [];
            const result: string[] = [];
            // Initialize in-degree for each node
            for (const node in graph) {
                if (!inDegree[node]) inDegree[node] = 0;
                for (const neighbor of graph[node]) {
                    if (!inDegree[neighbor]) inDegree[neighbor] = 0;
                    inDegree[neighbor]++;
                }
            }
            // Find all nodes with zero in-degree
            for (const node in inDegree) {
                if (inDegree[node] === 0) zeroInDegree.push(node);
            }
            // Process nodes with zero in-degree
            while (zeroInDegree.length > 0) {
                const node = zeroInDegree.pop()!;
                result.push(node);
                for (const neighbor of graph[node] || []) {
                    inDegree[neighbor]--;
                    if (inDegree[neighbor] === 0) zeroInDegree.push(neighbor);
                }
            }
            // Check for cycles (if all nodes are not processed)
            // if (result.length !== Object.keys(graph).length) {
            //     throw new Error("Graph has at least one cycle");
            // }

            return result;
        }
        private _readTask(taskId: string): any {
            return this.storage().find(taskId);
        }

        async _arun(): Promise<any> {
            const model = this.model as unknown as Model4LLMs.WorkFlow;

            const tasks: Record<string, string[]> = model.tasks;

            if (model.results["final"]) {
                if (model.results["input"]) {
                    model.results = { input: model.results["input"] };
                }
                else {
                    model.results = {}
                }
            }

            let result = null;

            for (const taskId of this._topologicalSort(tasks).reverse()) {
                if (model.results[taskId]) {
                    continue;
                }

                const dependencyResults = tasks[taskId].map((dep) => model.results[dep]);
                const allArgs = this._extractArgsKwargs(dependencyResults);

                try {

                    const obj = this._readTask(taskId);
                    if (obj['acall']) {
                        result = await obj.acall(...allArgs);
                    } else {
                        result = obj.call(...allArgs);
                    }

                    model.results[taskId] = result;
                } catch (e) {
                    throw new Error(`[WorkFlow]: Error at ${taskId}: ${e}`);
                }
            }

            if (result !== null) {
                model.results["final"] = result;
            }

            this.update({ results: model.results });
            return result;
        }

        private _extractArgsKwargs(dependencyResults: any[]): any[] {
            const allArgs: any[] = [];
            for (const argsKwargs of dependencyResults) {
                if (Array.isArray(argsKwargs)) {
                    allArgs.push(...argsKwargs);
                } else {
                    allArgs.push(argsKwargs);
                }
            }
            return allArgs;
        }
    };
}

// export class KeyOrEnv {
//     val: string
//     constructor(val: string) {
//         this.val = val;
//     }
//     get() {
//         return this.val
//     };
// }

export namespace Model4LLMs {
    export class AbstractObj extends Model4Basic.AbstractObj {
        init_controller(store: any): void {
            const controller_class = this._get_controller_class(Controller4LLMs);
            this._controller = new controller_class(store, this);
        }
    };

    export class AbstractVendor extends AbstractObj {
        vendor_name: string = 'AbstractVendor'; // e.g., 'OpenAI'
        api_url: string = 'https://api.yourai.com/v1/';; // e.g., 'https://api.openai.com/v1/'
        api_key?: string; // API key for authentication, if required
        timeout: number = 30; // Default timeout for API requests in seconds

        chat_endpoint?: string; // e.g., '/v1/chat/completions'
        models_endpoint?: string; // e.g., '/v1/models'
        embeddings_endpoint?: string;
        default_timeout: number = 30; // Timeout for API requests in seconds
        rate_limit?: number; // Requests per minute, if applicable

        constructor(init?: Partial<AbstractVendor>) {
            super();
            Object.assign(this, init);
        }

        formatLLMModelName(llmModelName: string): string {
            return llmModelName;
        }

        getApiKey(): string {
            return this.api_key ?? "";
        }

        async getAvailableModels(): Promise<any> {
            const url = this._buildUrl(this.models_endpoint || "");
            const headers = this._buildHeaders();
            try {
                const response: AxiosResponse = await axios.get(url, { headers, timeout: this.timeout * 1000 });
                return response.data;
            } catch (error) {
                return { error: error?.toString() };
            }
        }

        private _buildHeaders(): Record<string, string> {
            const headers: Record<string, string> = {
                "Content-Type": "application/json",
            };
            if (this.api_key) {
                headers["Authorization"] = `Bearer ${this.getApiKey()}`;
            }
            return headers;
        }

        private _buildUrl(endpoint: string): string {
            /**
             * Construct the full API URL for a given endpoint.
             * @param endpoint API endpoint to be called.
             * @return Full API URL.
             */
            return `${this.api_url.replace(/\/+$/, "")}/${endpoint.replace(/^\/+/, "")}`;
        }

        async chatRequest(payload: Record<string, any> = {}): Promise<any> {
            const url = this._buildUrl(this.chat_endpoint || "");
            const headers = this._buildHeaders();
            try {
                const response: AxiosResponse = await axios.post(
                    url, JSON.stringify(payload), { headers, timeout: this.timeout * 1000 });
                return response.data;
            } catch (error: any) {
                return { error: error.response ? error.response.data : error.toString() };
            }
        }

        async *chatStreamRequest(payload: Record<string, any> = {}) {
            const url = this._buildUrl(this.chat_endpoint || "");
            const headers = this._buildHeaders();
            try {
                payload.stream = true;

                const response = await fetch(url, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(this.timeout * 1000),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const reader = response.body?.getReader();
                if (!reader) {
                    throw new Error("Failed to obtain reader from the response body");
                }

                const decoder = new TextDecoder();
                let done = false;

                while (!done) {
                    const { value, done: readerDone } = await reader.read();
                    done = readerDone;

                    if (value) {
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk
                            .toString()
                            .split('\n')
                            .filter((line: string) => line.trim() !== '');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6); // Remove 'data: ' prefix
                                if (data === '[DONE]') return; // End of stream
                                yield this.chatStreamResult(data);
                            }
                        }
                    }
                }
            } catch (error: any) {
                console.error('Error in streaming:', error.message || error);
            }
        }


        chatResult(response: any): string {
            return response;
        }

        chatStreamResult(response: any): string {
            return response;
        }

        async embeddingRequest(payload: Record<string, any> = {}): Promise<any> {
            /**
             * Method to make embedding requests. Requires the payload to have 'model' and 'input' keys.
             */
            if (!this.embeddings_endpoint) {
                return { error: "Embeddings endpoint not defined for this vendor." };
            }

            const url = this._buildUrl(this.embeddings_endpoint);
            const headers = this._buildHeaders();
            try {
                const response: AxiosResponse = await axios.post(url, JSON.stringify(payload), { headers, timeout: this.timeout * 1000 });
                return response.data?.data?.[0]?.embedding || { error: "Invalid response format." };
            } catch (error: any) {
                return { error: error.response ? error.response.data : error.toString() };
            }
        }

        async getEmbedding(text: string, model: string): Promise<any> {
            const payload = { model, input: text };
            return this.embeddingRequest(payload);
        }

        getController(): Controller4LLMs.AbstractObjController | null {
            return this._controller;
        }

        initController(store: any): void {
            this._controller = new Controller4LLMs.AbstractObjController(store, this);
        }
    };

    export class AbstractLLM extends Model4LLMs.AbstractObj {
        vendor_id: string = "auto";
        llm_model_name: string = "null";
        context_window_tokens: number = 0;
        max_output_tokens: number = 0;
        stream: boolean = false;

        limit_output_tokens?: number;
        temperature?: number = 0.7;
        top_p?: number = 1.0;
        frequency_penalty?: number = 0.0;
        presence_penalty?: number = 0.0;
        system_prompt?: string;

        constructor(init?: Partial<Model4LLMs.AbstractLLM>) {
            super();
            Object.assign(this, init);
        }

        getVendor(): Model4LLMs.AbstractVendor {
            return this.getController().getVendor(this.vendor_id === "auto");
        }

        getUsageLimits(): Record<string, any> | void { }

        validateInput(prompt: string): boolean {
            if (prompt.length > this.context_window_tokens) {
                throw new Error(`Input exceeds the maximum token limit of ${this.context_window_tokens}.`);
            }
            return true;
        }

        calculateCost(tokensUsed: number = 0.0): number {
            return tokensUsed;
        }

        getTokenCount(text: string): number {
            return text.split(/\s+/).length;
        }

        buildSystem(purpose: string = "..."): string {
            return `Dummy implementation for building system prompt ${purpose}`;
        }

        constructPayload(messages: Record<string, any>[]): Record<string, any> {
            const payload: Record<string, any> = {
                model: this.getVendor().formatLLMModelName(this.llm_model_name),
                stream: this.stream,
                messages,
                max_tokens: this.limit_output_tokens,
                temperature: this.temperature,
                top_p: this.top_p,
                frequency_penalty: this.frequency_penalty,
                presence_penalty: this.presence_penalty,
            };
            return Object.fromEntries(Object.entries(payload).filter(([_, v]) => v !== null));
        }

        constructMessages(messages: string | Record<string, any>[]): Record<string, any>[] {
            const msgs: Record<string, any>[] = [];
            if (this.system_prompt) {
                msgs.push({ role: "system", content: this.system_prompt });
            }
            if (typeof messages === "string") {
                messages = [{ role: "user", content: messages }];
            }
            return msgs.concat(messages);
        }

        async acall(messages: string | Record<string, any>[], autoStr: boolean = true): Promise<string> {
            if (typeof messages !== "string" && !Array.isArray(messages) && autoStr) {
                messages = String(messages);
            }
            const payload = this.constructPayload(this.constructMessages(messages));
            const vendor = this.getVendor();
            const response = await vendor.chatRequest(payload);
            return vendor.chatResult(response);
        }

        async *scall(messages: string | Record<string, any>[], autoStr: boolean = true) {
            if (typeof messages !== "string" && !Array.isArray(messages) && autoStr) {
                messages = String(messages);
            }
            const payload = this.constructPayload(this.constructMessages(messages));

            const vendor = this.getVendor();
            for await (const chunk of vendor.chatStreamRequest(payload)) {
                if (chunk) yield chunk;
            }
        }

        getController(): Controller4LLMs.AbstractLLMController {
            return this._controller;
        }

        initController(store: any): void {
            this._controller = new Controller4LLMs.AbstractLLMController(store, this);
        }
    };

    export class OpenAIVendor extends Model4LLMs.AbstractVendor {
        vendor_name: string = "OpenAI";
        api_url: string = "https://api.openai.com";
        chat_endpoint: string = "/v1/chat/completions";
        models_endpoint: string = "/v1/models";
        embeddings_endpoint: string = "/v1/embeddings";
        default_timeout: number = 30;
        rate_limit?: number = 0;

        async getAvailableModels(): Promise<Record<string, any>> {
            const response = await super.getAvailableModels();
            if (!response || !response.data) {
                throw new Error("Failed to fetch models or invalid response format.");
            }
            return response.data.reduce((models: Record<string, any>, model: any) => {
                models[model.id] = model;
                return models;
            }, {});
        }

        chatResult(response: any): string {
            try {
                const content = response?.choices?.[0]?.message?.content;
                if (!content) {
                    throw new Error(`Cannot get result from response: ${JSON.stringify(response)}`);
                }
                return content;
            } catch (error) {
                console.error(`Error in chatResult: ${error}`);
                throw new Error(`Failed to extract chat result.`);
            }
        }

        chatStreamResult(response: any) {
            const json = JSON.parse(response);
            if (json.choices && json.choices.length > 0) {
                const delta = json.choices[0].delta || {};
                if (delta.content) {
                    return delta.content;
                }
            }
        }

        async getEmbedding(text: string, model: string = "text-embedding-3-small"): Promise<any> {
            const payload = { model, input: text };
            const response = await this.embeddingRequest(payload);
            if (response.error) {
                console.error(`Error in embedding request: ${response.error}`);
                throw new Error(response.error);
            }
            return response;
        }
    };

    export class OpenAIChatGPT extends Model4LLMs.AbstractLLM {
        limit_output_tokens: number = 1024;
        temperature: number = 0.7;
        top_p: number = 1.0;
        frequency_penalty: number = 0.0;
        presence_penalty: number = 0.0;
        system_prompt?: string = '';

        stop_sequences: string[] = [];
        n: number = 1;

        constructor(init?: Partial<Model4LLMs.OpenAIChatGPT>) {
            super();
            Object.assign(this, init);
        }

        constructPayload(messages: Record<string, any>[]): Record<string, any> {
            const payload = super.constructPayload(messages);
            payload.stop = this.stop_sequences;
            payload.n = this.n;
            return Object.fromEntries(Object.entries(payload).filter(([_, v]) => v !== null));
        }
    };

    export class ChatGPT4o extends Model4LLMs.OpenAIChatGPT {
        llm_model_name: string = "gpt-4o";
        context_window_tokens: number = 128000;
        max_output_tokens: number = 4096;
    };

    export class ChatGPT4oMini extends Model4LLMs.ChatGPT4o {
        llm_model_name: string = "gpt-4o-mini";
    };

    export class ChatGPTO1 extends Model4LLMs.OpenAIChatGPT {
        limit_output_tokens: number = 2048;
        llm_model_name: string = "o1-preview";
        context_window_tokens: number = 128000;
        max_output_tokens: number = 32768;
        temperature: number = 1.0;

        constructPayload(messages: Record<string, any>[]): Record<string, any> {
            const payload = {
                model: this.getVendor().formatLLMModelName(this.llm_model_name),
                stream: this.stream,
                messages,
                max_completion_tokens: this.limit_output_tokens,
                top_p: this.top_p,
                frequency_penalty: this.frequency_penalty,
                presence_penalty: this.presence_penalty,
            };
            return Object.fromEntries(Object.entries(payload).filter(([_, v]) => v !== null));
        }

        constructMessages(messages: string | Record<string, any>[]): Record<string, any>[] {
            if (typeof messages === "string") {
                messages = [{ role: "user", content: messages }];
            }
            if (this.system_prompt) {
                messages[0].content = `${this.system_prompt}\n${messages[0].content}`;
            }
            return messages;
        }
    };

    export class ChatGPTO1Mini extends Model4LLMs.ChatGPTO1 {
        llm_model_name: string = "o1-mini";
        context_window_tokens: number = 128000;
        max_output_tokens: number = 65536;
    };

    export class XaiVendor extends Model4LLMs.OpenAIVendor {
        vendor_name: string = "xAI";
        api_url: string = "https://api.x.ai";
        chat_endpoint: string = "/v1/chat/completions";
        models_endpoint: string = "/v1/models";
        embeddings_endpoint: string = "/v1/embeddings";
        rate_limit?: number = 0;
    };

    export class Grok extends Model4LLMs.OpenAIChatGPT {
        llm_model_name: string = "grok-beta";
        context_window_tokens: number = 128000;
        max_output_tokens: number = 4096;
    };

    export class OllamaVendor extends Model4LLMs.AbstractVendor {
        vendor_name: string = "Ollama";
        api_url: string = "http://localhost:11434";
        chat_endpoint: string = "/api/chat";
        models_endpoint: string = "/api/tags";
        embeddings_endpoint: string = "NULL";

        async getAvailableModels(): Promise<Record<string, any>> {
            const response = await this.getAvailableModels();
            if (!response) {
                throw new Error("Cannot get available models");
            }
            return response.data.models.reduce((models: Record<string, any>, model: any) => {
                models[model.name] = model;
                return models;
            }, {});
        }

        formatLLMModelName(llmModelName: string): string {
            llmModelName = llmModelName.toLowerCase().replace("meta-", "");
            const parts = llmModelName.split("-");
            if (parts.length < 3) {
                throw new Error(`Cannot parse name of ${llmModelName}`);
            }
            return `${parts[0]}${parts[1]}:${parts[2]}`;
        }

        chatResult(response: any): string {
            if (!response?.message?.content) {
                throw new Error(`Cannot get result from ${response}`);
            }
            return response.message.content;
        }
    };

    export class Gemma2 extends Model4LLMs.AbstractLLM {
        llm_model_name: string = "gemma-2-2b";
        context_window_tokens: number = -1;
        max_output_tokens: number = -1;
    };

    export class Phi3 extends Model4LLMs.AbstractLLM {
        llm_model_name: string = "phi-3-3.8b";
        context_window_tokens: number = -1;
        max_output_tokens: number = -1;
    };

    export class Llama extends Model4LLMs.AbstractLLM {
        llm_model_name: string = "llama-3.2-3b";
        context_window_tokens: number = -1;
        max_output_tokens: number = -1;
    };
    export class AbstractEmbedding extends Model4LLMs.AbstractObj {
        vendor_id: string = "auto"; // Vendor identifier (e.g., OpenAI, Google)
        embedding_model_name: string = "text-embedding"; // Model name (e.g., "text-embedding-3-small")
        embedding_dim: number = 0; // Dimensionality of the embeddings, e.g., 768 or 1024
        normalize_embeddings: boolean = true; // Whether to normalize the embeddings to unit vectors

        max_input_length?: number; // Optional limit on input length (e.g., max tokens or chars)
        pooling_strategy: string = "mean"; // Pooling strategy if working with sentence embeddings (e.g., "mean", "max")
        distance_metric: string = "cosine"; // Metric for comparing embeddings ("cosine", "euclidean", etc.)

        cache_embeddings: boolean = false; // Option to cache embeddings to improve efficiency
        cache?: Record<string, number[]>; // Cache to store embeddings
        embedding_context?: string; // Optional context or description to customize embedding generation
        additional_features?: string[]; // Additional features for embeddings, e.g., "entity", "syntax"

        async acall(inputText: string): Promise<number[]> {
            return await this.generateEmbedding(inputText);
        }

        async getVendor(): Promise<any> {
            return this.getController().getVendor(this.vendor_id === "auto");
        }

        async generateEmbedding(inputText: string): Promise<number[]> {
            throw new Error(`This method should be implemented by subclasses.${inputText}`);
        }

        similarityScore(embedding1: number[], embedding2: number[]): number {
            throw new Error(`This method should be implemented by subclasses.${embedding1},${embedding2}`);
        }

        getController(): any {
            return this._controller;
        }

        initController(store: any): void {
            this._controller = new Controller4LLMs.AbstractEmbeddingController(store, this);
        }
    };

    export class TextEmbedding3Small extends Model4LLMs.AbstractEmbedding {
        vendor_id: string = "auto";
        embedding_model_name: string = "text-embedding-3-small";
        embedding_dim: number = 1536;
        normalize_embeddings: boolean = true;
        max_input_length: number = 8192;

        async acall(inputText: string): Promise<number[]> {
            // Check for cached result
            if (this.cache_embeddings && this.cache && inputText in this.cache) {
                return this.cache[inputText];
            }

            // Generate embedding using Vendor API
            const vendor = await this.getVendor();
            let embedding = await vendor.getEmbedding(inputText, this.embedding_model_name);

            // Normalize if specified
            if (this.normalize_embeddings) {
                embedding = this._normalizeEmbedding(embedding);
            }

            // Cache result if caching is enabled
            if (this.cache_embeddings) {
                if (!this.cache) {
                    this.cache = {};
                }
                this.cache[inputText] = embedding;
            }

            return embedding;
        }

        similarityScore(embedding1: number[], embedding2: number[]): number {
            if (this.distance_metric === "cosine") {
                return this._cosineSimilarity(embedding1, embedding2);
            } else if (this.distance_metric === "euclidean") {
                return this._euclideanDistance(embedding1, embedding2);
            } else {
                throw new Error("Unsupported distance metric. Choose 'cosine' or 'euclidean'.");
            }
        }

        private _normalizeEmbedding(embedding: number[]): number[] {
            const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
            return norm !== 0 ? embedding.map((x) => x / norm) : embedding;
        }

        private _cosineSimilarity(embedding1: number[], embedding2: number[]): number {
            const dotProduct = embedding1.reduce((sum, x, i) => sum + x * embedding2[i], 0);
            const norm1 = Math.sqrt(embedding1.reduce((sum, x) => sum + x * x, 0));
            const norm2 = Math.sqrt(embedding2.reduce((sum, x) => sum + x * x, 0));
            return norm1 !== 0 && norm2 !== 0 ? dotProduct / (norm1 * norm2) : 0.0;
        }

        private _euclideanDistance(embedding1: number[], embedding2: number[]): number {
            return Math.sqrt(embedding1.reduce((sum, x, i) => sum + (x - embedding2[i]) ** 2, 0));
        }
    };

    export class Function extends Model4LLMs.AbstractObj {
        // static paramDescriptions(description: string, descriptions: Record<string, string>) {
        //     return function (func: Function): Function {
        //         func._parametersDescription = descriptions;
        //         func._description = description;
        //         return func;
        //     };
        // }

        name: string = "null";
        description: string = "null";
        _description: string = "null";
        _properties: Record<string, { type: string; description: string }> = {};
        parameters: Record<string, any> = { type: "object", properties: this._properties };
        required: string[] = [];
        _parametersDescription: Record<string, string> = {};
        _stringArguments: string = "{}";

        constructor(...args: any[]) {
            super(...args);
            this._extractSignature();
        }

        private _extractSignature(): void {
            this.name = this.constructor.name;
            // const sig = Reflect.getMetadata("design:paramtypes", this.__call__) || [];

            // const typeMap: Record<any, string> = {
            //     Number: "number",
            //     String: "string",
            //     Boolean: "boolean",
            //     Array: "array",
            //     Object: "object",
            // };

            this.required = [];
            // for (const [name, type] of Object.entries(sig)) {
            //     const paramType = typeMap[type as any] || "object";
            //     this._properties[name] = {
            //         type: paramType,
            //         description: this._parametersDescription[name] || "",
            //     };
            //     if (!type) this.required.push(name);
            // }

            this.parameters.properties = this._properties;
            this.description = this._description;
        }

        call(...args: any[]): any {
            throw new Error(`This method should be implemented by subclasses.${args}`);
        };

        // getDescription(): Record<string, any> {
        //     return this;
        // }
    }

    export class WorkFlow extends Model4LLMs.AbstractObj {
        tasks: Record<string, string[]> = {}; // Map of task dependencies
        results: Record<string, any> = {};

        constructor(data?: any) {
            super(data);
            this.tasks = data?.tasks || {};
            this.results = data?.results || {};
        }

        async acall(args: any[]): Promise<any> {
            if (Array.isArray(args) && args.length !== 0) {
                const firstTaskId = Object.keys(this.tasks).pop()!;
                const firstTaskDeps = this.tasks[firstTaskId] || [];
                if (!firstTaskDeps.includes("input")) {
                    this.tasks[firstTaskId].push("input");
                }
                this.results["input"] = [args];
            } else {
                this.results = {
                    ...this.results,
                    ...args
                }
            }
            return await this.getController()._arun();
        }

        getResult(taskId: string): any {
            return this.results[taskId] || null;
        }

        getController(): any {
            return this._controller;
        }

        initController(store: any): void {
            this._controller = new Controller4LLMs.WorkFlowController(store, this);
        }
    }

    // @Function.paramDescriptions(
    //     "Makes an HTTP request using the configured method, url, headers, and the provided params, data, or JSON.",
    //     {
    //         params: "Query parameters",
    //         data: "Form data",
    //         json: "JSON payload",
    //     }
    // )
    export class RequestsFunction extends Function {
        method: string = "GET";
        url!: string;
        headers: Record<string, string> = {};

        //     async acall(
        //         params: Record<string, any> = {},
        //         data: Record<string, any> = {},
        //         json: Record<string, any> = {},
        //         debug: boolean = false,
        //         debugData: any = null
        //     ): Promise<Record<string, any>> {
        //         if (debug) return debugData;

        //         try {
        //             const response = await axios({
        //                 method: this.method,
        //                 url: this.url,
        //                 headers: this.headers,
        //                 params,
        //                 data,
        //                 json,
        //             });
        //             return response.data || { text: response.statusText };
        //         } catch (error: any) {
        //             return { error: error.message, status: error.response?.status || null };
        //         }
        //     }
        // }

    }
}
export class LLMsStore extends BasicStore {
    MODEL_CLASS_GROUP = Model4LLMs;

    get_class(id: string): typeof Model4Basic.AbstractObj | typeof Model4Basic.AbstractGroup {
        return this._get_class(id);
    }

    protected _get_class(id: string): typeof Model4Basic.AbstractObj | typeof Model4Basic.AbstractGroup {
        const class_type = id.split(':')[0];
        const classes: Record<string, any> = {};
        // Dynamically add all classes from Model4LLMs
        Object.entries(this.MODEL_CLASS_GROUP).forEach((val) => {
            classes[val[0]] = val[1];
        });
        const res = classes[class_type];
        if (!res) throw new Error(`No such class of ${class_type}`);
        return res;
    }

    protected _get_as_obj(id: string, data_dict: Record<string, any>): Model4Basic.AbstractObj {
        const ClassConstructor = this._get_class(id);
        const obj = new ClassConstructor();
        Object.assign(obj, data_dict);
        obj.set_id(id).init_controller(this);
        return obj;
    }

    protected _add_new_obj(obj: Model4Basic.AbstractObj, id: string | null = null): Model4Basic.AbstractObj {
        if (!this.MODEL_CLASS_GROUP.hasOwnProperty(obj.constructor.name)) {
            var tmp: { [key: string]: any } = {};
            tmp[obj.constructor.name] = obj.constructor;
            Object.assign(this.MODEL_CLASS_GROUP, tmp);
        }
        id = id === null ? obj.gen_new_id() : id;
        const data = obj.model_dump_json_dict();
        this.set(id, data);
        return this._get_as_obj(id, data);
    }

    addNewOpenAIVendor(apiKey: string, apiUrl: string = "https://api.openai.com", timeout: number = 30): Model4LLMs.OpenAIVendor {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.OpenAIVendor({ api_url: apiUrl, api_key: apiKey, timeout }));
    }

    addNewXaiVendor(apiKey: string, apiUrl: string = "https://api.x.ai", timeout: number = 30): Model4LLMs.XaiVendor {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.XaiVendor({ api_url: apiUrl, api_key: apiKey, timeout }));
    }

    addNewOllamaVendor(apiUrl: string = "http://localhost:11434", timeout: number = 30): Model4LLMs.OllamaVendor {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.OllamaVendor({ api_url: apiUrl, api_key: "", timeout }));
    }

    addNewChatGPT4o(
        vendorId: string,
        systemPrompt: string = '',
        limitOutputTokens: number = 1024,
        temperature: number = 0.7,
        topP: number = 1.0,
        frequencyPenalty: number = 0.0,
        presencePenalty: number = 0.0,
        id?: string
    ): Model4LLMs.ChatGPT4o {
        return this.add_new_obj(
            new this.MODEL_CLASS_GROUP.ChatGPT4o({
                vendor_id: vendorId,
                limit_output_tokens: limitOutputTokens,
                temperature,
                top_p: topP,
                frequency_penalty: frequencyPenalty,
                presence_penalty: presencePenalty,
                system_prompt: systemPrompt,
            }),
            id
        );
    }

    addNewChatGPT4oMini(
        vendorId: string = "auto",
        systemPrompt: string = '',
        limitOutputTokens: number = 1024,
        temperature: number = 0.7,
        topP: number = 1.0,
        frequencyPenalty: number = 0.0,
        presencePenalty: number = 0.0,
        id?: string
    ): Model4LLMs.ChatGPT4oMini {
        return this.add_new_obj(
            new this.MODEL_CLASS_GROUP.ChatGPT4oMini({
                vendor_id: vendorId,
                limit_output_tokens: limitOutputTokens,
                temperature,
                top_p: topP,
                frequency_penalty: frequencyPenalty,
                presence_penalty: presencePenalty,
                system_prompt: systemPrompt,
            }),
            id
        );
    }

    addNewGrok(
        vendorId: string,
        systemPrompt: string = '',
        limitOutputTokens: number = 1024,
        temperature: number = 0.7,
        topP: number = 1.0,
        frequencyPenalty: number = 0.0,
        presencePenalty: number = 0.0,
        id?: string
    ): Model4LLMs.Grok {
        return this.add_new_obj(
            new this.MODEL_CLASS_GROUP.Grok({
                vendor_id: vendorId,
                limit_output_tokens: limitOutputTokens,
                temperature,
                top_p: topP,
                frequency_penalty: frequencyPenalty,
                presence_penalty: presencePenalty,
                system_prompt: systemPrompt,
            }),
            id
        );
    }

    addNewGemma2(vendorId: string, systemPrompt: string = '', id?: string): Model4LLMs.Gemma2 {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.Gemma2({ vendor_id: vendorId, system_prompt: systemPrompt }), id);
    }

    addNewPhi3(vendorId: string, systemPrompt: string = '', id?: string): Model4LLMs.Phi3 {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.Phi3({ vendor_id: vendorId, system_prompt: systemPrompt }), id);
    }

    addNewLlama(vendorId: string, systemPrompt: string = '', id?: string): Model4LLMs.Llama {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.Llama({ vendor_id: vendorId, system_prompt: systemPrompt }), id);
    }

    addNewFunction(functionObj: Model4LLMs.Function, id?: string): Model4LLMs.Function {
        return this.add_new_obj(functionObj, id);
    }

    addNewRequest(url: string, method: string = "GET", headers: Record<string, any> = {}, id?: string): Model4LLMs.RequestsFunction {
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.RequestsFunction({ method, url, headers }), id);
    }

    // addNewCeleryRequest(
    //     url: string,
    //     method: string = "GET",
    //     headers: Record<string, any> = {},
    //     taskStatusUrl: string = "http://127.0.0.1:8000/tasks/status/{task_id}",
    //     id?: string
    // ): Model4LLMs.AsyncCeleryWebApiFunction {
    //     return this.add_new_obj(
    //         new this.MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction({ method, url, headers, task_status_url: taskStatusUrl }),
    //         id
    //     );
    // }

    addNewWorkFlow(tasks: Record<string, string[]> | string[], metadata: Record<string, any> = {}, id?: string): Model4LLMs.WorkFlow {
        if (Array.isArray(tasks)) {
            const tasks_list = tasks as string[];
            tasks_list.reverse();
            const dependencies = tasks_list.map((_, i) => (tasks_list[i + 1] ? [tasks_list[i + 1]] : []));
            tasks = tasks_list.reduce((acc: Record<string, string[]>, task, index) => {
                acc[task] = dependencies[index] || [];
                return acc;
            }, {});
        }
        return this.add_new_obj(new this.MODEL_CLASS_GROUP.WorkFlow({ tasks, metadata }), id);
    }

    findFunction(functionId: string): Model4LLMs.Function {
        return this.find(functionId) as Model4LLMs.Function;
    }

    findAllVendors(): Model4LLMs.AbstractVendor[] {
        return this.find_all("*Vendor:*") as Model4LLMs.AbstractVendor[];
    }

    static chainDumps(cl: any[]): string {
        const acc: Record<string, any> = {};
        for (let index = 0; index < cl.length; index++) {
            const obj = cl[index];
            acc[obj.get_id()] = obj.model_dump_json_dict();
        }
        return JSON.stringify(acc);
    }

    chainLoads(clJson: string): Model4Basic.AbstractObj[] {
        const data: Record<string, any> = JSON.parse(clJson);
        const tmpStore = new LLMsStore();
        tmpStore.loads(clJson);
        const objs = Object.keys(data).map((key) => tmpStore.find(key));
        objs.forEach((obj) => {if(obj)obj._id = null;});
        return objs.map((obj) => this.add_new_obj(obj));
    }
}
