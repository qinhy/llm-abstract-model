import fs from 'fs';
import { Model4LLMs,LLMsStore } from "./LLMsModel";
import { RegxExtractor } from './utils';

function uuidv4(prefix = '') {
    return prefix + 'xxxx-xxxx-xxxx-xxxx-xxxx'.replace(/x/g, function () {
        return Math.floor(Math.random() * 16).toString(16);
    });
}

class TextContentNode {
    id: string;
    content: string;
    embedding: number[];
    parent_id: string;
    children: TextContentNode[];
    depth: number;
    rootNode: TextContentNode | null;

    static loadDict(obj:any) : TextContentNode {
        function traverse(tc: TextContentNode){
            const t = new TextContentNode(tc.content,
                tc.id,tc.embedding,tc.parent_id,tc.children,tc.depth);
            const cs = [];
            for (const child of t.children) {
                cs.push(traverse(child));
            }
            t.children = cs;
            return t;
        }
        const tc = traverse(obj);
        return tc;
    }

    constructor(
        content = "Root(Group)",
        id = null,
        embedding = [],
        parent_id = "NULL",
        children = [],
        depth = 0,
    ) {
        this.id = id || `Node:${uuidv4()}`;
        this.content = content;
        this.embedding = embedding;
        this.parent_id = parent_id;
        this.children = children;
        this.depth = depth;
        this.rootNode = null;
    }

    isGroup(): boolean {
        return this.content.endsWith("(Group)");
    }

    isRoot(): boolean {
        return this.content == "Root(Group)";
    }

    contentWithParents(): string {
        const groups = this.parents().map((g) => g.content);
        let content = this.content;
        if (groups.length > 0) {
            content += ` [${groups.join(";")}]`;
        }
        return content;
    }

    refreshRoot(): void {
        if (this.isRoot()) {
            for (const child of this.traverse(this)) {
                if (!child.isRoot()) {
                    child.rootNode = this;
                }
            }
        }
    }

    *traverse(node: TextContentNode): Generator<TextContentNode> {
        yield node;
        for (const child of node.children) {
            yield* this.traverse(child);
        }
    }

    getNode(nodeId: string): TextContentNode | null {
        const root = this.isRoot() ? this : this.rootNode;
        if (!root) return null;

        for (const child of root.traverse(root)) {
            if (child.id === nodeId) {
                return child;
            }
        }
        return null;
    }

    dispose(): void {
        const parent = this.getParent();
        if (parent) {
            parent.children = parent.children.filter((child) => child.id !== this.id);
        }
    }

    add(child: TextContentNode): void {
        this.children.push(child);
        child.parent_id = this.id;
        child.depth = this.depth + 1;
    }

    getParent(): TextContentNode | null {
        return this.parent_id ? this.getNode(this.parent_id) : null;
    }

    parents(): TextContentNode[] {
        const path: TextContentNode[] = [];
        let current: TextContentNode | null = this;

        while (current && !current.isRoot()) {
            path.push(current);
            current = current.getParent();
        }

        return path.reverse().slice(0, -1);
    }

    groups(): TextContentNode[] {
        return this.parents().filter((node) => node.isGroup());
    }

    getAllChildren(): TextContentNode[] {
        const descendants: TextContentNode[] = [];

        const collectChildren = (node: TextContentNode) => {
            descendants.push(node);
            for (const child of node.children) {
                collectChildren(child);
            }
        };

        collectChildren(this);
        return descendants;
    }
}

class TextMemoryTree {
    baseThreshold: number = 0.4;
    lambdaFactor: number = 0.5;
    private static embeddingCacheDict: Record<string, any> = {};

    root: TextContentNode;
    llm: Model4LLMs.AbstractLLM | null;
    textEmbedding: Model4LLMs.AbstractEmbedding;

    constructor(root: TextContentNode | null = null, llm: Model4LLMs.AbstractLLM | null = null, textEmbedding: Model4LLMs.AbstractEmbedding | any = null) {
        this.llm = llm;
        this.textEmbedding = textEmbedding;

        if (root && !(root instanceof TextContentNode)) {
            root = new TextContentNode(root);
        }

        this.root = root || new TextContentNode( "Root(Group)" );
        this.root.refreshRoot();

        for (const node of this.traverse(this.root)) {
            TextMemoryTree.embeddingCacheDict[node.content] = node.embedding;
        }
    }

    *traverse(node: TextContentNode): Generator<TextContentNode> {
        yield node;
        for (const child of node.children) {
            yield* this.traverse(child);
        }
    }

    removeContent(content: string): void {
        for (const node of this.findNodesByContent(content)) {
            node.dispose();
        }
    }

    findNodesByContent(keyword: string): TextContentNode[] {
        return [...this.traverse(this.root)].filter((node) => node.content.includes(keyword));
    }

    async classification(src: TextContentNode, ref: TextContentNode): Promise<number> {
        const sys = `
## Instructions:
Given two pieces of content, 'ref' and 'src', identify their relationship and label it with one of the following:
Provide only a single word as your output: Child, Parent, or Isolate.

## Procedure:
1. Classify Each Content:
   - Determine whether 'ref' and 'src' are a topic or an action.
     - *Topic*: A general subject area.
     - *Action*: A specific activity within a topic.

2. Assess Connection:
   - Evaluate whether there is any connection between 'ref' and 'src', considering both direct and indirect links.

3. Apply Inclusion Rule:
   - Remember that a topic always includes its related actions.

4. Compare Topics (if both are topics):
   - Determine if one topic includes the other.

5. Select the Appropriate Label:
   - Choose Child if 'src' potentially includes 'ref'.
   - Choose Parent if 'ref' potentially includes 'src'.
   - Choose Isolate if no connection exists.

## Notes:
- Carefully determine whether each piece of content is a topic or an action.
- Consider subtle connections for accurate labeling.
`;
        if (this.llm) {
            this.llm.system_prompt = sys;
            const res = await this.llm.acall(`## src\n${src.content}\n## ref\n${ref.content}`);
            return { parent: -1, isolate: 0, child: 1 }[res.trim().toLowerCase()] || 0;
        }
        return 0;
    }

    async tidyTree(res: string | null = null): Promise<this> {
        if (!res) {
            const sys = `
Please organize the following memory list and respond in the original tree structure format. 
If a node represents a group, add '(Group)' at the end of the node name. Feel free to edit, delete, 
move or add new nodes (or groups) as needed.`;

            if (this.llm) {
                this.llm.system_prompt = sys;
                res = await this.llm.acall(`\`\`\`text\n${this.printTree(null,0,false)}\n\`\`\``);
                const match = /```text\s*(.*)\s*```/s.exec(res);
                if (match) res = match[1];
            }
        }
        const newRoot = this.parseTextTree(res || "");
        const embeddings: Record<string, any> = {};

        for (const node of this.traverse(this.root)) {
            embeddings[node.contentWithParents()] = node.embedding;
            node.embedding = [];
        }

        for (const node of this.traverse(newRoot)) {
            node.embedding = embeddings[node.contentWithParents()] || [];
        }

        this.root = newRoot;
        return this;
    }

    getThreshold(depth: number): number {
        return this.baseThreshold * Math.exp(this.lambdaFactor * depth);
    }

    async calcEmbedding(node: TextContentNode): Promise<TextContentNode> {
        if (node.embedding.length > 0) return node;

        if (node.isGroup()) {
            node.embedding = Array(this.textEmbedding.embedding_dim).fill(0);
        } else {
            if (node.content in TextMemoryTree.embeddingCacheDict) {
                node.embedding = TextMemoryTree.embeddingCacheDict[node.content];
            }
            if (node.embedding.length === 0) {
                node.embedding = await this.textEmbedding.acall(node.contentWithParents());
            }
        }
        return node;
    }

    async similarity(src: TextContentNode, ref: TextContentNode): Promise<number> {
        const emb1 = (await this.calcEmbedding(ref)).embedding;
        const emb2 = (await this.calcEmbedding(src)).embedding;        

        const norm1 = Math.sqrt(emb1.reduce((sum, x) => sum + x ** 2, 0));
        const norm2 = Math.sqrt(emb2.reduce((sum, x) => sum + x ** 2, 0));

        if (norm1 === 0 || norm2 === 0) return 0;

        const dotProduct = emb1.reduce((sum, x, i) => sum + x * emb2[i], 0);
        return dotProduct / (norm1 * norm2);
    }

    extractAllEmbeddings(): Record<string, any> {
        const embeddings: Record<string, any> = {};
        for (const node of this.traverse(this.root)) {
            embeddings[node.id] = node.embedding;
            node.embedding = [];
        }
        return embeddings;
    }

    dumpAllEmbeddings(path = "embeddings.json"): void {
        const embeddings = this.extractAllEmbeddings();
        fs.writeFileSync(path, JSON.stringify(embeddings));
    }

    putAllEmbeddings(embeddings: Record<string, any>): void {
        for (const node of this.traverse(this.root)) {
            node.embedding = embeddings[node.id] || [];
        }
    }

    loadAllEmbeddings(path = "embeddings.json"): this {
        const embeddings = JSON.parse(fs.readFileSync(path, "utf8"));
        this.putAllEmbeddings(embeddings);
        return this;
    }

    async retrieve(query: string, topK = 3): Promise<[TextContentNode, number][]> {
        const results: [TextContentNode, number][] = [];
        const queryNode = this.calcEmbedding(new TextContentNode(query));

        for (const node of this.traverse(this.root)) {
            if (node.isRoot()) continue;
            const sim = await this.similarity(node, await queryNode);
            results.push([node, sim]);
        }

        return results.sort((a, b) => b[1] - a[1]).slice(0, topK);
    }

    async insert(content: string): Promise<void> {
        const newNode = await this.calcEmbedding(new TextContentNode(content.trim()));
        this._insertNode(this.root, newNode);
        console.log(`[TextMemoryTree]: insert new content [${content.trim()}]`);
        this.root.refreshRoot();
    }

    private async _insertNode(currentNode: TextContentNode, newNode: TextContentNode): Promise<void> {
        let bestSimilarity = -1;
        let bestChild: TextContentNode | null = null;

        for (const child of currentNode.children) {
            const sim = await this.similarity(child, newNode);
            if (sim > bestSimilarity) {
                bestSimilarity = sim;
                bestChild = child;
            }
        }

        if (bestChild && bestSimilarity >= this.getThreshold(currentNode.depth)) {
            this._insertNode(bestChild, newNode);
        } else {
            currentNode.add(newNode);
        }
    }

    parseTextTree(text: string): TextContentNode {
        const lines = text.split("\n").map((line) => line.trimEnd().replace('- ','')).filter(l=>l.length);
        const stack: [TextContentNode, number][] = [];
        const root = new TextContentNode(lines.shift()?.trimEnd() || "");
        
        if (!root.isRoot()) throw new Error("First node must be Root.");

        stack.push([root, -1]);

        for (const line of lines) {
            const stripped = line.trimStart();
            const indent = line.length - stripped.length;
            const content = stripped.trimEnd();
            const level = Math.floor(indent / 4);

            while (stack.length && stack[stack.length - 1][1] >= level) {
                stack.pop();
            }

            const parent = stack[stack.length - 1][0];
            const item = new TextContentNode(content);
            parent.add(item);
            stack.push([item, level]);
        }

        root.refreshRoot();        
        return root;
    }

    printTree(node: TextContentNode | null = null, level = 0, isPrint = true): string {
        let result = "";
        node = node || this.root;

        result += " ".repeat(level * 4) + `- ${node.content.trim()}\n`;
        
        for (const child of node.children) {
            result += this.printTree(child, level + 1, false);
        }

        if (node === this.root && isPrint) console.log(result);
        return result;
    }
}


class PersonalAssistantAgent extends Model4LLMs.AbstractObj {
    llm: Model4LLMs.AbstractLLM | null = null;
    textEmbedding: Model4LLMs.AbstractEmbedding | null = null;
    memoryRoot: TextContentNode = new TextContentNode();
    memoryTopK: number = 5;
    memoryMinScore: number = 0.3;
    systemPrompt: string = `
You are a capable and friendly assistant, combining past knowledge with new insights to provide effective answers.  
Note: When it is necessary to retain new and important information, such as preferences or routines, you may respond using a block of \`\`\`memory ...\`\`\`."`;


    constructor(
        memoryRoot: TextContentNode,
        llm: Model4LLMs.AbstractLLM | null,
        textEmbedding: Model4LLMs.AbstractEmbedding | null,
        memoryTopK: number=5,
    ) {
        super();
        this.memoryRoot = memoryRoot;
        this.llm = llm;
        this.textEmbedding = textEmbedding;
        this.memoryTopK = memoryTopK;
    }
    
    getMemory(): TextMemoryTree {
        return new TextMemoryTree(this.memoryRoot,this.llm,this.textEmbedding);
    }

    async tidyMemory(res: any = null): Promise<void> {
        this.memoryRoot = (await this.getMemory().tidyTree(res)).root;
    }

    printMemory(): void {
        this.getMemory().printTree();
    }

    addMemory(content: string): void {
        this.getMemory().insert(content);
    }

    loadEmbeddings(path: string = 'embeddings.json'): void {
        this.getMemory().loadAllEmbeddings(path);
    }

    async memoryRetrieval(query: string): Promise<string> {
        // Perform retrieval and format the results
        let res = "## Memories for the query:\n";
        const results = await this.getMemory().retrieve(query, this.memoryTopK);

        if (results.length === 0) return res + "No memories.\n";

        results.forEach((v, i) => {
            const [node, score] = v;
            if (score < this.memoryMinScore) return;
            res += `${i + 1}. Score: ${score.toFixed(3)} | Content: ${node.contentWithParents()}\n`;
        });

        return res;
    }

    async acall(query: string, printMemory: boolean = true): Promise<string> {
        const timestamp = new Date().toISOString();
        query = `## User Query\n${query} (${timestamp})\n`;
        const memoExtractor = new RegxExtractor('```memory\s*(.*?)\s*```');
        
        // Retrieve relevant memory and format it for context
        const memory = await this.memoryRetrieval(query);

        // Format the query with memory context
        query = `${memory}\n${query}`;

        if (printMemory) {
            console.log("############ For Debug ##############");
            console.log(query);
            console.log("#####################################\n");
        }

        // Generate a response using the LLM
        if (this.llm) {
            this.llm.system_prompt = this.systemPrompt;
            const response = await this.llm.acall(query);

            if (response.includes('```memory')) {
                const newMemo = memoExtractor.call(response);
                this.addMemory(newMemo);
            }

            return response;
        } else {
            throw new Error("LLM is not initialized.");
        }
    }
}

// Functions for secure data storage and retrieval using RSA key pair
function saveMemoryAgent(store: LLMsStore, rootNode: TextContentNode): void {
    // Save memory tree embeddings and RSA-encrypted data
    const memoryTree = new TextMemoryTree(rootNode);
    for (const child of memoryTree.traverse(rootNode)) {
        child.rootNode = null;
    }
    memoryTree.dumpAllEmbeddings('../tmp/embeddings.json');
    store.set('Memory', rootNode);
    const res = store.dumpRSAs('../tmp/public_key.pem');
    fs.writeFileSync('../tmp/store.rjson', res);
}

function loadMemoryAgent(): [PersonalAssistantAgent, LLMsStore] {    
    const fileContent = fs.readFileSync('../tmp/store.rjson', 'utf-8');
    // Load stored RSA-encrypted data and initialize the agent
    const store = new LLMsStore();
    store.loadRSAs(fileContent, '../tmp/private_key.pem');
    
    // Retrieve saved model instances
    const llm:Model4LLMs.ChatGPT4oMini = store.find_all('ChatGPT4oMini:*')[0];
    const textEmbedding:Model4LLMs.TextEmbedding3Small = store.find_all('TextEmbedding3Small:*')[0];
    const root = TextContentNode.loadDict(store.get('Memory'));
    const tree = new TextMemoryTree(root);
    
    // Reconstruct memory tree from stored data
    const agent = new PersonalAssistantAgent(
        TextContentNode.loadDict(store.get('Memory')),
        llm,textEmbedding,10);
    agent.loadEmbeddings('../tmp/embeddings.json');
    return [agent, store];
}

// Example usage of saving and loading the memory agent
// saveMemoryAgent(store, agent.memoryRoot);
// var [agent, store] = loadMemoryAgent();
// console.log(await agent.acall("Welcome back! What's planned for today?"));

// Main usage form here
// Sample queries reflecting various personal scenarios
const queries = [
    "Basic Info: Name - Alex Johnson, Birthday - 1995-08-15, Phone - +1-555-1234, Email - alex.johnson@email.com, Address - 123 Maple Street, Springfield",
    "Personal Details: Occupation - Software Developer, Hobbies - reading, hiking, coding, photography",
    "Friends: Taylor Smith (Birthday: 1994-02-20, Phone: +1-555-5678), Jordan Lee (Birthday: 1993-11-30, Phone: +1-555-9101), Morgan Brown (Birthday: 1996-05-25, Phone: +1-555-1213)",
    "Work & Goals: Company - Tech Solutions Inc., Position - Front-End Developer, Work Email - alex.j@techsolutions.com, Work Phone - +1-555-4321, Goals - Learn a new programming language, Complete a marathon, Read 20 books this year"
];

// Initialize the LLM Store and vendor
var store = new LLMsStore();
const vendor = store.addNewOpenAIVendor(process.env.OPENAI_API_KEY);

// Add the necessary components
const textEmbedding = store.add_new_obj(new Model4LLMs.TextEmbedding3Small());
const llm = store.addNewChatGPT4oMini('auto');

// Create the root node and initialize the memory tree
const root = new TextContentNode();
const memoryTree = new TextMemoryTree(root,llm,textEmbedding);

// Print the tree structure to visualize the organization
console.log("\n########## Memory Tree Structure:");
for (let index = 0; index < queries.length; index++) {
    const element = queries[index];
    await memoryTree.insert(element);        
}
memoryTree.printTree();

console.log("\n########## Tidied Memory Tree Structure:");
(await memoryTree.tidyTree()).printTree();

// Define a function to test memory retrieval with various sample queries
async function testRetrieval(memoryTree: TextMemoryTree, query: string, topK: number = 5): Promise<void> {
    console.log(`\nTop ${topK} matches for query: '${query}'`);
    const results = memoryTree.retrieve(query, topK);
    (await results).forEach((v, index) => {
        const [node, score] = v;
        console.log(`${index + 1}. score: ${score.toFixed(4)} | [ ${node.content} ]`);
    });
}

// Run retrieval tests
const questions = [
    // Basic Info
    "What is Alex Johnson's full name and birthday?",
    // Personal Details
    "What is Alex's occupation?",
    // Friends
    "Who are Alex's friends, and when are their birthdays?",
    // Work & Goals
    "Where does Alex work and what is their position?"
];

for (let index = 0; index < queries.length; index++) {
    const query = queries[index];
    await testRetrieval(memoryTree, query, 6);
}

// Initialize the personal assistant agent using the memory tree
const agent = new PersonalAssistantAgent(memoryTree.root,llm,textEmbedding);
console.log(await agent.acall("Hi! Please tell me Taylor info."));
