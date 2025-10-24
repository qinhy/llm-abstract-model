import copy
import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
from LLMAbstractModel.raptor_utils import split_text

store = LLMsStore()

vendor = store.add_new(Model4LLMs.OpenAIVendor)(
                            api_key='OPENAI_API_KEY')
text_embedding = store.add_new(Model4LLMs.TextEmbedding3Small)(
                            vendor_id=vendor.get_id())
llm = store.add_new_obj(Model4LLMs.ChatGPTDynamic(
            system_prompt="""
Your task is to read the text provide me summarization.
You should reply summarization only, without any additional information.""",
            llm_model_name='gpt-5-nano',
            vendor_id=vendor.get_id()))

docs = [
    "Basic Info: Name - Alex Johnson, Birthday - 1995-08-15, Phone - +1-555-1234, Email - alex.johnson@email.com, Address - 123 Maple Street, Springfield",
    "Personal Details: Occupation - Software Developer, Hobbies - reading, hiking, coding, photography",
    "Friends: Taylor Smith (Birthday: 1994-02-20, Phone: +1-555-5678), Jordan Lee (Birthday: 1993-11-30, Phone: +1-555-9101), Morgan Brown (Birthday: 1996-05-25, Phone: +1-555-1213)",
    "Work & Goals: Company - Tech Solutions Inc., Position - Front-End Developer, Work Email - alex.j@techsolutions.com, Work Phone - +1-555-4321, Goals - Learn a new programming language, Complete a marathon, Read 20 books this year"
]


raptor_tree = Model4LLMs.RaptorClusterTree(                
                summarization_model=llm,
                embedding_models={"OpenAI":text_embedding}
                )

chunks = split_text('\n'.join(docs), raptor_tree.tokenizer(), 20)

print("Creating Leaf Model4LLMs.RaptorNode")

leaf_nodes = {}
for index, chunk in enumerate(chunks):
    __, node = raptor_tree.create_node(index, chunk)
    leaf_nodes[index] = node

layer_to_nodes = {0: list(leaf_nodes.values())}
print(f"Created {len(leaf_nodes)} Leaf Embeddings")

print("Building All Model4LLMs.RaptorNode")
all_nodes = copy.deepcopy(leaf_nodes)
root_nodes = raptor_tree.construct_tree(all_nodes, all_nodes, layer_to_nodes)
raptor_tree = Model4LLMs.RaptorClusterTree(
    all_nodes=all_nodes,
    root_nodes=root_nodes,
    leaf_nodes=leaf_nodes,
    num_layers=raptor_tree.num_layers,
    layer_to_nodes=layer_to_nodes,
              
    summarization_model=llm,
    embedding_models={"OpenAI":text_embedding})

raptor_tree.retrieve("Who is Jordan Lee?")
# raptor_tree = Model4LLMs.RaptorClusterTree(
#                 summarization_model=llm,
#                 embedding_models={"OpenAI":text_embedding}
#                 ).build_from_text(text='\n'.join(docs))


