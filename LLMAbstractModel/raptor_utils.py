
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import logging
import pickle
import random
import re
import time
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np
from pydantic import BaseModel, Field
import tiktoken
from umap.umap_ import UMAP
from scipy import spatial
from sklearn.mixture import GaussianMixture

# from .LLMsModel import Model4LLMs
# Node = Model4LLMs.RaptorNode

def retry_with_exponential_backoff(
    retries: int = 6,
    min_wait: float = 1.0,
    max_wait: float = 20.0,
    multiplier: float = 2.0,
) -> Callable:
    """
    Decorator implementing a simple exponential backoff retry strategy.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = max(min_wait, 0.0)
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == retries - 1:
                        raise
                    upper = min(max(wait, min_wait), max_wait)
                    lower = min(min_wait, upper)
                    time.sleep(random.uniform(lower, upper))
                    wait = min(max(upper * multiplier, min_wait), max_wait)

        return wrapper

    return decorator

def reverse_mapping(layer_to_nodes: Dict[int, List["Node"]]) -> Dict[int, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens: int=150, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    """
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        if not sentence.strip():
            continue

        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            filtered_sub_sentences = [
                sub.strip() for sub in sub_sentences if sub.strip() != ""
            ]
            sub_token_counts = [
                len(tokenizer.encode(" " + sub_sentence))
                for sub_sentence in filtered_sub_sentences
            ]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(
                filtered_sub_sentences, sub_token_counts
            ):
                if sub_length + sub_token_count > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                    if overlap > 0:
                        sub_length = sum(sub_token_counts[-overlap:])
                    else:
                        sub_length = 0

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            if overlap > 0 and current_chunk:
                current_length = sum(
                    len(tokenizer.encode(" " + sentence)) for sentence in current_chunk
                )
            else:
                current_length = 0
            current_chunk.append(sentence)
            current_length += token_count

        else:
            current_chunk.append(sentence)
            current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    return [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]


def get_node_list(node_dict: Dict[int, "Node"]) -> List["Node"]:
    indices = sorted(node_dict.keys())
    return [node_dict[index] for index in indices]


def get_embeddings(node_list: List["Node"], embedding_model: str) -> List:
    return [node.embeddings[embedding_model] for node in node_list]

def get_children(node_list: List["Node"]) -> List[Set[int]]:
    return [node.children for node in node_list]


def get_text(node_list: List["Node"]) -> str:
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text().splitlines())}"
        text += "\n\n"
    return text

def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    return np.argsort(distances)


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 224
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

class RaptorNode(BaseModel):
    """
    Storage-native RAPTOR node.
    - node_type: 'root' | 'leaf' | 'summary'
    - text      : original content (leaf)
    - summary   : abstractive roll-up (summary/root)
    - embeddings: Dict[vendor_name, vector]
    - tokens    : cached token count (optional)
    """
    content: "Model4LLMs.TextContent"
    index: int
    node_type: str = "leaf" # 'root' | 'leaf' | 'summary'
    embeddings: Dict[str,"Model4LLMs.EmbeddingContent"]
    children: Set[int] = Field(default_factory=set)
    tokens: int = 0
    cluster_id: int = -1

    # convenience
    def text(self):
        return self.content.get_text()

    def summary(self):
        if self.node_type != "summary":return None
        return self.content.get_data().rLOD2
    
    def is_leaf(self) -> bool:
        return self.node_type == "leaf" and len(self.children_id) == 0

    def is_summary(self) -> bool:
        return self.node_type in ("summary", "root")

    def content_text(self) -> str:
        return self.text() if self.text() else self.summary()
    
class RaptorClusterTree(BaseModel):
    all_nodes: Dict[int, "Model4LLMs.RaptorNode"] = {}
    root_nodes: Dict[int, "Model4LLMs.RaptorNode"] = {}
    leaf_nodes: Dict[int, "Model4LLMs.RaptorNode"] = {}
    layer_to_nodes: Dict[int, List["Model4LLMs.RaptorNode"]] = {}

    tokenizer_name: str = "cl100k_base"
    max_tokens: int = 100
    start_layer: int = 0
    num_layers: int = 5
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: str = "top_k"
    summarization_length: int = 100
    summarization_model: "AbstractLLM"
    embedding_models: Dict[str, "Model4LLMs.AbstractEmbedding"]
    cluster_embedding_model: str = "OpenAI"
    
    # Cluster
    reduction_dimension: int = 10
    clustering_algorithm:str = "RAPTOR_Clustering"
    max_length_in_cluster: int = 3500
    threshold: float = 0.1
    verbose: bool = False
    tree_node_index_to_layer:Dict[int,int] = {}
    _summary_cache = {}
    _embedding_cache = {}

    def tokenizer(self):
        return tiktoken.get_encoding(self.tokenizer_name)
    def token_len(self, text: str):
        return len(self.tokenizer().encode(text))
    
    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None):
        
        if children_indices is None:
            children_indices = set()

        con = Model4LLMs.TextContent(text=text)
        con._id = con.gen_new_id()
        embeddings = {
            model_name: Model4LLMs.EmbeddingContent(
                target_id=con.get_id(),
                vec=model.generate_embedding(text).tolist())
            for model_name, model in self.embedding_models.items()
        }
        return (index, Model4LLMs.RaptorNode(                
                    content=Model4LLMs.TextContent(text=text),
                    index=index,
                    embeddings=embeddings,
                    children = children_indices))

    def create_embedding(self, text) -> List[float]:
        self._embedding_cache[text] = self._embedding_cache.get(text,
                        self.embedding_models[self.cluster_embedding_model].generate_embedding(
                            text).tolist())
        return self._embedding_cache[text]

    def summarize(self, context, max_tokens=150) -> str:
        return self.summarization_model(context, max_tokens)

    def get_relevant_nodes(self, current_node:"Model4LLMs.RaptorNode", list_nodes:List["Model4LLMs.RaptorNode"]
                            ) -> List["Model4LLMs.RaptorNode"]:
        embeddings = [node.embeddings[self.cluster_embedding_model] for node in list_nodes]

        distances = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]
        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]
        else:
            best_indices = []

        return [list_nodes[idx] for idx in best_indices]

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, "Model4LLMs.RaptorNode"]:
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> "Model4LLMs.RaptorTree":
        chunks = split_text(text, self.tokenizer(), self.max_tokens)

        print("Creating Leaf Model4LLMs.RaptorNode")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, chunk in enumerate(chunks):
                __, node = self.create_node(index, chunk)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}
        print(f"Created {len(leaf_nodes)} Leaf Embeddings")

        print("Building All Model4LLMs.RaptorNode")
        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
        tree = Model4LLMs.RaptorClusterTree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)
        return tree

    @staticmethod
    def static_perform_RAPTOR_clustering(
        nodes: List["Model4LLMs.RaptorNode"],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List["Model4LLMs.RaptorNode"]]:
        embeddings = np.array([node.embeddings[embedding_model_name].get_vec() for node in nodes])

        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        node_clusters = []

        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(
                [len(tokenizer.encode(node.text())) for node in cluster_nodes]
            )

            if total_length > max_length_in_cluster:
                if verbose:
                    print(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    Model4LLMs.RaptorClusterTree.static_perform_RAPTOR_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster,
                        tokenizer, reduction_dimension, threshold, verbose
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters

    def construct_tree(
        self,
        current_level_nodes: Dict[int, "Model4LLMs.RaptorNode"],
        all_tree_nodes: Dict[int, "Model4LLMs.RaptorNode"],
        layer_to_nodes: Dict[int, List["Model4LLMs.RaptorNode"]],
        use_multithreading: bool = True,
    ) -> Dict[int, "Model4LLMs.RaptorNode"]:
        print("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster:List[Model4LLMs.RaptorNode], new_level_nodes, next_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            child_ids = tuple(sorted(n.index for n in cluster))
            if child_ids in self._summary_cache:
                summarized_text = self._summary_cache[child_ids]
            else:
                summarized_text = self.summarize(context=node_texts, max_tokens=summarization_length)
                self._summary_cache[child_ids] = summarized_text

            print(
                f"Node Texts Length: {self.token_len(node_texts)}, "
                f"Summarized Text Length: {self.token_len(summarized_text)}"
            )

            __, new_parent_node = self.create_node(
                next_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_index] = new_parent_node

        for layer in range(self.num_layers):
            new_level_nodes = {}

            print(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                print(
                    "Stopping Layer construction: Cannot Create More Layers. "
                    f"Total Layers in tree: {layer}"
                )
                break

            clusters = Model4LLMs.RaptorClusterTree.static_perform_RAPTOR_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                self.max_length_in_cluster,
                tiktoken.get_encoding(self.tokenizer_name),
                self.reduction_dimension,
                self.threshold,
                self.verbose,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            print(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)
            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return current_level_nodes

    
    def retrieve_information_collapse_tree(
        self, query: str, top_k: int, max_tokens: int
    ) -> Tuple[List["Model4LLMs.RaptorNode"], str]:
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        indices:List[int] = sorted(self.all_nodes.keys())
        node_list = [self.all_nodes[index] for index in indices]
        embeddings = [node.embeddings[self.cluster_embedding_model].get_vec() for node in node_list]

        distances = distances_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:
            node = node_list[idx]
            node_tokens = self.token_len(node.text())

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List["Model4LLMs.RaptorNode"], query: str, num_layers: int
    ) -> Tuple[List["Model4LLMs.RaptorNode"], str]:
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.cluster_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)

            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices:List[int] = [
                    index for index in indices if distances[index] > self.threshold
                ]
            elif self.selection_mode == "top_k":
                best_indices:List[int] = indices[: self.top_k].tolist()
            else:
                best_indices:List[int] = []

            nodes_to_add = [node_list[idx] for idx in best_indices]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []

                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.num_layers
        ):
            raise ValueError(
                f"start_layer of [{self.start_layer}] must be an integer between 0 and {self.num_layers}"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError(f"num_layers must be less than or equal to {self.start_layer+1}")

        if collapse_tree:
            print("Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_information:
            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )

            return context, layer_information

        return context
