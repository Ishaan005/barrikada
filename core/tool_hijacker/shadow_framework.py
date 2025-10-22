"""
Shadow Framework for ToolHijacker.
Implements the shadow tool selection pipeline including shadow LLM, retriever, and tool library.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
from .tool_document import ToolDocument


@dataclass
class ShadowTask:
    """Represents a shadow task description (q' in Q')"""
    query: str
    task_type: str
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ShadowRetriever:
    """
    Shadow retriever (f'(·)) that simulates the retrieval step.
    Uses similarity scoring to retrieve top-k documents.
    """
    
    def __init__(self, similarity_function: Optional[Callable] = None):
        """
        Args:
            similarity_function: Custom similarity function. 
                                Defaults to cosine similarity on embeddings.
        """
        self.similarity_function = similarity_function or self._default_similarity
        self.embedding_cache = {}
    
    def _default_similarity(self, query: str, document: str) -> float:
        """
        Default similarity using simple embedding-based cosine similarity.
        In practice, this would use a proper embedding model.
        """
        # Simple bag-of-words similarity as placeholder
        # In real implementation, use sentence transformers or similar
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        intersection = query_tokens.intersection(doc_tokens)
        union = query_tokens.union(doc_tokens)
        
        return len(intersection) / len(union) if union else 0.0
    
    def retrieve(
        self, 
        query: str, 
        tool_documents: List[ToolDocument], 
        top_k: int = 5
    ) -> List[ToolDocument]:
        """
        Retrieve top-k most similar tool documents for a given query.
        
        Args:
            query: The task description/query
            tool_documents: List of available tool documents
            top_k: Number of documents to retrieve
            
        Returns:
            List of top-k retrieved tool documents
        """
        scores = []
        for doc in tool_documents:
            # Combine name and description for similarity calculation
            doc_text = f"{doc.name} {doc.description}"
            score = self.similarity_function(query, doc_text)
            scores.append((score, doc))
        
        # Sort by score (descending) and return top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]
    
    def compute_gradient(
        self, 
        query: str, 
        document_text: str
    ) -> Dict[str, Any]:
        """
        Compute gradient information for gradient-based optimization.
        This is a simplified version - real implementation would use autograd.
        
        Returns:
            Dictionary containing gradient information
        """
        # Placeholder for gradient computation
        # In real implementation, this would compute actual gradients
        # through the embedding model
        return {
            'similarity_score': self.similarity_function(query, document_text),
            'gradient': None  # Would contain actual gradient tensor
        }


class ShadowLLM:
    """
    Shadow LLM (E') that simulates the tool selection step.
    Selects a tool from the retrieved set based on the task description.
    """
    
    def __init__(self, model_name: str = "shadow_llm"):
        """
        Args:
            model_name: Identifier for the shadow LLM
        """
        self.model_name = model_name
    
    def select_tool(
        self,
        task_description: str,
        retrieved_tools: List[ToolDocument],
        return_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Select the most appropriate tool from the retrieved set.
        
        Args:
            task_description: The user's task description
            retrieved_tools: List of retrieved tool documents
            return_reasoning: Whether to return reasoning for the selection
            
        Returns:
            Dictionary containing selected tool and optional reasoning
        """
        if not retrieved_tools:
            return {
                'selected_tool': None,
                'tool_name': None,
                'confidence': 0.0,
                'reasoning': "No tools available"
            }
        
        # Simple selection based on keyword matching
        # In real implementation, this would use an actual LLM
        best_tool = None
        best_score = -1
        
        task_tokens = set(task_description.lower().split())
        
        for tool in retrieved_tools:
            tool_text = f"{tool.name} {tool.description}".lower()
            tool_tokens = set(tool_text.split())
            
            # Simple scoring based on overlap
            score = len(task_tokens.intersection(tool_tokens))
            
            if score > best_score:
                best_score = score
                best_tool = tool
        
        return {
            'selected_tool': best_tool,
            'tool_name': best_tool.name if best_tool else None,
            'confidence': min(best_score / max(len(task_tokens), 1), 1.0),
            'reasoning': f"Selected based on keyword overlap score: {best_score}" if return_reasoning else None
        }
    
    def compute_selection_probability(
        self,
        task_description: str,
        retrieved_tools: List[ToolDocument],
        target_tool_name: str
    ) -> float:
        """
        Compute the probability that the LLM selects the target tool.
        Used for gradient-based optimization.
        
        Args:
            task_description: The user's task description
            retrieved_tools: List of retrieved tool documents
            target_tool_name: Name of the target (malicious) tool
            
        Returns:
            Probability of selecting the target tool
        """
        selection_result = self.select_tool(task_description, retrieved_tools)
        selected_name = selection_result['tool_name']
        
        if selected_name == target_tool_name:
            return 1.0
        else:
            # Return partial probability based on confidence
            return selection_result['confidence'] * 0.1  # Small probability if not selected
    
    def generate_output(
        self,
        task_description: str,
        selected_tool: ToolDocument
    ) -> str:
        """
        Generate output containing the selected tool name.
        This simulates the LLM's response (o_t in the paper).
        
        Args:
            task_description: The user's task description
            selected_tool: The selected tool document
            
        Returns:
            Generated output string containing the tool name
        """
        return f"I will use {selected_tool.name} to complete this task."


class ShadowFramework:
    """
    Complete shadow framework that combines retriever, LLM, and tool library.
    """
    
    def __init__(
        self,
        retriever: Optional[ShadowRetriever] = None,
        llm: Optional[ShadowLLM] = None
    ):
        """
        Args:
            retriever: Shadow retriever instance
            llm: Shadow LLM instance
        """
        self.retriever = retriever or ShadowRetriever()
        self.llm = llm or ShadowLLM()
        self.shadow_tools: List[ToolDocument] = []
    
    def add_shadow_tool(self, tool: ToolDocument):
        """Add a tool to the shadow tool library"""
        self.shadow_tools.append(tool)
    
    def set_shadow_tools(self, tools: List[ToolDocument]):
        """Set the complete shadow tool library"""
        self.shadow_tools = tools
    
    def execute_pipeline(
        self,
        task_description: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute the complete shadow tool selection pipeline.
        
        Args:
            task_description: The user's task description
            top_k: Number of tools to retrieve
            
        Returns:
            Dictionary containing pipeline results
        """
        # Step 1: Retrieval
        retrieved_tools = self.retriever.retrieve(
            task_description,
            self.shadow_tools,
            top_k
        )
        
        # Step 2: Selection
        selection_result = self.llm.select_tool(
            task_description,
            retrieved_tools,
            return_reasoning=True
        )
        
        return {
            'task_description': task_description,
            'retrieved_tools': retrieved_tools,
            'selected_tool': selection_result['selected_tool'],
            'selected_tool_name': selection_result['tool_name'],
            'confidence': selection_result['confidence'],
            'reasoning': selection_result['reasoning']
        }
    
    def evaluate_attack_success(
        self,
        task_descriptions: List[str],
        malicious_tool_name: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate how often the malicious tool is selected across task descriptions.
        
        Args:
            task_descriptions: List of task descriptions to test
            malicious_tool_name: Name of the malicious tool
            top_k: Number of tools to retrieve
            
        Returns:
            Dictionary with success metrics
        """
        total_tasks = len(task_descriptions)
        retrieval_success = 0
        selection_success = 0
        
        for task in task_descriptions:
            result = self.execute_pipeline(task, top_k)
            
            # Check if malicious tool was retrieved
            retrieved_names = [t.name for t in result['retrieved_tools']]
            if malicious_tool_name in retrieved_names:
                retrieval_success += 1
            
            # Check if malicious tool was selected
            if result['selected_tool_name'] == malicious_tool_name:
                selection_success += 1
        
        return {
            'total_tasks': total_tasks,
            'retrieval_success_count': retrieval_success,
            'retrieval_success_rate': retrieval_success / total_tasks if total_tasks > 0 else 0,
            'selection_success_count': selection_success,
            'selection_success_rate': selection_success / total_tasks if total_tasks > 0 else 0,
            'overall_success_rate': selection_success / total_tasks if total_tasks > 0 else 0
        }
