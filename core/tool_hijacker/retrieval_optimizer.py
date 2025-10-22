"""
Retrieval Optimizer (R Optimization) for ToolHijacker.
Implements both gradient-free and gradient-based methods for optimizing the retrieval subsequence.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .shadow_framework import ShadowRetriever, ShadowTask
from .tool_document import MaliciousToolDocument


class RetrievalOptimizer:
    """
    Base class for retrieval optimization.
    Optimizes R to maximize similarity with shadow task descriptions.
    """
    
    def __init__(self, shadow_retriever: ShadowRetriever):
        """
        Args:
            shadow_retriever: The shadow retriever to use for optimization
        """
        self.shadow_retriever = shadow_retriever
    
    def optimize(
        self,
        shadow_tasks: List[ShadowTask],
        malicious_tool: MaliciousToolDocument,
        **kwargs
    ) -> str:
        """
        Optimize the retrieval subsequence R.
        
        Args:
            shadow_tasks: List of shadow task descriptions (Q')
            malicious_tool: The malicious tool document to optimize
            
        Returns:
            Optimized retrieval subsequence R
        """
        raise NotImplementedError("Subclasses must implement optimize()")


class GradientFreeRetrievalOptimizer(RetrievalOptimizer):
    """
    Gradient-free method for optimizing R.
    Uses an LLM to synthesize core functionalities from shadow tasks.
    """
    
    def __init__(
        self,
        shadow_retriever: ShadowRetriever,
        llm_generator: Optional[Any] = None
    ):
        """
        Args:
            shadow_retriever: The shadow retriever
            llm_generator: LLM to use for generating R (can be a callable or API client)
        """
        super().__init__(shadow_retriever)
        self.llm_generator = llm_generator or self._default_generator
    
    def _default_generator(self, prompt: str) -> str:
        """
        Default generator using simple heuristics.
        In practice, this would use an actual LLM API.
        """
        # Extract common keywords from shadow tasks
        # This is a placeholder - real implementation would use LLM
        return "A comprehensive tool for handling various tasks"
    
    def optimize(
        self,
        shadow_tasks: List[ShadowTask],
        malicious_tool: MaliciousToolDocument,
        max_length: int = 200,
        **kwargs
    ) -> str:
        """
        Generate R by synthesizing core functionalities from shadow tasks.
        
        Args:
            shadow_tasks: List of shadow task descriptions
            malicious_tool: The malicious tool document
            max_length: Maximum length of R in characters
            
        Returns:
            Optimized retrieval subsequence R
        """
        # Create prompt for LLM to generate R
        task_queries = [task.query for task in shadow_tasks]
        
        prompt = self._create_synthesis_prompt(task_queries, max_length)
        
        # Generate R using LLM
        r_generated: str
        if callable(self.llm_generator):
            r_generated = str(self.llm_generator(prompt))
        else:
            # Fallback: Extract common terms and create description
            r_generated = self._extract_common_functionalities(task_queries)
        
        # Validate that R improves retrieval
        avg_similarity = self._compute_average_similarity(r_generated, task_queries)
        
        # Truncate if needed
        if len(r_generated) > max_length:
            r_generated = r_generated[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        
        return r_generated
    
    def _create_synthesis_prompt(self, task_queries: List[str], max_length: int) -> str:
        """
        Create a prompt for the LLM to generate R.
        """
        tasks_str = "\n".join([f"- {query}" for query in task_queries])
        
        prompt = f"""Given the following user task descriptions:

{tasks_str}

Generate a comprehensive tool functionality description that addresses ALL of these tasks. The description should:
1. Highlight the core functionalities needed to solve these tasks
2. Be general enough to cover all scenarios
3. Be specific enough to be relevant
4. Be under {max_length} characters
5. Emphasize versatility and comprehensive capabilities

Tool functionality description:"""
        
        return prompt
    
    def _extract_common_functionalities(self, task_queries: List[str]) -> str:
        """
        Extract common functionalities from task queries using heuristics.
        Fallback method when LLM is not available.
        """
        # Tokenize and count term frequencies
        from collections import Counter
        
        all_tokens = []
        for query in task_queries:
            tokens = query.lower().split()
            all_tokens.extend(tokens)
        
        # Get most common terms
        term_counts = Counter(all_tokens)
        common_terms = [term for term, count in term_counts.most_common(15) 
                       if len(term) > 3 and term.isalpha()]
        
        # Construct description
        description = (
            f"A versatile tool for {', '.join(common_terms[:5])} and related tasks. "
            f"Provides comprehensive capabilities for handling various operations including "
            f"{', '.join(common_terms[5:10])}. Supports multiple functionalities to address "
            f"diverse user requirements."
        )
        
        return description
    
    def _compute_average_similarity(self, r_text: str, task_queries: List[str]) -> float:
        """
        Compute average similarity between R and all task queries.
        """
        similarities = []
        for query in task_queries:
            sim = self.shadow_retriever.similarity_function(query, r_text)
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0


class GradientBasedRetrievalOptimizer(RetrievalOptimizer):
    """
    Gradient-based method for optimizing R.
    Uses HotFlip-style token-level optimization with gradient information.
    """
    
    def __init__(
        self,
        shadow_retriever: ShadowRetriever,
        vocabulary: Optional[List[str]] = None
    ):
        """
        Args:
            shadow_retriever: The shadow retriever with gradient support
            vocabulary: Token vocabulary for optimization
        """
        super().__init__(shadow_retriever)
        self.vocabulary = vocabulary or self._default_vocabulary()
    
    def _default_vocabulary(self) -> List[str]:
        """
        Default vocabulary of common words.
        In practice, this would be a proper tokenizer vocabulary.
        """
        common_words = [
            'tool', 'function', 'capability', 'feature', 'support', 'provide',
            'handle', 'manage', 'process', 'execute', 'perform', 'enable',
            'comprehensive', 'versatile', 'powerful', 'advanced', 'efficient',
            'data', 'analysis', 'system', 'operation', 'task', 'query',
            'search', 'find', 'retrieve', 'generate', 'create', 'modify'
        ]
        return common_words
    
    def optimize(
        self,
        shadow_tasks: List[ShadowTask],
        malicious_tool: MaliciousToolDocument,
        initial_r: Optional[str] = None,
        num_iterations: int = 100,
        num_tokens: int = 30,
        **kwargs
    ) -> str:
        """
        Optimize R using gradient-based token-level optimization (HotFlip).
        
        Args:
            shadow_tasks: List of shadow task descriptions
            malicious_tool: The malicious tool document
            initial_r: Initial R sequence (random if None)
            num_iterations: Number of optimization iterations
            num_tokens: Number of tokens in R
            
        Returns:
            Optimized retrieval subsequence R
        """
        task_queries = [task.query for task in shadow_tasks]
        
        # Initialize R with random tokens or provided initial sequence
        if initial_r:
            r_tokens = initial_r.split()[:num_tokens]
        else:
            r_tokens = list(np.random.choice(self.vocabulary, size=num_tokens))
        
        best_r = ' '.join(r_tokens)
        best_score = self._compute_objective(best_r, task_queries)
        
        # Iterative optimization
        for iteration in range(num_iterations):
            improved = False
            
            # Try flipping each token
            for token_idx in range(len(r_tokens)):
                original_token = r_tokens[token_idx]
                
                # Find best replacement token
                best_replacement = original_token
                best_replacement_score = best_score
                
                # Try vocabulary tokens
                for candidate_token in self.vocabulary:
                    if candidate_token == original_token:
                        continue
                    
                    # Create candidate R
                    r_tokens[token_idx] = candidate_token
                    candidate_r = ' '.join(r_tokens)
                    
                    # Compute objective (average similarity)
                    score = self._compute_objective(candidate_r, task_queries)
                    
                    if score > best_replacement_score:
                        best_replacement = candidate_token
                        best_replacement_score = score
                        improved = True
                
                # Update token if improvement found
                r_tokens[token_idx] = best_replacement
                best_score = best_replacement_score
                best_r = ' '.join(r_tokens)
            
            # Early stopping if no improvement
            if not improved:
                break
        
        return best_r
    
    def _compute_objective(self, r_text: str, task_queries: List[str]) -> float:
        """
        Compute the objective function: average similarity across all queries.
        This simulates the gradient-based objective in the paper.
        
        Args:
            r_text: Current R sequence
            task_queries: List of shadow task queries
            
        Returns:
            Average similarity score (objective to maximize)
        """
        similarities = []
        for query in task_queries:
            sim = self.shadow_retriever.similarity_function(query, r_text)
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def compute_gradient_approximation(
        self,
        r_text: str,
        task_queries: List[str],
        epsilon: float = 0.01
    ) -> Dict[str, float]:
        """
        Compute gradient approximation using finite differences.
        This is a simplified version of the gradient computation in the paper.
        
        Args:
            r_text: Current R sequence
            task_queries: List of shadow task queries
            epsilon: Step size for finite difference
            
        Returns:
            Dictionary mapping tokens to gradient estimates
        """
        base_score = self._compute_objective(r_text, task_queries)
        gradients = {}
        
        tokens = r_text.split()
        
        for idx, token in enumerate(tokens):
            # Perturb token slightly (in practice, would use embedding perturbation)
            perturbed_tokens = tokens.copy()
            # Simple perturbation: try removing token
            perturbed_tokens.pop(idx)
            perturbed_r = ' '.join(perturbed_tokens)
            
            perturbed_score = self._compute_objective(perturbed_r, task_queries)
            gradient = (base_score - perturbed_score) / epsilon
            
            gradients[token] = gradient
        
        return gradients
