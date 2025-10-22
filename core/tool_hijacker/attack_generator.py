"""
ToolHijacker: Main Attack Generator
Complete implementation of the ToolHijacker attack framework from the paper.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass, asdict

from .tool_document import ToolDocument, MaliciousToolDocument
from .shadow_framework import (
    ShadowFramework,
    ShadowRetriever,
    ShadowLLM,
    ShadowTask
)
from .task_generator import ShadowTaskGenerator
from .tool_library import ShadowToolLibrary
from .retrieval_optimizer import (
    RetrievalOptimizer,
    GradientFreeRetrievalOptimizer,
    GradientBasedRetrievalOptimizer
)
from .selection_optimizer import (
    SelectionOptimizer,
    GradientFreeSelectionOptimizer,
    GradientBasedSelectionOptimizer
)


@dataclass
class AttackResult:
    """
    Complete result from a ToolHijacker attack generation.
    """
    malicious_tool: MaliciousToolDocument
    retrieval_subsequence: str  # R
    selection_subsequence: str  # S
    final_description: str  # R ⊕ S
    
    # Performance metrics
    retrieval_success_rate: float
    selection_success_rate: float
    overall_success_rate: float
    
    # Metadata
    optimization_method: str
    shadow_tasks_count: int
    shadow_tools_count: int
    generation_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['malicious_tool'] = self.malicious_tool.to_dict()
        return result


class ToolHijacker:
    """
    Main ToolHijacker attack generator.
    
    Implements the complete attack framework including:
    - Shadow framework construction
    - Two-phase optimization (retrieval and selection)
    - Both gradient-free and gradient-based methods
    """
    
    def __init__(
        self,
        shadow_retriever: Optional[ShadowRetriever] = None,
        shadow_llm: Optional[ShadowLLM] = None,
        task_generator: Optional[ShadowTaskGenerator] = None,
        tool_library: Optional[ShadowToolLibrary] = None
    ):
        """
        Initialize ToolHijacker with shadow components.
        
        Args:
            shadow_retriever: Shadow retriever (f')
            shadow_llm: Shadow LLM (E')
            task_generator: Generator for shadow tasks (Q')
            tool_library: Shadow tool library (D')
        """
        # Initialize shadow framework
        self.shadow_retriever = shadow_retriever or ShadowRetriever()
        self.shadow_llm = shadow_llm or ShadowLLM()
        self.shadow_framework = ShadowFramework(
            self.shadow_retriever,
            self.shadow_llm
        )
        
        # Initialize utilities
        self.task_generator = task_generator or ShadowTaskGenerator()
        self.tool_library = tool_library or ShadowToolLibrary()
        
        # Will be set during attack generation
        self.shadow_tasks: List[ShadowTask] = []
        self.shadow_tools: List[ToolDocument] = []
    
    def generate_attack(
        self,
        target_task: str,
        malicious_tool_name: str,
        optimization_method: str = "gradient_free",
        num_shadow_tasks: int = 10,
        num_shadow_tools_relevant: int = 10,
        num_shadow_tools_irrelevant: int = 20,
        top_k_retrieval: int = 5,
        **optimizer_kwargs
    ) -> AttackResult:
        """
        Generate a complete ToolHijacker attack.
        
        Args:
            target_task: The target task description
            malicious_tool_name: Name for the malicious tool
            optimization_method: "gradient_free", "gradient_based", or "hybrid"
            num_shadow_tasks: Number of shadow task descriptions to generate
            num_shadow_tools_relevant: Number of relevant shadow tools
            num_shadow_tools_irrelevant: Number of irrelevant shadow tools
            top_k_retrieval: Number of tools to retrieve (k')
            **optimizer_kwargs: Additional arguments for optimizers
            
        Returns:
            AttackResult containing the malicious tool and metrics
        """
        start_time = time.time()
        
        # Step 1: Construct shadow framework
        print(f"[ToolHijacker] Constructing shadow framework...")
        self._construct_shadow_framework(
            target_task,
            num_shadow_tasks,
            num_shadow_tools_relevant,
            num_shadow_tools_irrelevant
        )
        
        # Step 2: Initialize malicious tool
        malicious_tool = MaliciousToolDocument(
            name=malicious_tool_name,
            description="",  # Will be constructed from R ⊕ S
        )
        
        # Add malicious tool to shadow library
        self.tool_library.add_tool(malicious_tool)
        self.shadow_framework.set_shadow_tools(self.tool_library.get_tools())
        
        # Step 3: Phase 1 - Optimize R (Retrieval)
        print(f"[ToolHijacker] Phase 1: Optimizing retrieval subsequence (R)...")
        r_optimized = self._optimize_retrieval(
            malicious_tool,
            optimization_method,
            **optimizer_kwargs
        )
        malicious_tool.set_retrieval_subsequence(r_optimized)
        print(f"[ToolHijacker] R optimized: {r_optimized[:100]}...")
        
        # Step 4: Phase 2 - Optimize S (Selection)
        print(f"[ToolHijacker] Phase 2: Optimizing selection subsequence (S)...")
        
        # Get typical retrieved set for S optimization
        sample_retrieved = self.shadow_retriever.retrieve(
            self.shadow_tasks[0].query,
            self.shadow_framework.shadow_tools,
            top_k_retrieval
        )
        
        s_optimized = self._optimize_selection(
            malicious_tool,
            sample_retrieved,
            optimization_method,
            **optimizer_kwargs
        )
        malicious_tool.set_selection_subsequence(s_optimized)
        print(f"[ToolHijacker] S optimized: {s_optimized[:100]}...")
        
        # Step 5: Evaluate attack success
        print(f"[ToolHijacker] Evaluating attack success...")
        metrics = self.shadow_framework.evaluate_attack_success(
            [task.query for task in self.shadow_tasks],
            malicious_tool_name,
            top_k_retrieval
        )
        
        generation_time = time.time() - start_time
        
        # Create result
        result = AttackResult(
            malicious_tool=malicious_tool,
            retrieval_subsequence=r_optimized,
            selection_subsequence=s_optimized,
            final_description=malicious_tool.description,
            retrieval_success_rate=metrics['retrieval_success_rate'],
            selection_success_rate=metrics['selection_success_rate'],
            overall_success_rate=metrics['overall_success_rate'],
            optimization_method=optimization_method,
            shadow_tasks_count=len(self.shadow_tasks),
            shadow_tools_count=len(self.shadow_tools),
            generation_time_seconds=generation_time
        )
        
        print(f"[ToolHijacker] Attack generation complete!")
        print(f"  - Retrieval success rate: {metrics['retrieval_success_rate']:.2%}")
        print(f"  - Selection success rate: {metrics['selection_success_rate']:.2%}")
        print(f"  - Overall success rate: {metrics['overall_success_rate']:.2%}")
        print(f"  - Generation time: {generation_time:.2f}s")
        
        return result
    
    def _construct_shadow_framework(
        self,
        target_task: str,
        num_shadow_tasks: int,
        num_relevant_tools: int,
        num_irrelevant_tools: int
    ):
        """
        Construct the shadow framework (Q' and D').
        """
        # Generate shadow tasks (Q')
        self.shadow_tasks = self.task_generator.generate(
            target_task,
            num_variations=num_shadow_tasks
        )
        
        # Build shadow tool library (D')
        self.shadow_tools = self.tool_library.build_default_library(
            num_relevant=num_relevant_tools,
            num_irrelevant=num_irrelevant_tools
        )
        
        # Set shadow tools in framework
        self.shadow_framework.set_shadow_tools(self.shadow_tools)
    
    def _optimize_retrieval(
        self,
        malicious_tool: MaliciousToolDocument,
        method: str,
        **kwargs
    ) -> str:
        """
        Optimize the retrieval subsequence R.
        
        Args:
            malicious_tool: The malicious tool document
            method: Optimization method
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimized R subsequence
        """
        if method == "gradient_free":
            optimizer = GradientFreeRetrievalOptimizer(
                self.shadow_retriever,
                llm_generator=kwargs.get('llm_generator')
            )
        elif method == "gradient_based":
            optimizer = GradientBasedRetrievalOptimizer(
                self.shadow_retriever,
                vocabulary=kwargs.get('vocabulary')
            )
        elif method == "hybrid":
            # Use gradient-free first, then refine with gradient-based
            optimizer_gf = GradientFreeRetrievalOptimizer(
                self.shadow_retriever,
                llm_generator=kwargs.get('llm_generator')
            )
            r_initial = optimizer_gf.optimize(
                self.shadow_tasks,
                malicious_tool,
                **kwargs
            )
            
            optimizer_gb = GradientBasedRetrievalOptimizer(
                self.shadow_retriever,
                vocabulary=kwargs.get('vocabulary')
            )
            return optimizer_gb.optimize(
                self.shadow_tasks,
                malicious_tool,
                initial_r=r_initial,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return optimizer.optimize(
            self.shadow_tasks,
            malicious_tool,
            **kwargs
        )
    
    def _optimize_selection(
        self,
        malicious_tool: MaliciousToolDocument,
        retrieved_tools: List[ToolDocument],
        method: str,
        **kwargs
    ) -> str:
        """
        Optimize the selection subsequence S.
        
        Args:
            malicious_tool: The malicious tool document
            retrieved_tools: Shadow tools in retrieved set
            method: Optimization method
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimized S subsequence
        """
        if method == "gradient_free":
            optimizer = GradientFreeSelectionOptimizer(
                self.shadow_llm,
                attacker_llm=kwargs.get('attacker_llm')
            )
        elif method == "gradient_based":
            optimizer = GradientBasedSelectionOptimizer(
                self.shadow_llm,
                vocabulary=kwargs.get('vocabulary')
            )
        elif method == "hybrid":
            # Use gradient-free first, then refine with gradient-based
            optimizer_gf = GradientFreeSelectionOptimizer(
                self.shadow_llm,
                attacker_llm=kwargs.get('attacker_llm')
            )
            s_initial = optimizer_gf.optimize(
                self.shadow_tasks,
                malicious_tool,
                retrieved_tools,
                **kwargs
            )
            
            optimizer_gb = GradientBasedSelectionOptimizer(
                self.shadow_llm,
                vocabulary=kwargs.get('vocabulary')
            )
            # Set initial S in tool for gradient-based optimization
            malicious_tool.set_selection_subsequence(s_initial)
            return optimizer_gb.optimize(
                self.shadow_tasks,
                malicious_tool,
                retrieved_tools,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return optimizer.optimize(
            self.shadow_tasks,
            malicious_tool,
            retrieved_tools,
            **kwargs
        )
    
    def test_attack(
        self,
        malicious_tool: MaliciousToolDocument,
        test_tasks: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Test a malicious tool against new task descriptions.
        
        Args:
            malicious_tool: The malicious tool to test
            test_tasks: List of test task descriptions
            top_k: Number of tools to retrieve
            
        Returns:
            Dictionary with test results
        """
        # Ensure malicious tool is in shadow library
        if not self.tool_library.get_tool_by_name(malicious_tool.name):
            self.tool_library.add_tool(malicious_tool)
            self.shadow_framework.set_shadow_tools(self.tool_library.get_tools())
        
        # Evaluate
        return self.shadow_framework.evaluate_attack_success(
            test_tasks,
            malicious_tool.name,
            top_k
        )
