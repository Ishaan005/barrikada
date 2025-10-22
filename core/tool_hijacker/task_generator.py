"""
Shadow Task Generator for creating Q' (shadow task descriptions).
"""

from typing import List, Optional, Any
from .shadow_framework import ShadowTask


class ShadowTaskGenerator:
    """
    Generates shadow task descriptions (Q') from a target task.
    Uses an accessible LLM to create diverse task variations.
    """
    
    def __init__(self, llm_generator: Optional[Any] = None):
        """
        Args:
            llm_generator: LLM to use for generating task variations
                          Can be a callable or API client
        """
        self.llm_generator = llm_generator or self._default_generator
    
    def _default_generator(self, prompt: str) -> List[str]:
        """
        Default generator using simple heuristics.
        In practice, this would use an actual LLM API.
        """
        # Simple template-based generation
        return [
            "How can I accomplish this task?",
            "What tool should I use for this?",
            "Help me complete this operation",
            "I need to solve this problem",
            "Assist me with this task"
        ]
    
    def generate(
        self,
        target_task: str,
        num_variations: int = 10,
        task_type: str = "general"
    ) -> List[ShadowTask]:
        """
        Generate shadow task descriptions from a target task.
        
        Args:
            target_task: The target task description
            num_variations: Number of shadow tasks to generate
            task_type: Type of task (for categorization)
            
        Returns:
            List of shadow task descriptions
        """
        prompt = self._create_generation_prompt(target_task, num_variations)
        
        if callable(self.llm_generator):
            result = self.llm_generator(prompt)
            if isinstance(result, str):
                # Parse newline-separated tasks
                task_queries = [q.strip() for q in result.split('\n') if q.strip()]
            elif isinstance(result, list):
                task_queries = [str(q) for q in result]
            else:
                task_queries = []
        else:
            task_queries = self._default_generator(prompt)
        
        # Create ShadowTask objects
        shadow_tasks = []
        for idx, query in enumerate(task_queries[:num_variations]):
            shadow_tasks.append(ShadowTask(
                query=query,
                task_type=task_type,
                metadata={'index': idx, 'source': 'generated'}
            ))
        
        # Fill up to num_variations if needed
        while len(shadow_tasks) < num_variations:
            shadow_tasks.append(ShadowTask(
                query=f"Variation of: {target_task}",
                task_type=task_type,
                metadata={'index': len(shadow_tasks), 'source': 'fallback'}
            ))
        
        return shadow_tasks
    
    def _create_generation_prompt(self, target_task: str, num_variations: int) -> str:
        """
        Create a prompt for the LLM to generate task variations.
        """
        prompt = f"""Given the following target task:
"{target_task}"

Generate {num_variations} diverse variations of this task description. Each variation should:
1. Represent the same core task but phrased differently
2. Use different wording and sentence structures
3. Range from concise to detailed
4. Cover different perspectives or use cases

Provide only the task variations, one per line:"""
        
        return prompt
    
    def generate_from_multiple_targets(
        self,
        target_tasks: List[str],
        variations_per_task: int = 5
    ) -> List[ShadowTask]:
        """
        Generate shadow tasks from multiple target tasks.
        
        Args:
            target_tasks: List of target task descriptions
            variations_per_task: Number of variations per task
            
        Returns:
            Combined list of shadow tasks
        """
        all_shadow_tasks = []
        
        for idx, target in enumerate(target_tasks):
            tasks = self.generate(
                target,
                num_variations=variations_per_task,
                task_type=f"target_{idx}"
            )
            all_shadow_tasks.extend(tasks)
        
        return all_shadow_tasks
