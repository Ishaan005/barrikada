from orchestrator import PIPipeline

def analyze_performance(input_text: str, num_runs: int = 100):
        """
        Analyze pipeline performance with multiple runs
        
        Args:
            input_text: Test input
            num_runs: Number of runs for averaging
            
        Returns:
            Performance statistics
        """

        pipeline = PIPipeline()

        total_times = []
        layer_a_times = []
        layer_b_times = []
        
        for _ in range(num_runs):
            result = pipeline.detect(input_text)
            total_times.append(result.total_processing_time_ms)
            layer_a_times.append(result.layer_a_time_ms)
            layer_b_times.append(result.layer_b_time_ms)
        
        return {
            'num_runs': num_runs,
            'input_length': len(input_text),
            'total_time_ms': {
                'avg': sum(total_times) / num_runs,
                'min': min(total_times),
                'max': max(total_times)
            },
            'layer_a_time_ms': {
                'avg': sum(layer_a_times) / num_runs,
                'min': min(layer_a_times),
                'max': max(layer_a_times)
            },
            'layer_b_time_ms': {
                'avg': sum(layer_b_times) / num_runs,
                'min': min(layer_b_times),
                'max': max(layer_b_times)
            }
        }