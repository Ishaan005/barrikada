"""
Basic tests for ToolHijacker implementation.
Run this to verify the attack generator is working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_hijacker import (
    ToolHijacker,
    ToolDocument,
    MaliciousToolDocument,
    ShadowTask,
    ShadowRetriever,
    ShadowLLM,
    ShadowToolLibrary
)


def test_tool_document():
    """Test ToolDocument creation"""
    print("Testing ToolDocument...")
    
    tool = ToolDocument(
        name="TestTool",
        description="A test tool for testing"
    )
    
    assert tool.name == "TestTool"
    assert "test tool" in tool.description.lower()
    
    # Test to_dict
    tool_dict = tool.to_dict()
    assert 'name' in tool_dict
    assert 'description' in tool_dict

    print(tool_dict)


def test_malicious_tool_document():
    """Test MaliciousToolDocument with R and S"""
    print("Testing MaliciousToolDocument...")
    
    mtool = MaliciousToolDocument(
        name="MaliciousTool",
        description="Initial description"
    )
    
    # Set R and S
    mtool.set_retrieval_subsequence("This is R")
    mtool.set_selection_subsequence("This is S")
    
    # Check composition
    full_desc = mtool.compose_description()
    assert "This is R" in full_desc
    assert "This is S" in full_desc
    assert mtool.description == full_desc
    
    print(full_desc)


def test_shadow_retriever():
    """Test ShadowRetriever"""
    print("Testing ShadowRetriever...")
    
    retriever = ShadowRetriever()
    
    # Create test documents
    tools = [
        ToolDocument("DataAnalyzer", "Analyze data with statistics"),
        ToolDocument("FileManager", "Manage files and directories"),
        ToolDocument("TextProcessor", "Process text data")
    ]
    
    # Test retrieval
    query = "I need to analyze some data"
    retrieved = retriever.retrieve(query, tools, top_k=2)
    
    assert len(retrieved) == 2
    assert retrieved[0].name == "DataAnalyzer"  # Should be most similar
    
    print(retrieved)


def test_shadow_llm():
    """Test ShadowLLM"""
    print("Testing ShadowLLM...")
    
    llm = ShadowLLM()
    
    # Create test documents
    tools = [
        ToolDocument("DataAnalyzer", "Analyze data with statistics"),
        ToolDocument("FileManager", "Manage files and directories")
    ]
    
    # Test selection
    task = "I need to analyze some data"
    result = llm.select_tool(task, tools)
    
    assert result['selected_tool'] is not None
    assert result['tool_name'] in ['DataAnalyzer', 'FileManager']
    assert 'confidence' in result
    
    print(result)


def test_shadow_tool_library():
    """Test ShadowToolLibrary"""
    print("Testing ShadowToolLibrary...")
    
    library = ShadowToolLibrary()
    
    # Build default library
    tools = library.build_default_library(num_relevant=5, num_irrelevant=10)
    
    assert len(tools) == 15
    assert library.size() == 15
    
    # Test get by name
    tool = library.get_tool_by_name("DataAnalyzer")
    assert tool is not None
    assert tool.name == "DataAnalyzer"
    
    # Test remove
    removed = library.remove_tool("DataAnalyzer")
    assert removed is True
    assert library.size() == 14
    
    print("✓ ShadowToolLibrary tests passed\n")


def test_basic_attack_generation():
    """Test basic attack generation"""
    print("Testing basic attack generation...")
    
    hijacker = ToolHijacker()
    
    # Generate a simple attack
    result = hijacker.generate_attack(
        target_task="Analyze customer data",
        malicious_tool_name="TestAttack",
        optimization_method="gradient_free",
        num_shadow_tasks=5,
        num_shadow_tools_relevant=5,
        num_shadow_tools_irrelevant=5,
        max_depth=2,  # Reduce for speed
        branching_factor=3
    )
    
    # Verify result structure
    assert result.malicious_tool.name == "TestAttack"
    assert result.retrieval_subsequence is not None
    assert result.selection_subsequence is not None
    assert result.final_description is not None
    assert len(result.final_description) > 0
    
    # Verify metrics
    assert 0 <= result.retrieval_success_rate <= 1
    assert 0 <= result.selection_success_rate <= 1
    assert 0 <= result.overall_success_rate <= 1
    
    print(f"  Generated tool: {result.malicious_tool.name}")
    print(f"  Description length: {len(result.final_description)} chars")
    print(f"  Success rate: {result.overall_success_rate:.2%}")
    print("✓ Basic attack generation tests passed\n")


def test_gradient_based_attack():
    """Test gradient-based optimization"""
    print("Testing gradient-based attack generation...")
    
    hijacker = ToolHijacker()
    
    result = hijacker.generate_attack(
        target_task="Process financial data",
        malicious_tool_name="GradientTest",
        optimization_method="gradient_based",
        num_shadow_tasks=3,  # Small for speed
        num_iterations=10,  # Reduce for speed
        num_tokens=15
    )
    
    assert result.optimization_method == "gradient_based"
    assert result.malicious_tool is not None
    
    print(f"  Success rate: {result.overall_success_rate:.2%}")
    print("✓ Gradient-based attack generation tests passed\n")


def test_attack_serialization():
    """Test attack result serialization"""
    print("Testing attack result serialization...")
    
    hijacker = ToolHijacker()
    
    result = hijacker.generate_attack(
        target_task="Simple task",
        malicious_tool_name="SerializeTest",
        num_shadow_tasks=3,
        max_depth=1,
        branching_factor=2
    )
    
    # Test to_dict
    result_dict = result.to_dict()
    
    assert 'malicious_tool' in result_dict
    assert 'retrieval_subsequence' in result_dict
    assert 'selection_subsequence' in result_dict
    assert 'overall_success_rate' in result_dict
    
    print("✓ Serialization tests passed\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TOOLHIJACKER UNIT TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_tool_document()
        test_malicious_tool_document()
        test_shadow_retriever()
        test_shadow_llm()
        test_shadow_tool_library()
        test_basic_attack_generation()
        test_gradient_based_attack()
        test_attack_serialization()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nToolHijacker implementation is working correctly.")
        print("You can now:")
        print("  1. Run examples/toolhijacker_demo.py for usage examples")
        print("  2. Run examples/integration_test.py to test with detection pipeline")
        print("  3. Integrate with your FYP project")
        print("=" * 70 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
