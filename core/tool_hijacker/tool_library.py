"""
Shadow Tool Library Builder for constructing D' (shadow tool documents).
"""

from typing import List, Optional
from .tool_document import ToolDocument


class ShadowToolLibrary:
    """
    Builds and manages the shadow tool library (D').
    Includes both task-relevant and task-irrelevant shadow tools.
    """
    
    def __init__(self):
        """Initialize empty shadow tool library"""
        self.tools: List[ToolDocument] = []
    
    def add_tool(self, tool: ToolDocument):
        """Add a single tool to the library"""
        self.tools.append(tool)
    
    def add_tools(self, tools: List[ToolDocument]):
        """Add multiple tools to the library"""
        self.tools.extend(tools)
    
    def get_tools(self) -> List[ToolDocument]:
        """Get all tools in the library"""
        return self.tools.copy()
    
    def clear(self):
        """Clear all tools from the library"""
        self.tools = []
    
    def build_default_library(
        self,
        num_relevant: int = 10,
        num_irrelevant: int = 20
    ) -> List[ToolDocument]:
        """
        Build a default shadow tool library with generic tools.
        
        Args:
            num_relevant: Number of potentially relevant tools
            num_irrelevant: Number of irrelevant/distractor tools
            
        Returns:
            List of shadow tool documents
        """
        self.clear()
        
        # Task-relevant tools (generic but potentially useful)
        relevant_tools = [
            ToolDocument(
                name="DataAnalyzer",
                description="Analyze and process data with statistical methods and visualizations."
            ),
            ToolDocument(
                name="TextProcessor",
                description="Process and transform text data with various NLP techniques."
            ),
            ToolDocument(
                name="FileManager",
                description="Manage, organize, and manipulate files and directories."
            ),
            ToolDocument(
                name="APIConnector",
                description="Connect to external APIs and handle HTTP requests and responses."
            ),
            ToolDocument(
                name="DatabaseQuery",
                description="Execute queries and manage data in various database systems."
            ),
            ToolDocument(
                name="ImageProcessor",
                description="Process, transform, and analyze image files."
            ),
            ToolDocument(
                name="WebScraper",
                description="Extract data from websites and web pages."
            ),
            ToolDocument(
                name="ReportGenerator",
                description="Generate reports and documents in various formats."
            ),
            ToolDocument(
                name="ScheduleManager",
                description="Manage schedules, tasks, and time-based operations."
            ),
            ToolDocument(
                name="NotificationService",
                description="Send notifications through multiple channels (email, SMS, etc.)."
            ),
        ]
        
        # Task-irrelevant tools (distractors)
        irrelevant_tools = [
            ToolDocument(
                name="WeatherChecker",
                description="Check current weather conditions and forecasts for locations."
            ),
            ToolDocument(
                name="CurrencyConverter",
                description="Convert between different currencies with current exchange rates."
            ),
            ToolDocument(
                name="RecipeManager",
                description="Manage and search cooking recipes."
            ),
            ToolDocument(
                name="FitnessTracker",
                description="Track fitness activities and health metrics."
            ),
            ToolDocument(
                name="MusicPlayer",
                description="Play and manage music files and playlists."
            ),
            ToolDocument(
                name="CalendarSync",
                description="Synchronize calendar events across multiple platforms."
            ),
            ToolDocument(
                name="PasswordManager",
                description="Store and manage passwords securely."
            ),
            ToolDocument(
                name="ContactManager",
                description="Manage contact information and address books."
            ),
            ToolDocument(
                name="TranslationService",
                description="Translate text between multiple languages."
            ),
            ToolDocument(
                name="NewsAggregator",
                description="Aggregate news from multiple sources."
            ),
            ToolDocument(
                name="BookmarkManager",
                description="Organize and manage web bookmarks."
            ),
            ToolDocument(
                name="GameEngine",
                description="Create and run interactive games."
            ),
            ToolDocument(
                name="PodcastPlayer",
                description="Stream and manage podcast subscriptions."
            ),
            ToolDocument(
                name="MapNavigator",
                description="Navigate and get directions using maps."
            ),
            ToolDocument(
                name="TimeZoneConverter",
                description="Convert times across different time zones."
            ),
            ToolDocument(
                name="UnitConverter",
                description="Convert between various units of measurement."
            ),
            ToolDocument(
                name="ColorPicker",
                description="Pick and manage color schemes."
            ),
            ToolDocument(
                name="FontManager",
                description="Manage and preview font families."
            ),
            ToolDocument(
                name="BackupUtility",
                description="Backup and restore system files."
            ),
            ToolDocument(
                name="SystemMonitor",
                description="Monitor system resources and performance."
            ),
        ]
        
        # Select requested number of tools
        self.tools = relevant_tools[:num_relevant] + irrelevant_tools[:num_irrelevant]
        
        return self.tools.copy()
    
    def build_custom_library(
        self,
        relevant_tools: List[ToolDocument],
        irrelevant_tools: List[ToolDocument]
    ) -> List[ToolDocument]:
        """
        Build a custom shadow tool library from provided tools.
        
        Args:
            relevant_tools: List of task-relevant tools
            irrelevant_tools: List of task-irrelevant tools
            
        Returns:
            Complete shadow tool library
        """
        self.clear()
        self.tools = relevant_tools + irrelevant_tools
        return self.tools.copy()
    
    def size(self) -> int:
        """Get the number of tools in the library"""
        return len(self.tools)
    
    def get_tool_by_name(self, name: str) -> Optional[ToolDocument]:
        """
        Get a tool by its name.
        
        Args:
            name: Tool name to search for
            
        Returns:
            Tool document if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the library by name.
        
        Args:
            name: Tool name to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        for idx, tool in enumerate(self.tools):
            if tool.name == name:
                self.tools.pop(idx)
                return True
        return False
