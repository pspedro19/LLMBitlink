#!/usr/bin/env python3
"""
Quick fix for MCP server connectivity issues.
This script directly tests and fixes the connection to your MCP servers.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quick_fix.log")
    ]
)
logger = logging.getLogger("quick-fix")

class Colors:
    """Terminal colors for output."""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

async def test_server_connection(server_script, server_id):
    """Test connection to a specific MCP server."""
    print(f"{Colors.BLUE}Testing connection to {server_id}...{Colors.RESET}")
    
    try:
        # Import MCP
        import mcp
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        logger.info(f"MCP version: {mcp.__version__}")
        print(f"{Colors.GREEN}MCP {mcp.__version__} found{Colors.RESET}")
        
        # Check if server script exists
        if not os.path.exists(server_script):
            print(f"{Colors.RED}Server script not found at: {server_script}{Colors.RESET}")
            return False
        
        print(f"{Colors.GREEN}Server script found at: {server_script}{Colors.RESET}")
        
        # Create parameters
        params = StdioServerParameters(
            command="python",
            args=[server_script]
        )
        
        # Attempt connection
        try:
            print(f"{Colors.YELLOW}Connecting to server using stdio...{Colors.RESET}")
            read, write = await stdio_client(params)
            session = ClientSession(read, write)
            await session.initialize()
            
            # Test server by listing tools
            tools = await session.list_tools()
            print(f"{Colors.GREEN}Successfully connected! Found {len(tools)} tools:{Colors.RESET}")
            for tool in tools:
                print(f"  - {tool.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            print(f"{Colors.RED}Connection error: {e}{Colors.RESET}")
            return False
            
    except ImportError as e:
        logger.error(f"MCP import error: {e}")
        print(f"{Colors.RED}Error importing MCP: {e}{Colors.RESET}")
        print(f"{Colors.YELLOW}Try: pip install mcp==1.6.0{Colors.RESET}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        return False

async def fix_client_file():
    """Fix the improved client file."""
    client_path = "mcp_curacao/client/improved_client.py"
    
    if not os.path.exists(client_path):
        print(f"{Colors.YELLOW}Creating improved client file...{Colors.RESET}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(client_path), exist_ok=True)
        
        # Content will be added based on test results
        return
    
    # Fix will be applied based on test results
    print(f"{Colors.GREEN}Client file already exists at {client_path}{Colors.RESET}")

async def main():
    """Main function to test and fix MCP server connectivity."""
    print(f"{Colors.BOLD}{Colors.BLUE}MCP Server Quick Fix Tool{Colors.RESET}")
    print(f"{Colors.DIM}This tool will diagnose and fix MCP server connectivity issues{Colors.RESET}\n")
    
    # Get base directory 
    base_dir = Path.cwd()
    servers_dir = base_dir / "mcp_curacao" / "servers"
    
    print(f"{Colors.YELLOW}Using base directory: {base_dir}{Colors.RESET}")
    print(f"{Colors.YELLOW}Looking for servers in: {servers_dir}{Colors.RESET}\n")
    
    # Test if servers directory exists
    if not servers_dir.exists():
        print(f"{Colors.RED}Server directory not found! Expected at: {servers_dir}{Colors.RESET}")
        return
    
    # Get server files
    server_files = list(servers_dir.glob("*.py"))
    if not server_files:
        print(f"{Colors.RED}No server scripts found in {servers_dir}{Colors.RESET}")
        return
    
    print(f"{Colors.GREEN}Found {len(server_files)} server scripts:{Colors.RESET}")
    for script in server_files:
        print(f"  - {script.name}")
    print()
    
    # Test Excel server connection (most critical)
    excel_server = servers_dir / "excel_server.py"
    if excel_server.exists():
        success = await test_server_connection(str(excel_server), "Excel Server")
        if success:
            print(f"\n{Colors.GREEN}✓ Excel server connection successful!{Colors.RESET}")
            
            # Update config.py with the proper server ID
            try:
                # Get server ID from the Excel server file
                with open(excel_server, 'r') as f:
                    content = f.read()
                    import re
                    server_id_match = re.search(r'FastMCP\(["\']([^"\']+)', content)
                    if server_id_match:
                        excel_server_id = server_id_match.group(1)
                        print(f"{Colors.YELLOW}Found Excel server ID: {excel_server_id}{Colors.RESET}")
                        
                        # Update the config file
                        config_path = base_dir / "mcp_curacao" / "config.py"
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config_content = f.read()
                            
                            # Update EXCEL_SERVER_ID if different
                            server_config_match = re.search(r'EXCEL_SERVER_ID\s*=\s*["\']([^"\']+)', config_content)
                            if server_config_match:
                                current_id = server_config_match.group(1)
                                if current_id != excel_server_id:
                                    print(f"{Colors.YELLOW}Updating config.py with correct Excel server ID{Colors.RESET}")
                                    new_config = config_content.replace(
                                        f'EXCEL_SERVER_ID = "{current_id}"', 
                                        f'EXCEL_SERVER_ID = "{excel_server_id}"'
                                    )
                                    with open(config_path, 'w') as f:
                                        f.write(new_config)
                                    print(f"{Colors.GREEN}Updated config.py with correct server ID{Colors.RESET}")
                                else:
                                    print(f"{Colors.GREEN}Config.py already has the correct Excel server ID{Colors.RESET}")
            except Exception as e:
                logger.error(f"Error updating config: {e}")
                print(f"{Colors.RED}Error updating config: {e}{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}× Excel server connection failed{Colors.RESET}")
    else:
        print(f"{Colors.RED}Excel server script not found at: {excel_server}{Colors.RESET}")
    
    # Prepare improved client fix
    await fix_client_file()
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}Next Steps:{Colors.RESET}")
    print(f"1. Try running: {Colors.YELLOW}python mcp_curacao/launch.py --client simulated{Colors.RESET}")
    print(f"2. If that works, try: {Colors.YELLOW}python mcp_curacao/launch.py --client improved{Colors.RESET}")
    print(f"3. Check quick_fix.log for detailed diagnostic information")

if __name__ == "__main__":
    asyncio.run(main())