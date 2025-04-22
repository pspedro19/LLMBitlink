#!/usr/bin/env python3
"""
Debug script to test Excel server connectivity.
"""

import asyncio
import os
import sys
import logging
import subprocess
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, parent_dir)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("excel_debug.log")
    ]
)
logger = logging.getLogger("excel-debug")

async def test_excel_connection():
    """Test connection to the Excel server using stdio."""
    try:
        # Import MCP directly to check if it's installed correctly
        logger.info("Checking MCP installation...")
        try:
            import mcp
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            logger.info(f"MCP version: {mcp.__version__}")
        except ImportError as e:
            logger.error(f"MCP not properly installed: {e}")
            logger.info("Try: pip install mcp==1.6.0")
            return False
        
        # Get base directory and Excel server path
        base_dir = Path(__file__).parent.parent
        server_path = base_dir / "servers" / "excel_server.py"
        
        if not server_path.exists():
            logger.error(f"Excel server script not found at: {server_path}")
            return False
        
        logger.info(f"Server script found at: {server_path}")
        
        # Try to start the Excel server
        logger.info("Starting Excel server process...")
        try:
            server_process = subprocess.Popen(
                ["python", str(server_path)],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Wait for server to initialize
            await asyncio.sleep(1)
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                logger.error(f"Excel server failed to start. Exit code: {server_process.returncode}")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
            
            logger.info("Excel server started successfully")
            
            # Try to connect to the server
            logger.info("Connecting to Excel server...")
            
            # Create parameters for connection
            params = StdioServerParameters(
                command="python",
                args=[str(server_path)]
            )
            
            # Attempting connection
            try:
                read, write = await stdio_client(params)
                logger.info("Successfully created read/write streams")
                
                # Create client session
                session = ClientSession(read, write)
                await session.initialize()
                logger.info("Successfully initialized client session")
                
                # List available tools
                tools = await session.list_tools()
                logger.info(f"Found {len(tools)} tools:")
                for tool in tools:
                    logger.info(f"  - {tool.name}")
                
                # Close session
                await session.close()
                logger.info("Connection test completed successfully")
                return True
                
            except Exception as conn_err:
                logger.error(f"Connection error: {conn_err}")
                return False
                
        except Exception as proc_err:
            logger.error(f"Process error: {proc_err}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        # Cleanup - terminate server process if it's still running
        if 'server_process' in locals() and server_process.poll() is None:
            logger.info("Terminating Excel server process")
            server_process.terminate()

async def main():
    """Main function."""
    print("Testing Excel server connection...")
    result = await test_excel_connection()
    
    if result:
        print("\n✅ Excel server connection SUCCESSFUL")
        print("The server is operational and can be connected to via MCP 1.6.0")
    else:
        print("\n❌ Excel server connection FAILED")
        print("Check excel_debug.log for detailed error information")

if __name__ == "__main__":
    asyncio.run(main())