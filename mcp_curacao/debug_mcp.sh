#!/bin/bash
# Debug script for MCP Curaçao system

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MCP Curaçao System Debug Tool ===${NC}"
echo -e "${YELLOW}This tool will help diagnose connection issues with your MCP servers${NC}\n"

# Check MCP installation
echo -e "${BLUE}Checking MCP installation...${NC}"
if python -c "import mcp; print(f'MCP version: {mcp.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ MCP is installed correctly${NC}"
else
    echo -e "${RED}✗ MCP is not properly installed${NC}"
    echo -e "${YELLOW}Installing MCP 1.6.0...${NC}"
    pip install mcp==1.6.0
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install MCP. Please try manually:${NC}"
        echo -e "  pip install mcp==1.6.0"
        exit 1
    fi
fi

# Create debug directory if it doesn't exist
mkdir -p debug

# Test Excel server connection
echo -e "\n${BLUE}Testing Excel server connection...${NC}"
python debug/debug_excel_connection.py

# Check if debug was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Debug completed. You can now run the improved client:${NC}"
    echo -e "  python mcp_curacao/launch.py --client improved"
else
    echo -e "\n${RED}Debug failed. Please check the logs for more information.${NC}"
fi