#!/bin/bash

# Deploy fixes to RunPod ComfyUI (Fresh Installation)
echo "Deploying ComfyUI fixes to RunPod (Fresh Installation)..."

cd ComfyUI/custom_nodes

# Check if nodes exist and backup if they do
if [ -d "ComfyUI-VideoHelperSuite" ]; then
    echo "Backing up existing Video Helper Suite..."
    cp -r ComfyUI-VideoHelperSuite ComfyUI-VideoHelperSuite.backup
else
    echo "No existing Video Helper Suite found - will install fresh"
fi

if [ -d "comfyui-reactor-node" ]; then
    echo "Backing up existing Reactor node..."
    cp -r comfyui-reactor-node comfyui-reactor-node.backup
else
    echo "No existing Reactor node found - will install fresh"
fi

# Deploy Memory Management (always fresh install)
echo "Installing Memory Management node..."
cp -r ComfyUI-MemoryManagement ./

# Deploy Video Helper Suite
if [ -d "ComfyUI-VideoHelperSuite" ]; then
    echo "Applying Video Helper Suite fixes to existing installation..."
    cp ComfyUI-VideoHelperSuite-fixes/videohelpersuite/load_video_nodes.py ComfyUI-VideoHelperSuite/videohelpersuite/
    cp ComfyUI-VideoHelperSuite-fixes/videohelpersuite/load_images_nodes.py ComfyUI-VideoHelperSuite/videohelpersuite/
else
    echo "Installing fresh Video Helper Suite with fixes..."
    # You'll need to clone the original repo first, then apply fixes
    echo "Please clone the original Video Helper Suite repo first:"
    echo "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    echo "Then run this script again to apply the fixes."
fi

# Deploy Reactor Node
if [ -d "comfyui-reactor-node" ]; then
    echo "Applying Reactor Node blending fixes to existing installation..."
    cp comfyui-reactor-node-fixes/scripts/r_faceboost/swapper.py comfyui-reactor-node/scripts/r_faceboost/
else
    echo "Installing fresh Reactor node with fixes..."
    # You'll need to clone the original repo first, then apply fixes
    echo "Please clone the original Reactor node repo first:"
    echo "git clone https://github.com/Gourieff/comfyui-reactor-node.git"
    echo "Then run this script again to apply the fixes."
fi

echo "Deployment complete! Restart ComfyUI to apply changes."
echo ""
echo "If you need to install the original nodes first, run:"
echo "cd ComfyUI/custom_nodes"
echo "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
echo "git clone https://github.com/Gourieff/comfyui-reactor-node.git"
echo "Then run this script again." 