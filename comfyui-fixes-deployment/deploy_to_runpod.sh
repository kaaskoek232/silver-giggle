#!/bin/bash 
 
# Deploy fixes to RunPod ComfyUI 
echo "Deploying ComfyUI fixes to RunPod..." 
 
# Backup current versions 
cd ComfyUI/custom_nodes 
cp -r ComfyUI-VideoHelperSuite ComfyUI-VideoHelperSuite.backup 
cp -r comfyui-reactor-node comfyui-reactor-node.backup 
 
# Deploy Memory Management 
if [ ! -d "ComfyUI-MemoryManagement" ]; then 
    echo "Installing Memory Management node..." 
    cp -r ComfyUI-MemoryManagement ./ 
fi 
 
# Deploy Video Helper Suite fixes 
echo "Applying Video Helper Suite fixes..." 
cp ComfyUI-VideoHelperSuite-fixes/videohelpersuite/load_video_nodes.py ComfyUI-VideoHelperSuite/videohelpersuite/ 
cp ComfyUI-VideoHelperSuite-fixes/videohelpersuite/load_images_nodes.py ComfyUI-VideoHelperSuite/videohelpersuite/ 
 
# Deploy Reactor Node fixes 
echo "Applying Reactor Node blending fixes..." 
cp comfyui-reactor-node-fixes/scripts/r_faceboost/swapper.py comfyui-reactor-node/scripts/r_faceboost/ 
 
echo "Deployment complete! Restart ComfyUI to apply changes." 
