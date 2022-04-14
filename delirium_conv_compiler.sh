# Create a root directory for VHDL generated files as a tree structure
mkdir "vhdl_generated"

# CNN Hyperparameters listing for Delirium VHDL code generator. ATTENTION: This must match the parameters of conv_gen.py file!
# Operations: Conv kxk
# Input tensor size: HeightxWeight HxW
# Input channel depth: C
# Number of filters/output channels: N
#CONV=("conv1x1" "conv3x3" "conv5x5" "conv7x7" "conv11x11")

# AlexNet CNN Layers
# K=(11 5 3 3 3)
# HW=(224 27 13 13 13)
# C=(3 64 192 384 256)
# N=(64 192 384 256 256)

# VGG16 CNN Layers
# K=(3 3 3 3 3 3 3 3 3 3 3 3 3)
# HW=(224 224 112 112 56 56 56 28 28 28 14 14 14)
# C=(3 64 64 128 128 256 256 256 512 512 512 512 512)
# N=(64 64 128 128 256 256 256 512 512 512 512 512 512)

# SqueezeNet CNN Layers
# K=(7 1 1 3 1 1 3 1 1 3 1 1 3 1 1 3 1 1 3 1 1 3 1 1 3)
# HW=(224 54 54 54 54 54 54 54 54 54 27 27 27 27 27 27 27 27 27 27 27 27 13 13 13)
# C=(3 96 16 16 128 16 16 128 32 32 256 32 32 256 48 48 384 48 48 384 64 64 512 64 64)
# N=(96 16 64 64 16 64 64 32 128 128 32 128 128 48 192 192 48 192 192 64 256 256 64 256 256)

# InceptionV3
K=(3 3 3 1 3 1 1 5 1 3 3 1 1 1 5 1 3 3 1 1 1 5 1 3 3 1 3 1 3 3 1 1 1 7 1 7 1 7 1 1 1 1 1 7 1 7 1 7 1 1 1 1 1 7 1 7 1 7 1 1 1 1 1 7 1 7 1 7 1 1 1 5 1 3 1 1 7 3 1 1 1 3 1 3 1 3 1 1 1 1 3 1 3 1 3 1)
HW=(224 224 111 111 109 109 54 54 54 54 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12)
C=(3 3 32 32 32 32 64 64 80 80 192 192 192 192 48 48 192 192 64 64 96 96 192 192 256 256 256 256 48 48 256 256 64 64 96 96 256 256 288 288 288 288 48 48 288 288 64 64 96 96 288 288 288 288 288 288 64 64 96 96 768 768 768 768 128 128 128 128 768 768 128 128 128 128 128 128 128 128 768 768 768 768 768 768 160 160 160 160 768 768 160 160 160 160 160 160)
N=(32 32 64 80 192 64 48 64 64 96 96 32 64 48 64 64 96 96 64 64 48 64 64 96 96 64 384 64 96 96 192 128 128 192 128 128 128 128 192 192 192 160 160 192 160 160 160 160 192 192 192 160 160 192 160 160 160 160 192 192 192 192 192 192 192 192 192 192 192 192 128 768 192 320 192 192 192 192 320 384 384 384 448 384 384 384 192 320 384 384 384 448 384 384 384 192)

# Generate an iterated folder for each case OperationsxHWxCxN
for i in "${!K[@]}";
do
	mkdir "vhdl_generated/conv"${K[i]}"x"${K[i]}
	mkdir "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}
	mkdir "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}"/c_"${C[i]}
done

# Create .onnx files on each folder vhdl_generated/conkxk/hw_HW/c_C for all n cases in N
python3 CNN_profiler.py -m InceptionV3 -g 1;

# Using Delirium to automatically generate .vhdl files (weights and compilers) from .onnx files. ATTENTION: Delirium is not included!
# Create a folder vhdl_generated/convkxk/hw_HW/c_C/convkxk_hw_c_n_vhdl_generated with all the output .vhdl files
for i in "${!K[@]}";
do
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}"/c_"${C[i]}"/conv"${K[i]}"x"${K[i]}"_"${HW[i]}"_"${C[i]}"_"${N[i]}".onnx" --out "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}"/c_"${C[i]}"/conv"${K[i]}"x"${K[i]}"_"${HW[i]}"_"${C[i]}"_"${N[i]}"_vhdl_generated" --nbits 8;	
done

