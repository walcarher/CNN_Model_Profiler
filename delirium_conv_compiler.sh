# Create a root directory for VHDL generated files as a tree structure
mkdir "vhdl_generated"

# CNN Hyperparameters listing for Delirium VHDL code generator. ATTENTION: This must match the parameters of conv_gen.py file!
# Future modificatios: Add [args] arguments to conv_gen.py
# Operations: Conv kxk
# Input tensor size: HeightxWeight HxW
# Input channel depth: C
# Number of filters/output channels: N
#CONV=("conv1x1" "conv3x3" "conv5x5" "conv7x7" "conv11x11")
# AlexNet CNN Layers
K=(11 5 3 3 3)
HW=(224 27 13 13 13)
C=(3 64 192 384 256)
N=(64 192 384 256 256)

# Generate an iterated folder for each case OperationsxHWxCxN
for i in "${!K[@]}";
do
	mkdir "vhdl_generated/conv"${K[i]}"x"${K[i]}
	mkdir "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}
	mkdir "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}"/c_"${C[i]}
done

# Create .onnx files on each folder vhdl_generated/conkxk/hw_HW/c_C for all n cases in N
python3 CNN_profiler.py -m AlexNet -g 1;

# Using Delirium to automatically generate .vhdl files (weights and compilers) from .onnx files. ATTENTION: Delirium is not included!
# Create a folder vhdl_generated/convkxk/hw_HW/c_C/convkxk_hw_c_n_vhdl_generated with all the output .vhdl files
for i in "${!K[@]}";
do
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}"/c_"${C[i]}"/conv"${K[i]}"x"${K[i]}"_"${HW[i]}"_"${C[i]}"_"${N[i]}".onnx" --out "vhdl_generated/conv"${K[i]}"x"${K[i]}"/hw_"${HW[i]}"/c_"${C[i]}"/conv"${K[i]}"x"${K[i]}"_"${HW[i]}"_"${C[i]}"_"${N[i]}"_vhdl_generated" --nbits 8;	
done

