ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
TARGET=u50
echo "-----------------------------------------"
echo "COMPILING MODEL FOR ALVEO U50.."
echo "-----------------------------------------"

compile() {
  vai_c_xir \
  --xmodel      quantize_result/CifarResNet_int.xmodel \
  --arch        $ARCH \
  --net_name    CifarResNet_int_${TARGET} \
  --output_dir  deploy
}


compile 2>&1 | tee compile.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"