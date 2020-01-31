#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building roi align op..."
cd mmdet/ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi align rotated op..."
cd ../roi_align_rotated
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building ps roi align rotated op..."
cd ../psroi_align_rotated
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building dcn..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building sigmoid focal loss op..."
cd ../sigmoid_focal_loss
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building masked conv op..."
cd ../masked_conv
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building poly_nms op..."
cd ../poly_nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building cpu_nms..."
cd ../../core/bbox
$PYTHON setup_linux.py build_ext --inplace

