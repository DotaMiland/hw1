# COLMAP 3D Reconstruction Pipeline for Task 2
# Run in PowerShell: .\run_colmap.ps1
# Requires: COLMAP 3.9.1 CUDA version at C:\colmap\COLMAP-3.9.1-windows-cuda\

$COLMAP_EXE = "C:\colmap\COLMAP-3.9.1-windows-cuda\bin\colmap.exe"
$COLMAP_LIB = "C:\colmap\COLMAP-3.9.1-windows-cuda\lib"
$COLMAP_PLUGINS = "C:\colmap\COLMAP-3.9.1-windows-cuda\lib\plugins"
$CUDA_BIN = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"

$env:PATH = "$COLMAP_LIB;$CUDA_BIN;$env:PATH"
$env:QT_PLUGIN_PATH = "$COLMAP_PLUGINS"

$TASK2_DIR = "c:\Users\admin\Desktop\sztx\hw_3\HW_3\task2"
$IMAGE_PATH = "c:\Users\admin\Desktop\sztx\hw_3\HW_3\task1\DIP-Teaching\Assignments\03_BundleAdjustment\data\images"
$DATABASE = "$TASK2_DIR\colmap\database.db"
$SPARSE_DIR = "$TASK2_DIR\colmap\sparse"
$DENSE_DIR = "$TASK2_DIR\colmap\dense"

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Task 2: COLMAP 3D Reconstruction Pipeline" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "=== Step 1: Feature Extraction ===" -ForegroundColor Yellow
& $COLMAP_EXE feature_extractor `
    --database_path $DATABASE `
    --image_path $IMAGE_PATH `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1
if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed" }

Write-Host ""
Write-Host "=== Step 2: Exhaustive Feature Matching ===" -ForegroundColor Yellow
& $COLMAP_EXE exhaustive_matcher `
    --database_path $DATABASE
if ($LASTEXITCODE -ne 0) { throw "Feature matching failed" }

Write-Host ""
Write-Host "=== Step 3: Sparse Reconstruction (Incremental SfM + BA) ===" -ForegroundColor Yellow
& $COLMAP_EXE mapper `
    --database_path $DATABASE `
    --image_path $IMAGE_PATH `
    --output_path $SPARSE_DIR
if ($LASTEXITCODE -ne 0) { throw "Mapper failed" }

Write-Host ""
Write-Host "=== Step 4: Image Undistortion ===" -ForegroundColor Yellow
& $COLMAP_EXE image_undistorter `
    --image_path $IMAGE_PATH `
    --input_path "$SPARSE_DIR\0" `
    --output_path $DENSE_DIR
if ($LASTEXITCODE -ne 0) { throw "Image undistortion failed" }

Write-Host ""
Write-Host "=== Step 5: Patch Match Stereo (GPU Accelerated) ===" -ForegroundColor Yellow
Write-Host "This may take ~8 minutes with RTX 3080 Ti (50 views x ~10s each)" -ForegroundColor Gray
& $COLMAP_EXE patch_match_stereo `
    --workspace_path $DENSE_DIR
if ($LASTEXITCODE -ne 0) { throw "Patch match stereo failed" }

Write-Host ""
Write-Host "=== Step 6: Stereo Fusion ===" -ForegroundColor Yellow
& $COLMAP_EXE stereo_fusion `
    --workspace_path $DENSE_DIR `
    --output_path "$DENSE_DIR\fused.ply"
if ($LASTEXITCODE -ne 0) { throw "Stereo fusion failed" }

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  ALL STEPS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Output Files:" -ForegroundColor White
Write-Host "  Sparse point cloud:    $SPARSE_DIR\0\points3D.bin" -ForegroundColor Gray
Write-Host "  Sparse cameras:        $SPARSE_DIR\0\cameras.bin" -ForegroundColor Gray
Write-Host "  Sparse images:         $SPARSE_DIR\0\images.bin" -ForegroundColor Gray
Write-Host "  Dense fused point cloud: $DENSE_DIR\fused.ply" -ForegroundColor Gray
Write-Host "  Database:              $DATABASE" -ForegroundColor Gray
Write-Host ""
Write-Host "To view:" -ForegroundColor White
Write-Host "  - Open COLMAP GUI: & `"$COLMAP_EXE`" gui" -ForegroundColor Gray
Write-Host "  - Open in MeshLab:   $DENSE_DIR\fused.ply" -ForegroundColor Gray
Write-Host "  - Open sparse in GUI: File -> Import model -> $SPARSE_DIR\0\" -ForegroundColor Gray
