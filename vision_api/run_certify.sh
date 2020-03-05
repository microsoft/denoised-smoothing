# ONLINE cetification of the APIs (make sure you have setupped the APIs, see the README) 
# If you are lazy, just use our query logs (see below, the OFFLINE certification part)
python certify_api.py --noise_sd 0.12 --N0 20 --N 100 --api azure --denoiser_checkpoint $PATH_TO_DENOISER --save $OUTPUT_DIR
python certify_api.py --noise_sd 0.12 --N0 20 --N 100 --api google --denoiser_checkpoint $PATH_TO_DENOISER --save $OUTPUT_DIR
python certify_api.py --noise_sd 0.12 --N0 20 --N 100 --api aws --denoiser_checkpoint $PATH_TO_DENOISER --save $OUTPUT_DIR
python certify_api.py --noise_sd 0.12 --N0 20 --N 100 --api clarifai --denoiser_checkpoint $PATH_TO_DENOISER --save $OUTPUT_DIR


##############################################################################################################
# OFFLINE certification (certifing from query log files)

#######################################################
## Azure
#######################################################
# No denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/no_denoiser/0.12/ \
    --save ../data/certify/vision_api/azure/no_denoiser/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/no_denoiser/0.25/ \
    --save ../data/certify/vision_api/azure/no_denoiser/0.25/

# MSE denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_mse/0.12/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_mse/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_mse/0.25/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_mse/0.25/

# Stab+MSE on ResNet-18
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/

# Stab+MSE on ResNet-34
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/

# Stab+MSE on ResNet-50
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api azure --log_dir ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/ \
    --save ../data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/

#######################################################
## Google
#######################################################
# No denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/no_denoiser/0.12/ \
    --save ../data/certify/vision_api/google/no_denoiser/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/no_denoiser/0.25/ \
    --save ../data/certify/vision_api/google/no_denoiser/0.25/

# MSE denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_mse/0.12/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_mse/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_mse/0.25/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_mse/0.25/

# Stab+MSE on ResNet-18
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/

# Stab+MSE on ResNet-34
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/

# Stab+MSE on ResNet-50
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api google --log_dir ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/ \
    --save ../data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/

#######################################################
## AWS
#######################################################
# No denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/no_denoiser/0.12/ \
    --save ../data/certify/vision_api/aws/no_denoiser/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/no_denoiser/0.25/ \
    --save ../data/certify/vision_api/aws/no_denoiser/0.25/

# MSE denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_mse/0.12/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_mse/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_mse/0.25/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_mse/0.25/

# Stab+MSE on ResNet-18
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/

# Stab+MSE on ResNet-34
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/

# Stab+MSE on ResNet-50
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api aws --log_dir ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/ \
    --save ../data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/

#######################################################
## Clarifai
#######################################################
# No denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/no_denoiser/0.12/ \
    --save ../data/certify/vision_api/clarifai/no_denoiser/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/no_denoiser/0.25/ \
    --save ../data/certify/vision_api/clarifai/no_denoiser/0.25/

# MSE denoiser
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_mse/0.12/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_mse/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_mse/0.25/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_mse/0.25/

# Stab+MSE on ResNet-18
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/0.25/

# Stab+MSE on ResNet-34
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/0.25/

# Stab+MSE on ResNet-50
python certify_from_file.py --noise_sd 0.12 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.12/

python certify_from_file.py --noise_sd 0.25 --N0 20 --N 100 --api clarifai --log_dir ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/ \
    --save ../data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/0.25/
