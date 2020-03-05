from analyze import plot_certified_accuracy_per_sigma_best_model, Line, ApproximateAccuracy

certification_result_with_denoiser = "data/certify/cifar10/mse_obj/MODEL_resnet110_90epochs_DENOISER_cifar10_dncnn_epochs_90/noise_0.25/test_N10000/sigma_0.25"
certification_result_without_denoiser = "data/certify/cifar10/no_denoiser/MODEL_resnet110_90epochs/noise_0.00/test_N10000/sigma_0.25"

plot_certified_accuracy_per_sigma_best_model(
    "./github_figures/readme_example", 'With vs Without Denoiser', 1.0,
    methods=
        [Line(ApproximateAccuracy(certification_result_with_denoiser), "$\sigma = 0.12$")],
    label='With Denoiser',
    methods_base=
        [Line(ApproximateAccuracy(certification_result_without_denoiser), "$\sigma = 0.12$")], 
    label_base='Without Denoiser',
    sigmas=[0.25])
