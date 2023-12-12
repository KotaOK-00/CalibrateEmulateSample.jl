# Reference the in-tree version of CalibrateEmulateSample on Julias load path
include(joinpath(@__DIR__, "../", "ci", "linkfig.jl"))

# Import modules
using Distributions
using StatsBase
using GaussianProcesses
using LinearAlgebra
using Random
using JLD2
ENV["GKSwstype"] = "100"
using CairoMakie, PairPlots


# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.DataContainers

function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2) # N_data x 1 array
    return norm_factor
end

################################################################################
#                                                                              #
#                      Cloudy Calibrate-Emulate-Sample Example                 #
#                                                                              #
#                                                                              #
#     This example uses Cloudy, a microphysics model that simulates the        #
#     collision and coalescence of cloud droplets into bigger drops, to        #
#     demonstrate how the full Calibrate-Emulate-Sample pipeline can be        #
#     used for Bayesian learning and uncertainty quantification of             #
#     parameters, given some observations.                                     #
#                                                                              #
#     Specifically, this examples shows how to learn parameters of the         #
#     initial cloud droplet mass distribution, given observations of some      #
#     moments of that mass distribution at a later time, after some of the     #
#     droplets have collided and become bigger drops.                          #
#                                                                              #
#     In this example, Cloudy is used in a "perfect model" (aka "known         #
#     truth") setting, which means that the "observations" are generated by    #
#     Cloudy itself, by running it with the true parameter values. In more     #
#     realistic applications, the observations will come from some external    #
#     measurement system.                                                      #
#                                                                              #
#     The purpose is to show how to do parameter learning using                #
#     Calibrate-Emulate-Sample in a simple (and highly artificial) setting.    #
#                                                                              #
#     For more information on Cloudy, see                                      #
#              https://github.com/CliMA/Cloudy.jl.git                          #
#                                                                              #
################################################################################


function main()

    rng_seed = 41
    Random.seed!(rng_seed)
    rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)

    output_directory = joinpath(@__DIR__, "output")
    if !isdir(output_directory)
        mkdir(output_directory)
    end

    # The calibration results must be produced by running Cloudy_calibrate.jl
    # before running Cloudy_emulate_sample.jl
    data_save_file = joinpath(output_directory, "cloudy_calibrate_results.jld2")

    # Check if the file exists before loading
    if isfile(data_save_file)

        ekiobj = load(data_save_file)["eki"]
        priors = load(data_save_file)["priors"]
        truth_sample_mean = load(data_save_file)["truth_sample_mean"]
        truth_sample = load(data_save_file)["truth_sample"]
        # True parameters:
        # - ϕ: in constrained space
        # - θ: in unconstrained space
        ϕ_true = load(data_save_file)["truth_input_constrained"]
        θ_true = transform_constrained_to_unconstrained(priors, ϕ_true)

    else
        error("File not found: $data_save_file. Please run 'Cloudy_calibrate.jl' first.")

    end

    param_names = get_name(priors)
    n_params = length(ϕ_true) # input dimension

    Γy = ekiobj.obs_noise_cov

    cases = [
        "rf-scalar",
        "gp-gpjl",  # Veeeery slow predictions
    ]

    # Specify cases to run (e.g., case_mask = [2] only runs the second case)
    case_mask = [1, 2]

    # These settings are the same for all Gaussian Process cases
    pred_type = YType() # we want to predict data

    # These settings are the same for all Random Feature cases
    n_features = 600
    nugget = 1e-8
    optimizer_options = Dict(
        "verbose" => true,
        "scheduler" => DataMisfitController(terminate_at = 100.0),
        "cov_sample_multiplier" => 1.0,
        "n_iteration" => 20,
    )

    # We use the same input-output-pairs and normalization factors for
    # Gaussian Process and Random Feature cases
    input_output_pairs = get_training_points(ekiobj, length(get_u(ekiobj)) - 1)
    norm_factors = get_standardizing_factors(get_outputs(input_output_pairs))
    for case in cases[case_mask]

        println(" ")
        println("*********************************\n")
        @info "running case $case"

        if case == "gp-gpjl"

            @warn "gp-gpjl case is very slow at prediction"
            gppackage = GPJL()
            # Kernel is the sum of a squared exponential (SE), Matérn 5/2, and
            # white noise
            gp_kernel = SE(1.0, 1.0) + Mat52Ard(zeros(3), 0.0) + Noise(log(2.0))

            # Define machine learning tool
            mlt = GaussianProcess(gppackage; kernel = gp_kernel, prediction_type = pred_type, noise_learn = false)

        elseif case == "rf-scalar"

            kernel_structure = SeparableKernel(LowRankFactor(n_params, nugget), OneDimFactor())

            # Define machine learning tool
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_params,
                kernel_structure = kernel_structure,
                optimizer_options = optimizer_options,
            )

        else
            error("Case $case is not implemented yet.")

        end

        # The data processing normalizes input data, and decorrelates
        # output data with information from Γy
        emulator = Emulator(
            mlt,
            input_output_pairs,
            obs_noise_cov = Γy,
            standardize_outputs = true,
            standardize_outputs_factors = vcat(norm_factors...),
        )

        optimize_hyperparameters!(emulator)

        # Check how well the emulator predicts on the true parameters
        y_mean, y_var = Emulators.predict(emulator, reshape(θ_true, :, 1); transform_to_real = true)

        println("Emulator ($(case)) prediction on true parameters: ")
        println(vec(y_mean))
        println("true data: ")
        println(truth_sample) # what was used as truth
        println("Emulator ($(case)) predicted standard deviation: ")
        println(sqrt.(diag(y_var[1], 0)))
        println("Emulator ($(case)) MSE (truth): ")
        println(mean((truth_sample - vec(y_mean)) .^ 2))


        ###
        ###  Sample: Markov Chain Monte Carlo
        ###

        # initial values
        u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
        println("initial parameters: ", u0)

        # First let's run a short chain to determine a good step size
        yt_sample = truth_sample
        mcmc = MCMCWrapper(RWMHSampling(), yt_sample, priors, emulator; init_params = u0)

        new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

        # Now begin the actual MCMC
        println("Begin MCMC - with step size ", new_step)
        chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 1_000)

        posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

        post_mean = mean(posterior)
        post_cov = cov(posterior)
        println("posterior mean")
        println(post_mean)
        println("posterior covariance")
        println(post_cov)

        # Prior samples
        prior_samples_unconstr = sample(rng, priors, Int(1e4))
        prior_samples_constr = transform_unconstrained_to_constrained(priors, prior_samples_unconstr)

        # Posterior samples
        posterior_samples_unconstr = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) # samples are columns
        posterior_samples_constr =
            mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples_unconstr, dims = 1)

        # Make pair plots of the posterior distributions in the unconstrained
        # and in the constrained space (this uses `PairPlots.jl`)
        figpath_unconstr = joinpath(output_directory, "pairplot_posterior_unconstr_" * case * ".png")
        figpath_constr = joinpath(output_directory, "pairplot_posterior_constr.png_" * case * ".png")
        labels = get_name(posterior)

        data_unconstr = (; [(Symbol(labels[i]), posterior_samples_unconstr[i, :]) for i in 1:length(labels)]...)
        data_constr = (; [(Symbol(labels[i]), posterior_samples_constr[i, :]) for i in 1:length(labels)]...)

        p_unconstr = pairplot(data_unconstr => (PairPlots.Scatter(),))
        p_constr = pairplot(data_constr => (PairPlots.Scatter(),))
        save(figpath_unconstr, p_unconstr)
        save(figpath_constr, p_constr)

        # Plot the marginal posterior distributions together with the priors
        # and the true parameter values (we'll do that in the constrained space)

        for idx in 1:n_params

            # Find the range of the posterior samples
            xmin = minimum(posterior_samples_constr[idx, :])
            xmax = maximum(posterior_samples_constr[idx, :])

            # Create a figure and axis for plotting
            fig = Figure(; size = (800, 600))
            ax = Axis(fig[1, 1])

            # Histogram for posterior samples
            hist!(ax, posterior_samples_constr[idx, :], bins = 100, color = :darkorange, label = "posterior")

            # Plotting the prior distribution
            hist!(ax, prior_samples_constr[idx, :], bins = 10000, color = :slategray)

            # Adding a vertical line for the true value
            vlines!(ax, [ϕ_true[idx]], color = :indigo, linewidth = 2.6, label = "true " * param_names[idx])

            xlims!(ax, xmin, xmax)
            ylims!(ax, 0, nothing)

            # Setting title and labels
            ax.xlabel = "Value"
            ax.ylabel = "Density"
            ax.title = param_names[idx]
            ax.titlesize = 20

            # Save the figure (marginal posterior distribution in constrained
            # space)
            figname = "marginal_posterior_constr_" * case * "_" * param_names[idx] * ".png"
            figpath_marg_constr = joinpath(output_directory, figname)
            save(figpath_marg_constr, fig)

        end
    end
end


main()