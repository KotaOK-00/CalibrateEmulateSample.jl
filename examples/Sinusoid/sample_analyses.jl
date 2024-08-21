# # [Sample](@id sinusoid-example)

# In this example we have a model that produces a sinusoid
# ``f(A, v) = A \sin(\phi + t) + v, \forall t \in [0,2\pi]``, with a random
# phase ``\phi``. We want to quantify uncertainties on parameters ``A`` and ``v``,
# given noisy observations of the model output.
# Previously, in the emulate step, we built an emulator to allow us to make quick and
# approximate model evaluations. This will be used in our Markov chain Monte Carlo
# to sample the posterior distribution.

# First, we load the packages we need:
using LinearAlgebra, Random

using Distributions, Plots
using JLD2

using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo

const CES = CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses

# Next, we need to load the emulator we built in the previous step (emulate.jl must be run before this script
# We will start with the Gaussian process emulator.
example_directory = @__DIR__
data_save_directory = joinpath(example_directory, "output")
# Get observations, true parameters and observation noise
obs_file = joinpath(data_save_directory, "observations.jld2")
y_obs = load(obs_file)["y_obs"]
theta_true = load(obs_file)["theta_true"]
Γ = load(obs_file)["Γ"]
# Get GP emulators and prior
emulator_file = joinpath(data_save_directory, "emulators.jld2")
emulator_gp = load(emulator_file)["emulator_gp"]
prior = load(emulator_file)["prior"]
# Get random number generator to start where we left off
rng = load(emulator_file)["rng"]

# We will also need a suitable value to initiate MCMC. To reduce burn-in, we will use the
# final ensemble mean from EKI.
calibrate_file = joinpath(data_save_directory, "calibrate_results.jld2")
ensemble_kalman_process = load(calibrate_file)["eki"]

## Markov chain Monte Carlo (MCMC)
# Here, we set up an MCMC sampler, using the API. The MCMC will be run in the unconstrained space, for computational
# efficiency. First, we need to find a suitable starting point, ideally one that is near the posterior distribution.
# We start the MCMC from the final ensemble mean from EKI as this will increase the chance of acceptance near
# the start of the chain, and reduce burn-in time.
init_sample = EKP.get_u_mean_final(ensemble_kalman_process)
println("initial parameters: ", init_sample)
#@info typeof(emulator_gp.machiune_learning_tool)


#=
# Create MCMC from the wrapper: we will use a random walk Metropolis-Hastings MCMC (RWMHSampling())
# We need to provide the API with the observations (y_obs), priors (prior) and our emulator (emulator_gp).
# The emulator is used because it is cheap to evaluate so we can generate many MCMC samples.
mcmc = MCMCWrapper(BarkerSampling(), y_obs, prior, emulator_gp; init_params = init_sample)
# First let's run a short chain to determine a good step size
new_step =0.01
# new_step = optimize_stepsize(mcmc; rng = rng, init_stepsize = 0.1, N = 2000, discard_initial = 0)
# Now begin the actual MCMC
T = 10_000
# println("Begin MCMC - with step size ", new_step)     # 0.4


########################
#=
# using BenchmarkTools
println("MALA start")
new_step =0.01
mcmc = MCMCWrapper(MALASampling(), y_obs, prior, emulator_gp; init_params = init_sample)
@benchmark chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = new_step, discard_initial = 0)
println("MALA end")

println("RWMH start")
new_step =0.4
mcmc = MCMCWrapper(RWMHSampling(), y_obs, prior, emulator_gp; init_params = init_sample)
@benchmark chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = new_step, discard_initial = 0)
println("RWMH end")

println("pCNMH start")
new_step =0.01
mcmc = MCMCWrapper(pCNMHSampling(), y_obs, prior, emulator_gp; init_params = init_sample)
@benchmark chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = new_step, discard_initial = 0)
println("pCNMH end")

println("Barker start")
new_step =0.01
mcmc = MCMCWrapper(BarkerSampling(), y_obs, prior, emulator_gp; init_params = init_sample)
@benchmark chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = new_step, discard_initial = 0)
println("Barker end")
=#
########################


# We can print summary statistics of the MCMC chain
display(chain)
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
# Back to constrained coordinates
constrained_posterior = Emulators.transform_unconstrained_to_constrained(
prior, MarkovChainMonteCarlo.get_distribution(posterior)
)
samples = zeros(Float64, T, 2)
samples[:,1] = constrained_posterior["amplitude"]'
samples[:,2] = constrained_posterior["vert_shift"]'
covariance = cov(samples)
println("covariance: ", covariance)

# Note that these values are provided in the unconstrained space. The vertical shift
# seems reasonable, but the amplitude is not. This is because the amplitude is constrained to be
# positive, but the MCMC is run in the unconstrained space.  We can transform to the real
# constrained space and re-calculate these values.

# Extract posterior samples and plot
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))

constrained_amp = vec(constrained_posterior["amplitude"])
constrained_vshift = vec(constrained_posterior["vert_shift"])

println("Amplitude mean: ", mean(constrained_amp))
println("Amplitude std: ", std(constrained_amp))
println("Vertical Shift mean: ", mean(constrained_vshift))
println("Vertical Shift std: ", std(constrained_vshift))

# We can quickly plot priors and posterior using built-in capabilities
p = plot(prior, fill = :lightgray, rng = rng)
plot!(posterior, fill = :darkblue, alpha = 0.5, rng = rng, size = (800, 200))
savefig(p, joinpath(data_save_directory, "sinusoid_posterior_GP.png"))

# This shows the posterior distribution has collapsed around the true values for theta.
# Note, these are marginal distributions but this is a multi-dimensional problem with a
# multi-dimensional posterior. Marginal distributions do not show us how parameters co-vary,
# so we also plot the 2D posterior distribution.

# Plot 2D histogram (in constrained space)
amp_lims = (0, 6)           # manually set limits based on our priors
vshift_lims = (-6, 10)

hist2d = histogram2d(
    constrained_amp,
    constrained_vshift,
    colorbar = :false,
    xlims = amp_lims,
    ylims = vshift_lims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
)

# Let's also plot the marginal distributions along the top and the right hand
# panels. We can plot the prior and marginal posteriors as histograms.
prior_samples = sample(rng, prior, Int(1e4))
constrained_prior_samples = EKP.transform_unconstrained_to_constrained(prior, prior_samples)

tophist = histogram(
    constrained_prior_samples[1, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    legend = :false,
    lab = "Prior",
    yaxis = :false,
    xlims = amp_lims,
    xlabel = "Amplitude",
)
histogram!(
    tophist,
    constrained_amp,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = "Posterior",
)
righthist = histogram(
    constrained_prior_samples[2, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    orientation = :h,
    ylim = vshift_lims,
    xlims = (0, 1.4),
    xaxis = :false,
    legend = :false,
    lab = :false,
    ylabel = "Vertical Shift",
)

histogram!(
    righthist,
    constrained_vshift,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = :false,
    orientation = :h,
)

layout = @layout [
    tophist{0.8w, 0.2h} _
    hist2d{0.8w, 0.8h} righthist{0.2w, 0.8h}
]

plot_all = plot(
    tophist,
    hist2d,
    righthist,
    layout = layout,
    size = (600, 600),
    legend = :true,
    guidefontsize = 14,
    tickfontsize = 12,
    legendfontsize = 12,
)

savefig(plot_all, joinpath(data_save_directory, "sinusoid_MCMC_hist_GP.png"))


#=
### MCMC Sampling using Random Features Emulator

# We could repeat the above process with the random features (RF) emulator in place of the GP
# emulator. We hope to see similar results, since our RF emulator should be a good approximation
# to the GP emulator.

emulator_random_features = load(emulator_file)["emulator_random_features"]
mcmc = MCMCWrapper(RWMHSampling(), y_obs, prior, emulator_random_features; init_params = init_sample)
new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

println("Begin MCMC - with step size ", new_step)      # 0.4
chain = MarkovChainMonteCarlo.sample(mcmc, 1_000; stepsize = new_step, discard_initial = 2_000)

# We can print summary statistics of the MCMC chain
display(chain)

# The output of the random features MCMC is almost identical. Again, these are in the unconstrained space
# so we need to transform to the real (constrained) space and re-calculate these values.

# Extract posterior samples and plot
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))

constrained_amp = vec(constrained_posterior["amplitude"])
constrained_vshift = vec(constrained_posterior["vert_shift"])

println("Amplitude mean: ", mean(constrained_amp))
println("Amplitude std: ", std(constrained_amp))
println("Vertical shift mean: ", mean(constrained_vshift))
println("Vertical shift std: ", std(constrained_vshift))

# These numbers are very similar to our GP results. We can also check the posteriors look similar
# using the same plotting functions as before.
p = plot(prior, fill = :lightgray, rng = rng)
plot!(posterior, fill = :darkblue, alpha = 0.5, rng = rng, size = (800, 200))
savefig(p, joinpath(data_save_directory, "sinusoid_posterior_RF.png"))

# Plot 2D histogram (in constrained space)
# Using the same set up as before, with the same xlims, ylims.
hist2d = histogram2d(
    constrained_amp,
    constrained_vshift,
    colorbar = :false,
    xlims = amp_lims,
    ylims = vshift_lims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
)

# As before, we will plot the marginal distributions for both prior and posterior
# We will use the same prior samples generated for the GP histogram.
tophist = histogram(
    constrained_prior_samples[1, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    legend = :false,
    lab = "Prior",
    yaxis = :false,
    xlims = amp_lims,
    xlabel = "Amplitude",
)
histogram!(
    tophist,
    constrained_amp,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = "Posterior",
)
righthist = histogram(
    constrained_prior_samples[2, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    orientation = :h,
    ylim = vshift_lims,
    xlims = (0, 1.4),
    xaxis = :false,
    legend = :false,
    lab = :false,
    ylabel = "Vertical Shift",
)

histogram!(
    righthist,
    constrained_vshift,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = :false,
    orientation = :h,
)

layout = @layout [
    tophist{0.8w, 0.2h} _
    hist2d{0.8w, 0.8h} righthist{0.2w, 0.8h}
]

plot_all = plot(
    tophist,
    hist2d,
    righthist,
    layout = layout,
    size = (600, 600),
    legend = :true,
    guidefontsize = 14,
    tickfontsize = 12,
    legendfontsize = 12,
)

savefig(plot_all, joinpath(data_save_directory, "sinusoid_MCMC_hist_RF.png"))

# It is reassuring to see that this method is robust to the choice of emulator. The MCMC using
# both GP and RF emulators give very similar posterior distributions.
=#

=#


#= ESJD
# computing ESJD
function compute_ESJD(sigma_vec::Vector{Float64}, T::Int, n::Int, method)::Matrix{Float64}
    ESJD = Matrix{Float64}(undef, length(sigma_vec), n)
    for (i, sigma) in enumerate(sigma_vec)
        ESJD[i, :] = grad_MCMC_ESJD(T, n, method, sigma)
    end
    return ESJD
end

function grad_MCMC_ESJD(T::Int, n::Int, method, sigma::Float64)
    mcmc = MCMCWrapper(method, y_obs, prior, emulator_gp; init_params = init_sample)
    chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = sigma, discard_initial = 500)
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
    constrained_posterior = Emulators.transform_unconstrained_to_constrained(
    prior, MarkovChainMonteCarlo.get_distribution(posterior)
    )
    samples = zeros(Float64, T, 2)
    samples[:,1] = vec(constrained_posterior["amplitude"])
    samples[:,2] = vec(constrained_posterior["vert_shift"])
    n_samples, n_params = size(samples)
    esjd = zeros(Float64, n_params)
    for i in 2:n_samples
        esjd = esjd .+ (samples[i, :] .- samples[i - 1, :]).^ 2 ./ n_samples
    end
    return esjd
end


T = 10_000
n = 2
sqrt_n = sqrt(n)
sigma_vec = 2.38 ./ sqrt_n .* exp.(range(-7.5, stop = 3, length = 50))
println("sigma_vec: ", sigma_vec)
ESJD_MALA = compute_ESJD(sigma_vec, T, n, MALASampling())
ESJD_RW = compute_ESJD(sigma_vec, T, n, RWMHSampling())
ESJD_pCN = compute_ESJD(sigma_vec, T, n, pCNMHSampling())
ESJD_BARKER = compute_ESJD(sigma_vec, T, n, BarkerSampling())
# ESJD_gBARKER = compute_ESJD(sigma_vec, T, n, guidedBarkerSampling())
# ESJD_HMC = compute_ESJD(sigma_vec, T, n, HMCSampling())
# println("ESJD_HMC: ", ESJD_HMC)

println("maximum value of ESJD_MALA 1st: ", maximum(ESJD_MALA[:, 1]))
println("maximum value of ESJD_RW 1st: ", maximum(ESJD_RW[:, 1]))
println("maximum value of ESJD_pCN 1st: ", maximum(ESJD_pCN[:, 1]))
println("maximum value of ESJD_BARKER 1st: ", maximum(ESJD_BARKER[:, 1]))
# println("maximum value of ESJD_gBARKER 1st: ", maximum(ESJD_gBARKER[:, 1]))
# println("maximum value of ESJD_HMC 1st: ", maximum(ESJD_HMC[:, 1]))

println("maximum value of ESJD_MALA 2nd: ", maximum(ESJD_MALA[:, 2]))
println("maximum value of ESJD_RW 2nd: ", maximum(ESJD_RW[:, 2]))
println("maximum value of ESJD_pCN 2nd: ", maximum(ESJD_pCN[:, 2]))
println("maximum value of ESJD_BARKER 2nd: ", maximum(ESJD_BARKER[:, 2]))
# println("maximum value of ESJD_gBARKER 2nd: ", maximum(ESJD_gBARKER[:, 2]))
# println("maximum value of ESJD_HMC 2nd: ", maximum(ESJD_HMC[:, 2]))

# (2.38 ./ 2 .* exp.(range(-7.5, stop = 2, length = 50)))[15]
# maximum([ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]])
# maximum([ESJD_MALA[:, 2]; ESJD_RW[:, 2]; ESJD_pCN[:, 2]; ESJD_BARKER[:, 2]])
# minimum([ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]])
# minimum([ESJD_MALA[:, 2]; ESJD_RW[:, 2]; ESJD_pCN[:, 2]; ESJD_BARKER[:, 2]])


##############################
####### PLOTTING #############
##############################
# PLOT ESJD OF FIRST COORDINATE AND MEDIAN ESJD OF OTHER COORDINATES

#plot ESJD of first coordinate
ylim = (exp(-15), maximum([ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]])+0.1)
#ylim = (exp(-15), maximum([ESJD_MALA[:, 1]; ESJD_RW[:, 1]; ESJD_pCN[:, 1]; ESJD_BARKER[:, 1]; ESJD_HMC[:, 1]])+0.1)

p1 = plot(
    sigma_vec,
    ESJD_MALA[:, 1],
    xscale = :log10,
    yscale = :log10,
    label = "MALA",
    color = :red,
    marker = :diamond,
    xlabel = "proposal step-size",
    ylabel = "ESJD",
    ylim = ylim,
    legend = :bottomright,
)
plot!(p1, sigma_vec, ESJD_MALA[:, 1], label = nothing, color = :red)
scatter!(p1, sigma_vec, ESJD_RW[:, 1], label = "RW", color = :black, marker = :circle)
plot!(p1, sigma_vec, ESJD_RW[:, 1], label = nothing, color = :black)
scatter!(p1, sigma_vec, ESJD_pCN[:, 1], label = "pCN", color = :green, marker = :cross)
plot!(p1, sigma_vec, ESJD_pCN[:, 1], label = nothing, color = :green)
scatter!(p1, sigma_vec, ESJD_BARKER[:, 1], label = "Barker", color = :blue, marker = :utriangle)
plot!(p1, sigma_vec, ESJD_BARKER[:, 1], label = nothing, color = :blue)
# scatter!(p1, sigma_vec, ESJD_gBARKER[:, 1], label = "guided Barker", color = :orange, marker = :star5)
#plot!(p1, sigma_vec, ESJD_gBARKER[:, 1], label = nothing, color = :orange)
#scatter!(p1, sigma_vec, ESJD_HMC[:, 1], label = "HMC", color = :pink, marker = :dtriangle)
#plot!(p1, sigma_vec, ESJD_HMC[:, 1], label = nothing, color = :pink)
title!(p1, "ESJD of coordinate 1")
display(p1)
savefig(p1, joinpath(data_save_directory, "sinusoid_MCMC_ESJD_1.png"))

# PLOT ESJD OF FIRST COORDINATE AND MEDIAN ESJD OF OTHER COORDINATES
#plot ESJD of second coordinate
 ylim = (exp(-9), maximum([ESJD_MALA[:, 2]; ESJD_RW[:, 2]; ESJD_pCN[:, 2]; ESJD_BARKER[:, 2]])+0.1)

p2 = plot(
    sigma_vec,
    ESJD_MALA[:, 2],
    xscale = :log10,
    yscale = :log10,
    label = "MALA",
    color = :red,
    marker = :diamond,
    xlabel = "proposal step-size",
    ylabel = "ESJD",
    ylim = ylim,
    legend = :bottomright,
)
plot!(p2, sigma_vec, ESJD_MALA[:, 2], label = nothing, color = :red)
scatter!(p2, sigma_vec, ESJD_RW[:, 2], label = "RW", color = :black, marker = :circle)
plot!(p2, sigma_vec, ESJD_RW[:, 2], label = nothing, color = :black)
scatter!(p2, sigma_vec, ESJD_pCN[:, 2], label = "pCN", color = :green, marker = :cross)
plot!(p2, sigma_vec, ESJD_pCN[:, 2], label = nothing, color = :green)
scatter!(p2, sigma_vec, ESJD_BARKER[:, 2], label = "Barker", color = :blue, marker = :utriangle)
plot!(p2, sigma_vec, ESJD_BARKER[:, 2], label = nothing, color = :blue)
# scatter!(p2, sigma_vec, ESJD_gBARKER[:, 2], label = "guided Barker", color = :orange, marker = :star5)
# plot!(p2, sigma_vec, ESJD_gBARKER[:, 2], label = nothing, color = :orange)
# scatter!(p2, sigma_vec, ESJD_HMC[:, 2], label = "HMC", color = :pink, marker = :dtriangle)
# plot!(p2, sigma_vec, ESJD_HMC[:, 2], label = nothing, color = :pink)
title!(p2, "ESJD of coordinate 2")
display(p2)
savefig(p2, joinpath(data_save_directory, "sinusoid_MCMC_ESJD_2.png"))
ESJD =#



# plotting convergences of the chain in terms of l_2 distance for diagonal covariance and MSE

T = 500_000
plots_id = T-10_000:T
num_repeats = 1
methods = [MALASampling(), RWMHSampling(), pCNMHSampling(), BarkerSampling()]
sigmas = [0.0099, 0.0099, 0.0099, 0.0099]
colors = [:red, :black, :green, :blue]

samples_all = zeros(Float64, T, 2*length(methods))
d_t_values_all = zeros(Float64, T, length(methods))
frob_values_all = zeros(Float64, T, length(methods))
mse_values_all = zeros(Float64, T, length(methods))
# msle_values_all = zeros(Float64, T, length(methods))

for (j, method) in enumerate(methods)
    sum_samples = zeros(Float64, T, 2)

    mcmc = MCMCWrapper(method, y_obs, prior, emulator_gp; init_params = init_sample)
    for i in 1:num_repeats
        chain = MarkovChainMonteCarlo.sample(mcmc, T; rng = rng, stepsize = sigmas[j], discard_initial = 0)
        # sum_samples .+= chain.value[:, :, 1]
        posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
        # Back to constrained coordinates
        constrained_posterior = Emulators.transform_unconstrained_to_constrained(
        prior, MarkovChainMonteCarlo.get_distribution(posterior)
        )
        sum_samples[:,1] += constrained_posterior["amplitude"]'
        sum_samples[:,2] += constrained_posterior["vert_shift"]'
    end

    samples = sum_samples / num_repeats
    samples_all[:, 2j-1:2j] = samples
    #println(samples)
    means = zeros(Float64, 2)
    covariances = zeros(Float64, 2, 2, T)

    #= The true covariance matrix and means are picked from the 1 million runs of RWMH
    Iterations        = 2001:1:1002000
    Number of chains  = 1
    Samples per chain = 1000000
    parameters        = amplitude, vert_shift
    internals         = log_density, accepted

    covariance: [0.0808561649950171 0.007419140642478836; 0.007419140642478836 0.20971275263629424]
    Amplitude mean: 3.032846636107279
    Amplitude std: 0.2843521847903003
    Vertical Shift mean: 6.380403018472636
    Vertical Shift std: 0.45794404967888214
    =#

    Σ_true = [0.0808561649950171 0.007419140642478836;
           0.007419140642478836 0.20971275263629424]

    mean_true = [3.032846636107279, 6.380403018472636]

    for t in plots_id
        sample_subset = samples[1:t, :]
        #=
        if t == 1
            println("Skipping t=1 because it cannot be computed with a sample.")
            continue
        end
        =#
        covariances[:, :, t] = cov(sample_subset, dims=1)
        #if t%5000 == 0
        #    println(t, cov(sample_subset, dims=1))
        #end
    end

    for t in plots_id # 2:T
        Σ_t = covariances[:, :, t]
        d_t_values_all[t, j] = 1/sqrt(2) * sqrt( sum((log.(diag(Σ_t)) .- log.(diag(Σ_true))).^2) )
        #d_t_values_all[t, j] = sqrt(mean((diag(Σ_t) .- diag(Σ_true))).^2)
    end
#=
    for t in plots_id
        if t == 1
            println("Skipping t=1 because it cannot be computed with a sample.")
            continue
        end
        Σ_t = covariances[:, :, t]
        d_cov = size(Σ_t, 1)
        d_t_values_all[t, j] = 1 / sqrt(d_cov) * sqrt( sum((log.(diag(Σ_t)) .- log.(diag(Σ_true))).^2) )
        println("aa", d_t_values_all[t, j])
    end
    d_t_values_all[:,j] = cumsum(d_t_values_all[:,j]) ./ collect(1:length(plots_id))
=#

    for t in plots_id
        Σ_t = covariances[:, :, t]
        frob_values_all[t, j] = norm(Σ_t - Σ_true)
    end

    for t in plots_id
        sample_mean = mean(samples[1:t, :], dims=1)
        mse_values_all[t, j] = mean((sample_mean[:] - mean_true).^2)
        #if t%5000 == 0
        #    println(sample_mean - mean_true')
        #end
    end


    for t in plots_id
        sample_mean = mean(samples[1:t, :], dims=1)
        msle_values_all[t, j] = mean((sample_mean[:] - mean_true).^2)
    end

end

#= traceplots
p_tp = plot(1:T, samples_all[:, 1:2], label="MALA",
color=colors[1], xlabel="Iteration (t)", ylabel="d_t", title="samples", lw=2)
plot!(1:T, samples_all[:, 3:4], label="RW", color=colors[2], lw=2)
plot!(1:T, samples_all[:, 5:6].+1, label="pCN", color=colors[3], lw=2)
plot!(1:T, samples_all[:, 7:8].+1, label="BARKER", color=colors[4], lw=2)
display(p_tp)
savefig(p_tp, joinpath(data_save_directory, "sinusoid_MCMC_traceplots.png"))
=#

plot_d_t = plot(plots_id, d_t_values_all[plots_id, 1], label="d_t MALA",
color=colors[1], xlabel="Iteration (t)", ylabel="d_t",
title="Convergence of d_t over iterations", lw=2)
plot!(plots_id, d_t_values_all[plots_id, 2], label="d_t RW", color=colors[2], lw=2)
plot!(plots_id, d_t_values_all[plots_id, 3], label="d_t pCN", color=colors[3], lw=2)
plot!(plots_id, d_t_values_all[plots_id, 4], label="d_t BARKER", color=colors[4], lw=2)
display(plot_d_t)
savefig(plot_d_t, joinpath(data_save_directory, "sinusoid_MCMC_cov_convergence.png"))

plot_frob_norm = plot(plots_id, frob_values_all[plots_id, 1],
label="Frobenius Norm MALA", color=colors[1], xlabel="Iteration (t)", ylabel="Frobenius Norm",
title="Convergence of Frobenius Norm over iterations", lw=2)
plot!(plots_id, frob_values_all[plots_id, 2], label="Frobenius Norm RW", color=colors[2], lw=2)
plot!(plots_id, frob_values_all[plots_id, 3], label="Frobenius Norm pCN", color=colors[3], lw=2)
plot!(plots_id, frob_values_all[plots_id, 4], label="Frobenius Norm BARKER", color=colors[4], lw=2)
display(plot_frob_norm)
savefig(plot_frob_norm, joinpath(data_save_directory, "sinusoid_MCMC_frob_convergence.png"))

plot_mse = plot(plots_id, mse_values_all[plots_id, 1], label="MSE MALA",
color=colors[1], xlabel="Iteration (t)", ylabel="MSE",
title="Convergence of MSE over iterations", lw=2)
plot!(plots_id, mse_values_all[plots_id, 2], label="MSE RW", color=colors[2], lw=2)
plot!(plots_id, mse_values_all[plots_id, 3], label="MSE pCN", color=colors[3], lw=2)
plot!(plots_id, mse_values_all[plots_id, 4], label="MSE BARKER", color=colors[4], lw=2)
display(plot_mse)
savefig(plot_mse, joinpath(data_save_directory, "sinusoid_MCMC_mse_convergence.png"))


plot_mse2 = plot(plots_id, mse_values_all[plots_id, 1], label="MSE MALA",
color=colors[1], xlabel="Iteration (t)", ylabel="MSE",
title="Convergence of MSE over iterations", lw=2)
plot!(plots_id, mse_values_all[plots_id, 2], label="MSE RW", color=colors[2], lw=2)
plot!(plots_id, mse_values_all[plots_id, 4], label="MSE BARKER", color=colors[4], lw=2)
display(plot_mse2)
savefig(plot_mse2, joinpath(data_save_directory, "sinusoid_MCMC_mse2_convergence.png"))


plot_msle = plot(plots_id, msle_values_all[plots_id, 1], label="MSLE MALA",
color=colors[1], xlabel="Iteration (t)", ylabel="MSE",
title="Convergence of MSLE over iterations", lw=2)
plot!(plots_id, msle_values_all[plots_id, 2], label="MSLE RW", color=colors[2], lw=2)
plot!(plots_id, msle_values_all[plots_id, 3], label="MSLE pCN", color=colors[3], lw=2)
plot!(plots_id, msle_values_all[plots_id, 4], label="MSLE BARKER", color=colors[4], lw=2)
display(plot_msle)
savefig(plot_msle, joinpath(data_save_directory, "sinusoid_MCMC_msle_convergence.png"))

plot_msle = plot(plots_id, msle_values_all[plots_id, 1], label="MSLE MALA",
color=colors[1], xlabel="Iteration (t)", ylabel="MSE",
title="Convergence of MSLE over iterations", lw=2)
plot!(plots_id, msle_values_all[plots_id, 2], label="MSLE RW", color=colors[2], lw=2)
plot!(plots_id, msle_values_all[plots_id, 4], label="MSLE BARKER", color=colors[4], lw=2)
display(plot_msle)
savefig(plot_msle, joinpath(data_save_directory, "sinusoid_MCMC_msle2_convergence.png"))
