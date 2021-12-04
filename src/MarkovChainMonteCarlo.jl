module MarkovChainMonteCarlo

using ..Emulators
using ..ParameterDistributions

using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
using Random

export MCMC
export mcmc_sample!
export accept_ratio
export reset_with_step!
export get_posterior
export find_mcmc_step!
export sample_posterior!


abstract type AbstractMCMCAlgo end
struct RandomWalkMetropolis <: AbstractMCMCAlgo end

"""
$(DocStringExtensions.TYPEDEF)

Structure to organize MCMC parameters and data.

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
mutable struct MCMC{FT <: AbstractFloat, IT <: Int}
    "A single sample from the observations. Can e.g. be picked from an `Obs` struct using `get_obs_sample`."
    obs_sample::AbstractVector{FT}
    "Covariance of the observational noise."
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}}
    "Array of length *N\\_parameters* with the parameters' prior distributions."
    prior::ParameterDistribution
    "MCMC step size."
    step::FT
    "Number of MCMC steps that are considered burnin."
    burnin::IT
    "The current parameters."
    param::AbstractVector{FT}
    "Array of accepted MCMC parameter samples (*param\\_dim* × *n\\_samples*). The histogram of these samples gives an approximation of the posterior distribution of the parameters."
    posterior::AbstractMatrix{FT}
    "The current value of the logarithm of the posterior (= `log_likelihood` + `log_prior` of the current parameters)."
    log_posterior::Union{FT, Nothing}
    "Iteration/step of the MCMC."
    iter::IT
    "Number of accepted proposals."
    accept::IT
    "MCMC algorithm to use. Currently implemented: `'rmw'` (random walk Metropolis), `'pCN'` (preconditioned Crank-Nicholson)."
    algtype::String
    "Random number generator object (algorithm + seed) used for sampling and noise, for reproducibility."
    rng::Random.AbstractRNG
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructor for [`MCMC`](@ref).
- `max_iter` - The number of MCMC steps to perform (e.g., 100_000).
"""
function MCMC(
    obs_sample::AbstractVector{FT},
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    prior::ParameterDistribution,
    step::FT,
    param_init::AbstractVector{FT},
    max_iter::IT,
    algtype::String,
    burnin::IT;
    svdflag = true,
    standardize = false,
    norm_factor::Union{AbstractVector{FT}, Nothing} = nothing,
    truncate_svd = 1.0,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
) where {FT <: AbstractFloat, IT <: Int}
    param_init_copy = deepcopy(param_init)

    # Standardize MCMC input?
    println(obs_sample)
    println(obs_noise_cov)
    if standardize
        obs_sample = obs_sample ./ norm_factor
        cov_norm_factor = norm_factor .* norm_factor
        obs_noise_cov = obs_noise_cov ./ cov_norm_factor
    end
    println(obs_sample)
    println(obs_noise_cov)

    # We need to transform obs_sample into the correct space
    if svdflag
        println("Applying SVD to decorrelating outputs, if not required set svdflag=false")
        obs_sample, _ = Emulators.svd_transform(obs_sample, obs_noise_cov; truncate_svd = truncate_svd)
    else
        println("Assuming independent outputs.")
    end
    println(obs_sample)

    # first row is param_init
    posterior = zeros(length(param_init_copy), max_iter + 1)
    posterior[:, 1] = param_init_copy
    param = param_init_copy
    log_posterior = nothing
    iter = 1
    accept = 0
    if !(algtype in ("rwm", "pCN"))
        error(
            "Unrecognized method: ",
            algtype,
            "Currently implemented methods: 'rwm' = random walk metropolis, ",
            "'pCN' = preconditioned Crank-Nicholson",
        )
    end
    MCMC{FT, IT}(
        obs_sample,
        obs_noise_cov,
        prior,
        step,
        burnin,
        param,
        posterior,
        log_posterior,
        iter,
        accept,
        algtype,
        rng,
    )
end


function reset_with_step!(mcmc::MCMC{FT, IT}, step::FT) where {FT <: AbstractFloat, IT <: Int}
    # reset to beginning with new stepsize
    mcmc.step = step
    mcmc.log_posterior = nothing
    mcmc.iter = 1
    mcmc.accept = 0
    mcmc.posterior[:, 2:end] = zeros(size(mcmc.posterior[:, 2:end]))
    mcmc.param[:] = mcmc.posterior[:, 1]
end


function get_posterior(mcmc::MCMC)
    #Return a parameter distributions object
    parameter_slices = batch(mcmc.prior)
    posterior_samples = [Samples(mcmc.posterior[slice, (mcmc.burnin + 1):end]) for slice in parameter_slices]
    flattened_constraints = get_all_constraints(mcmc.prior)
    parameter_constraints = [flattened_constraints[slice] for slice in parameter_slices] #live in same space as prior
    parameter_names = get_name(mcmc.prior) #the same parameters as in prior
    posterior_distribution = ParameterDistribution(posterior_samples, parameter_constraints, parameter_names)
    return posterior_distribution

end

function mcmc_sample!(
    mcmc::MCMC{FT, IT},
    g::AbstractVector{FT},
    gcov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
) where {FT <: AbstractFloat, IT <: Int}
    if mcmc.algtype == "rwm"
        log_posterior = log_likelihood(mcmc, g, gcov) + log_prior(mcmc)
    elseif mcmc.algtype == "pCN"
        # prior factors effectively cancel in acceptance ratio, so omit
        log_posterior = log_likelihood(mcmc, g, gcov)
    else
        error("Unrecognized algtype: ", mcmc.algtype)
    end

    if mcmc.log_posterior === nothing # do an accept step.
        mcmc.log_posterior = log_posterior - log(FT(0.5)) # this makes p_accept = 0.5
    end
    # Get new parameters by comparing likelihood_current * prior_current to
    # likelihood_proposal * prior_proposal - either we accept the proposed
    # parameter or we stay where we are.
    p_accept = exp(log_posterior - mcmc.log_posterior)

    if p_accept > rand(mcmc.rng, Distributions.Uniform(0, 1))
        mcmc.posterior[:, 1 + mcmc.iter] = mcmc.param
        mcmc.log_posterior = log_posterior
        mcmc.accept = mcmc.accept + 1
    else
        mcmc.posterior[:, 1 + mcmc.iter[1]] = mcmc.posterior[:, mcmc.iter]
    end
    mcmc.param = proposal(mcmc)[:]
    mcmc.iter = mcmc.iter + 1

end

function mcmc_sample!(
    mcmc::MCMC{FT, IT},
    g::AbstractVector{FT},
    gvar::AbstractVector{FT},
) where {FT <: AbstractFloat, IT <: Int}
    return mcmc_sample!(mcmc, g, Diagonal(gvar))
end

function accept_ratio(mcmc::MCMC{FT, IT}) where {FT <: AbstractFloat, IT <: Int}
    return convert(FT, mcmc.accept) / mcmc.iter
end


function log_likelihood(
    mcmc::MCMC{FT, IT},
    g::AbstractVector{FT},
    gcov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
) where {FT <: AbstractFloat, IT <: Int}
    log_rho = 0.0
    #if gcov == nothing
    #    diff = g - mcmc.obs_sample
    #    log_rho[1] = -FT(0.5) * diff' * (mcmc.obs_noise_cov \ diff)
    #else
    # det(log(Γ))
    # Ill-posed numerically for ill-conditioned covariance matrices with det≈0
    #log_gpfidelity = -FT(0.5) * log(det(Diagonal(gvar))) # = -0.5 * sum(log.(gvar))
    # Well-posed numerically for ill-conditioned covariance matrices with det≈0
    #full_cov = Diagonal(gvar)
    eigs = eigvals(gcov)
    log_gpfidelity = -FT(0.5) * sum(log.(eigs))
    # Combine got log_rho
    diff = g - mcmc.obs_sample
    log_rho = -FT(0.5) * diff' * (gcov \ diff) + log_gpfidelity
    #end
    return log_rho
end


function log_prior(mcmc::MCMC)
    return get_logpdf(mcmc.prior, mcmc.param)
end


function proposal(mcmc::MCMC)
    proposal_covariance = cov(mcmc.prior)
    prop_dist = MvNormal(zeros(length(mcmc.param)), proposal_covariance)

    if mcmc.algtype == "rwm"
        sample = mcmc.posterior[:, 1 + mcmc.iter] .+ mcmc.step * rand(mcmc.rng, prop_dist)
    elseif mcmc.algtype == "pCN"
        # Use prescription in Beskos et al (2017) "Geometric MCMC for infinite-dimensional 
        # inverse problems." for relating ρ to Euler stepsize:
        ρ = (1 - mcmc.step / 4) / (1 + mcmc.step / 4)
        sample = ρ * mcmc.posterior[:, 1 + mcmc.iter] .+ sqrt(1 - ρ^2) * rand(mcmc.rng, prop_dist)
    else
        error("Unrecognized algtype: ", mcmc.algtype)
    end
    return sample
end


function find_mcmc_step!(
    mcmc_test::MCMC{FT, IT},
    em::Emulator{FT};
    max_iter::IT = 2000,
) where {FT <: AbstractFloat, IT <: Int}
    step = mcmc_test.step
    mcmc_accept = false
    doubled = false
    halved = false
    countmcmc = 0

    println("Begin step size search")
    println("iteration 0; current parameters ", mcmc_test.param')
    flush(stdout)
    it = 0
    local acc_ratio
    while mcmc_accept == false

        param = reshape(mcmc_test.param, :, 1)
        em_pred, em_predvar = predict(em, param)
        if ndims(em_predvar[1]) != 0
            mcmc_sample!(mcmc_test, vec(em_pred), diag(em_predvar[1]))
        else
            mcmc_sample!(mcmc_test, vec(em_pred), vec(em_predvar))
        end
        it += 1
        if it % max_iter == 0
            countmcmc += 1
            acc_ratio = accept_ratio(mcmc_test)
            println("iteration ", it, "; acceptance rate = ", acc_ratio, ", current parameters ", param)
            flush(stdout)
            if countmcmc == 20
                println("failed to choose suitable stepsize in ", countmcmc, "iterations")
                exit()
            end
            it = 0
            if doubled && halved
                step *= 0.75
                reset_with_step!(mcmc_test, step)
                doubled = false
                halved = false
            elseif acc_ratio < 0.15
                step *= 0.5
                reset_with_step!(mcmc_test, step)
                halved = true
            elseif acc_ratio > 0.35
                step *= 2.0
                reset_with_step!(mcmc_test, step)
                doubled = true
            else
                mcmc_accept = true
            end
            if mcmc_accept == false
                println("new step size: ", step)
                flush(stdout)
            end
        end

    end

    return mcmc_test.step
end


function sample_posterior!(mcmc::MCMC{FT, IT}, em::Emulator{FT}, max_iter::IT) where {FT <: AbstractFloat, IT <: Int}
    for mcmcit in 1:max_iter
        param = reshape(mcmc.param, :, 1)
        # test predictions (param is 1 x N_parameters)
        em_pred, em_predvar = predict(em, param)

        if ndims(em_predvar[1]) != 0
            mcmc_sample!(mcmc, vec(em_pred), diag(em_predvar[1]))
        else
            mcmc_sample!(mcmc, vec(em_pred), vec(em_predvar))
        end

    end
end

end # module MarkovChainMonteCarlo
