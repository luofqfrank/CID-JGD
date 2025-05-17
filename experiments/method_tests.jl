using Printf
using DelimitedFiles
using Random
using ForwardDiff # for auto differentiate

#include("../utils/utils.jl")
#include("../utils/joint_gradient_QP_solver.jl")

include("../optimization_methods/BIGD_method.jl")
include("../optimization_methods/GS_method.jl")
include("../optimization_methods/TRB_method.jl")
include("../optimization_methods/QNGS_method.jl")

include("../problem_instances/gen_MAXQ.jl")
include("../problem_instances/gen_MXHILB.jl")
include("../problem_instances/Chained_LQ.jl")
include("../problem_instances/Chained_CB3_I.jl")
include("../problem_instances/Chained_CB3_II.jl")
include("../problem_instances/num_active_faces.jl")
include("../problem_instances/brown_func2.jl")
include("../problem_instances/Chained_Crescent_I.jl")
include("../problem_instances/Chained_Crescent_II.jl")

#=
obj_func_name_list = ["gen_MAXQ", "gen_MXHILB", "Chained_LQ", "Chained_CB3_I", "Chained_CB3_II", 
                      "num_active_faces", "brown_func2", "Chained_Crescent_I", "Chained_Crescent_II"]
=#
# hard instances "Chained_LQ"
obj_func_name_list = ["num_active_faces"]
n = 150
for func_name in obj_func_name_list
    println(func_name)
    func_cpt_name = func_name * "_cpt"
    func_init_name = "$(func_name)_init"
    f = (n, x) -> eval(Symbol(func_name))(n, x)
    f_cpt = (n, code, x) -> eval(Symbol(func_cpt_name))(n, code, x)
    f_init = (n) -> eval(Symbol(func_init_name))(n)
    Random.seed!(1)
    #x_init = 2 * rand(n) .- 1 
    x_init = f_init(n)
    file_path = nothing
    params = Dict("epsilon0" => 1e-1, "nu0" => 1e-3, "epsilon_opt" => 1e-5, "nu_opt" => 1e-4, 
              "gamma" => 0.5, "theta_epsilon" => 0.1, "theta_nu" => 0.9,
              "t_lb" => 1e-8, "rho_0" => 1e-2, "max_T" => 320)
    access = func_name in ["num_active_faces", "brown_func2"] ? 0 : 1
    BIGD_method(f, f_cpt, x_init, params, access, file_path)
    
    #=
    params = Dict("epsilon0" => 1e-3, "nu0" => 1e-3, "beta" => 1e-16, "gamma" => 0.5, "m" => 2 * n,
              "epsilon_opt" => 1e-5, "nu_opt" => 1e-4, "theta_epsilon" => 0.1, "theta_nu" => 0.9, "max_T" => 320)
    GS_method(f, f_cpt, x_init, params, file_path)
    =#
    #=
    params = Dict("tol" => 1e-6, "omega" => 1.2, "max_bundle_size" => 100, "gamma" => 0.01, "gamma_bar" => 0.1, "gamma_hat" => 0.4, "max_T" => 320)
    TRB_method(f, f_cpt, x_init, params, file_path)
    =#
    #=
    params = Dict("grad_opt" => 1e-4, "epsilon_opt" => 1e-5, "epsilon_0" => 0.1, "nu" => 1, "psi" => 0.5, "xi" => 1e-4, "eta_lb" => 1e-9, "eta_ub" => 0.9,
              "alpha_lb" => 1e-4, "alpha_ub" => 1, "gamma" => 0.5, "J_lb" => 5, "J_ub" => 10,
              "N0" => 100, "mu_lb" => 0.2, "mu_ub" => 100, "w_lb" => 1e-4, "w_ub" => 1, "m" => 100, "max_T" => 320)
    QNGS_method(f, f_cpt, x_init, params, file_path)
    =#
end
