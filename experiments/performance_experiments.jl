using DelimitedFiles
using Random
using ForwardDiff # for auto differentiate

include("../utils/utils.jl")
include("../utils/joint_gradient_QP_solver.jl")

include("../optimization_methods/joint_gradient_descent_pure_gap_method.jl")
include("../optimization_methods/joint_gradient_descent_bounded_components_method.jl")
include("../optimization_methods/joint_gradient_descent_trust_region_method.jl")
include("../optimization_methods/trust_region_bundle_method.jl")
include("../optimization_methods/variable_metric_method.jl")
include("../optimization_methods/quasi_newton_gradient_sampling_method.jl")

include("../problem_instances/gen_MAXQ.jl")
include("../problem_instances/gen_MXHILB.jl")
include("../problem_instances/Chained_LQ.jl")
include("../problem_instances/Chained_CB3_I.jl")
include("../problem_instances/Chained_CB3_II.jl")
include("../problem_instances/num_active_faces.jl")
include("../problem_instances/brown_func2.jl")
include("../problem_instances/Chained_Mifflin2.jl")
include("../problem_instances/Chained_Crescent_I.jl")
include("../problem_instances/Chained_Crescent_II.jl")

# access command-line arguments
args = ARGS 

obj_func_name_list = ["gen_MAXQ", "gen_MXHILB", "Chained_LQ", "Chained_CB3_I", "Chained_CB3_II", 
                      "num_active_faces", "brown_func2", "Chained_Mifflin2", "Chained_Crescent_I", "Chained_Crescent_II"]
      
domain_sensitive_func_name_list = ["num_active_faces", "brown_func2"]

dim_list = [100, 500, 1000, 2000, 5000]

for func_name in obj_func_name_list
    func_cpt_name = func_name * "_cpt"
    f = (n, x) -> eval(Symbol(func_name))(n, x)
    f_cpt = (n, code, x) -> eval(Symbol(func_cpt_name))(n, code, x)
    for n in dim_list
        Random.seed!(1)
        x_init = 2 * rand(n) .- 1 
        if args[1] == "joint_gradient_descent_method"
            file_path = "../results/joint_gradient_descent_method/" * "$func_name" * "_$(n).txt"
            if func_name == "gen_MAXQ"
                params = Dict("term_error" => 1e-5, "gap_to_grad_ratio" => 0.1, "rho_0" => 0.1, "mu_inc" => 2.0, "mu_dec" => 0.5)
                joint_gradient_descent_pure_gap_method(f, f_cpt, x_init, params, file_path)
            elseif func_name in domain_sensitive_func_name_list
                params = Dict("tol" => 1e-5, "max_cpts" => 50, "rho_0" => 0.1, "mu_inc" => 2, "mu_dec" => 0.5, "access_mode" => 0)
                joint_gradient_descent_trust_region_method(f, f_cpt, x_init, params, file_path)
            else
                params = Dict("tol" => 1e-5, "max_cpts" => 70, "rho_0" => 0.1, "mu_inc" => 2, "mu_dec" => 0.5)
                joint_gradient_descent_bounded_components_method(f, f_cpt, x_init, params, file_path)
            end
        else
            println("The input method is not available")
        end
    end
end



