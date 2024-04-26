using Statistics
using LinearAlgebra
using JuMP 
using ForwardDiff # to use auto gradient
include("../utils/utils.jl")

function joint_gradient_QP_solver(grad_list, lambda_start)
    println("QP solver started")
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_optimizer_attributes(model, "tol" => 1e-16,  # Desired optimality tolerance
        "constr_viol_tol" => 1e-16,  # Desired constraint violation tolerance
        "max_iter" => 10000)  # Maximum number of iterations
    k = length(grad_list)
    @assert k >= 1
    n = length(grad_list[1])
    @variable(model, lambda[1:k] >= 0)
    @constraint(model, sum(lambda[i] for i in 1:k) == 1)
    #@objective(model, Min, sum(v[i]^2 for i in 1:n))
    @objective(model, Min, sum(sum(lambda[j] * grad_list[j][i] for j in 1:k)^2 for i in 1:n))

    # warm start
    if lambda_start != nothing
        for i = 1:k 
            set_start_value(lambda[i], lambda_start[i])
        end
    end
    
    # Solve the optimization problem
    optimize!(model) 
    # retrieve the optimal solution and optimal objective
    opt_lambda = value.(lambda)
    opt_vec = sum(opt_lambda[i] * grad_list[i] for i in 1:k)
    return opt_vec, opt_lambda
end


# code2point: mapping code --> (x, grad); x is the last point where grad of the code has been computed
# reset -> true: re-compute the gradient; false: use the grad cached in code2point
# lambda_start is corresponding to code_lst, but it can be nothing
# new_code may or may not be in code_lst
function joint_gradient_extending(f_cpt::Function, code2point, code_lst, new_code, lambda_start, x, tol, reset::Bool)
    n = length(x)
    if lambda_start != nothing 
        @assert length(code_lst) == length(lambda_start)
    end
    if new_code != nothing && !in(new_code, Set(code_lst))
        new_grad = autoGrad(f_cpt, n, new_code, x) 
        code2point[new_code] = (x, new_grad)
    else 
        new_code = nothing
    end
    if reset 
        grad_lst = [autoGrad(f_cpt, n, c, x) for c in code_lst] 
    else
        grad_lst = []
        for c in code_lst 
            z, grad = code2point[c]
            if norm(x - z) <= tol 
                push!(grad_lst, grad) 
            else 
                push!(grad_lst, autoGrad(f_cpt, n, c, x))
            end
        end
    end
    if new_code != nothing
        push!(grad_lst, new_grad) 
        if lambda_start != nothing 
            push!(lambda_start, 0.0)   
        end     
    end 
    joint_grad, lambda_opt = joint_gradient_QP_solver(grad_lst, lambda_start)
    return joint_grad, lambda_opt
end