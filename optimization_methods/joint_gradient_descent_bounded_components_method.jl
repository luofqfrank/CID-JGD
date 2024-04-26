using LinearAlgebra
using JuMP 
using Ipopt
using DelimitedFiles
include("../utils/utils.jl")
include("../utils/joint_gradient_QP_solver.jl")

function joint_gradient_descent_bounded_components_method(f::Function, f_cpt::Function, x_init, params, file_path)
    file = open(file_path, "w")
    println(file, "Iter     CPU_time     tot_codes     act_codes     obj_dec     obj     flag_opt     flag_obj     flag_timing")
    tol = params["tol"] # 1e-5
    max_cpts = params["max_cpts"] # 70
    rho_0 = params["rho_0"] # 0.1
    mu_inc = params["mu_inc"] # 2
    mu_dec = params["mu_dec"] # 0.5 
    gap_to_grad_ratio = 0.05
    x = x_init
    n = length(x)
    code, val = f(n, x)
    code2point = Dict(code => (x, autoGrad(f_cpt, n, code, x)))
    code_tot = Set()
    code_lst = [code]
    push!(code_tot, code)
    obj_prev = Inf 
    obj_cur = val 
    obj_dec_list = [Inf]
    new_code = nothing 
    lambda_start = nothing 
    reset = true 
    iter = 0
    alpha_max = 0.5
    gap_tot, gap_below = 0, 0
    joint_grad, lambda_opt = joint_gradient_extending(f_cpt, code2point, code_lst, new_code, lambda_start, x, tol, reset)
    start_time = time() 
    elapsed_time = 0
    flag_opt = flag_obj = flag_timing = false 
    println(file, "0     0     1     1     0     $val     $flag_opt     $flag_obj     $flag_timing")
    while !flag_opt && !flag_timing #!flag_opt && !flag_obj && !flag_timing
        iter += 1
        # compute joint_grad
        reset = iter % 10 == 0 ? true : false 
        if lambda_start != nothing 
            @assert length(lambda_start) == length(code_lst)
        end 
        joint_grad, lambda_opt = joint_gradient_extending(f_cpt, code2point, code_lst, new_code, lambda_start, x, tol, reset)
        lambda_start = lambda_opt 
        d = joint_grad / norm(joint_grad)
        if new_code != nothing && !in(new_code, Set(code_lst)) 
            @assert haskey(code2point, new_code)
            push!(code_lst, new_code) 
            push!(code_tot, new_code)
            code2point[new_code] = (x, autoGrad(f_cpt, n, new_code, x))
        end 
        @assert length(lambda_start) == length(code_lst)
        # performing line search
        alpha = alpha_max 
        c1, val1 = f(n, x) 
        c2, val2 = f(n, x - alpha * d) 
        union!(code_tot, Set([c1, c2]))
        rho = (val1 - val2) / (alpha * norm(joint_grad))
        alpha_max = rho >= 0.5 ? mu_inc * alpha_max : alpha_max 
        alpha_max = rho <= 0.01 ? mu_dec * alpha_max : alpha_max  
        while rho < rho_0
            alpha = mu_dec * alpha 
            c2, val2 = f(n, x - alpha * d) 
            push!(code_tot, c2)
            rho = (val1 - val2) / (alpha * norm(joint_grad))
        end
        # the last unqualified point
        c3, val3 = f(n, x - alpha * d / mu_dec)
        push!(code_tot, c3)
        code_set = Set(code_lst)
        x_prev = x 
        if in(c2, code_set) && !in(c3, code_set) 
            c_lb, val_lb, alpha_lb, c_ub, val_ub, alpha_ub = cross_border_bin_search(f, code_set, x, d, alpha, alpha / mu_dec)
            new_code = in(c_ub, code_set) ? nothing : c_ub 
            val_test = f(n, x - alpha_lb * d)[2]
            rho = (val1 - val_test) / (alpha_lb * norm(joint_grad))
            x = rho >= rho_0 ? x - alpha_lb * d : x - alpha * d
        else 
            x = x - alpha * d # corresponding to c2   
        end
        dist = norm(x - x_prev)
        c, val = f(n, x)
        obj_prev = val1  
        obj_cur = val 
        pushfirst!(obj_dec_list, obj_prev - obj_cur) 
        new_code = in(c, code_set) ? new_code : c 
        # if gap > 0.05 * norm(joint_grad), keep a single code in code_lst
        gap_tot, gap_below = component_gap(f_cpt::Function, code_set, x, c)
        if gap_tot >= max(gap_to_grad_ratio * norm(joint_grad), tol) || length(code_lst) > max_cpts
            code_lst = [c]
            code2point[c] = (x, autoGrad(f_cpt, n, c, x))
            lambda_start = nothing 
            reset = true
        end
        end_time = time()
        elapsed_time = end_time - start_time  
        flag_opt = (gap_below <= 10 * tol && norm(joint_grad) <= 1e2 * tol) ? true : false 
        flag_obj = (length(obj_dec_list) >= 200 && maximum(obj_dec_list[1:10]) <= 1e-8) ? true : false 
        flag_timing = elapsed_time >= 1200 ? true : false 
        # output to file
        println(file, "$iter     $elapsed_time     $(length(code_tot))     $(length(code_set))     $(obj_dec_list[1])     $obj_cur     $flag_opt     $flag_obj     $flag_timing")
        # output to screen
        println("Iter ", iter, ", alpha = ", alpha, ", move = ", dist, ", joint_grad = ", norm(joint_grad), ", gap_tot = ", gap_tot, ", code_set = ", length(code_set), " obj = ", val)
    end
    close(file)
end

# code is the active component at x
function component_gap(f_cpt::Function, code_set, x, code)
    n = length(x)
    obj = f_cpt(n, code, x)
    val_lst = [f_cpt(n, c, x) for c in code_set]
    gap_tot = maximum(val_lst) - minimum(val_lst)
    gap_below = max(obj - minimum(val_lst), 0) 
    return gap_tot, gap_below
end


