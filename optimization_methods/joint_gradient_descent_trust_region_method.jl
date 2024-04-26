using LinearAlgebra
using JuMP 
using Ipopt
using DelimitedFiles

# code2point: mapping code ---> (pt, val, grad)
function joint_gradient_extrapolate_in_gap(f::Function, f_cpt::Function, code2point, x, gap, r)
    n = length(x)
    code_x, val_x = f(n, x) 
    grad_x = autoGrad(f_cpt, n, code_x, x)
    code2point[code_x] = (x, val_x, grad_x)
    grad_list = []
    for c in keys(code2point)
        z, val, grad = code2point[c]
        if norm(x - z) <= r && val_x - gap <= val + dot(grad, x - z) <= val_x 
            push!(grad_list, grad) 
        end
    end
    opt_vec, components = joint_gradient_QP_solver(grad_list, nothing)
    return opt_vec, components
end

function joint_gradient_in_region(f_cpt, code2point, radius, x, val_x)
    grad_list = []
    for code in keys(code2point) 
        (z, val, grad) = code2point[code]
        if norm(x - z) <= radius && val_x >= val + dot(grad, x - z)
            push!(grad_list, grad)
        end
    end
    #println("number of local components = ", length(grad_list))
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_optimizer_attributes(model, "tol" => 1e-16,  # Desired optimality tolerance
        "constr_viol_tol" => 1e-16,  # Desired constraint violation tolerance
        "max_iter" => 10000)  # Maximum number of iterations
    n = length(x)
    k = length(grad_list)
    @assert k >= 1
    @variable(model, lambda[1:k] >= 0)
    @variable(model, v[1:n])
    for i in 1:n
        @constraint(model, v[i] == sum(lambda[j] * grad_list[j][i] for j in 1:k))
    end
    @constraint(model, sum(lambda[i] for i in 1:k) == 1)
    @objective(model, Min, sum(v[i]^2 for i in 1:n))
    # Solve the optimization problem
    optimize!(model)
    # retrieve the optimal solution and optimal objective
    opt_lambda = value.(lambda)
    opt_vec = sum(opt_lambda[i] * grad_list[i] for i in 1:k)
    #println("components = ", length(grad_list), ", complete solving QP")
    return opt_vec 
end


function joint_gradient_descent_trust_region_method(f::Function, f_cpt::Function, x_init, params, file_path) 
    file = open(file_path, "w")
    println(file, "Iter     CPU_time     tot_codes     act_codes     obj_dec     obj     flag_opt     flag_obj     flag_timing")
    tol = params["tol"] # 1e-5
    rho_0 = params["rho_0"] # 0.1
    mu_inc = params["mu_inc"] # 2
    mu_dec = params["mu_dec"] # 0.5 
    max_cpts = params["max_cpts"] # 50
    access_mode = params["access_mode"] # access_mode = 1: proactively accessable, 0: not proactively accessable
    x = x_init
    n = length(x)
    code, val = f(n, x)
    if access_mode == 0
        code2point = Dict(code => (x, val, autoGrad(f_cpt, n, code, x)))
    else 
        code2point = Dict(code => x)
    end
    joint_grad = autoGrad(f_cpt, n, code, x)
    iter = 0
    rho = 0
    x_prev = x_init # value of x in the previous iteration. It will be used to compute the moving distance 
    test_grad_norm = 1 
    r = 1e-3
    block_set = Set()
    dist_lst = []
    obj_prev = Inf 
    obj_cur = val
    obj_dec_list = [Inf]
    running_mode = "macro" # "macro", "micro"
    start_time = time() 
    elapsed_time = 0
    flag_opt = flag_obj = flag_timing = false 
    println(file, "0     0     1     1     0     $val     $flag_opt     $flag_obj     $flag_timing")
    while test_grad_norm > 1e2 * tol && !(length(obj_dec_list) >= 10 && maximum(obj_dec_list[1:10]) <= 1e-8) && elapsed_time < 1200
        iter += 1
        # compute the joint gradient
        if running_mode == "macro"
            if access_mode == 0
                joint_grad, components = joint_gradient_extrapolate_in_gap(f, f_cpt, code2point, x, Inf, r)
            else 
                # cpts: mapping code --> (pos, grad)
                cpts = load_components(f, f_cpt, code2point, x, r, Inf, access_mode)
                println("cpts = ", length(cpts))
                joint_grad, components = joint_gradient_QP_solver([cpts[c][2] for c in keys(cpts)], nothing)
            end
        else 
            if isempty(block_set)
                code, val = f(n, x)
                if access_mode == 0
                    code2point[code] = (x, val, autoGrad(f_cpt, n, code, x))
                else 
                    code2point[code] = x
                end
                push!(block_set, code)
            end 
            if access_mode == 0
                joint_grad, components = joint_gradient_QP_solver([code2point[c][3] for c in block_set], nothing) 
            else 
                joint_grad, components = joint_gradient_QP_solver([autoGrad(f_cpt, n, c, x) for c in block_set], nothing)
            end
        end
        # implement the line search
        alpha = 0.5
        println("joint_grad norm = ", norm(joint_grad))
        d = joint_grad / norm(joint_grad)
        code1, val1 = f(n, x)
        code2, val2 = f(n, x - alpha * d)
        rho = (val1 - val2) / (alpha * norm(joint_grad))
        if access_mode == 0
            code2point[code1] = (x, val1, autoGrad(f_cpt, n, code1, x))
        else 
            code2point[code1] = x
        end
        code2point_update(f_cpt, code2point, code2, x - alpha * d, val2, x, access_mode)
        while rho < rho_0 
            #println("alpha = ", alpha, ", rho = ", rho)
            alpha = mu_dec * alpha
            code2, val2 = f(n, x - alpha * d)
            rho = (val1 - val2) / (alpha * norm(joint_grad))
            code2point_update(f_cpt, code2point, code2, x - alpha * d, val2, x, access_mode)
        end
        # moving distance
        dist = norm(alpha * d)
        pushfirst!(dist_lst, dist)
        x_prev = x
        # x in the next iteration
        x = x - alpha * d
        # access the code and fun value at the new x
        code, val = f(n, x)
        obj_prev = obj_cur 
        obj_cur = val 
        pushfirst!(obj_dec_list, obj_prev - obj_cur)
        if access_mode == 0
            code2point[code] = (x, val, autoGrad(f_cpt, n, code, x))
        else 
            code2point[code] = x
        end
        ratio = dist / r
        if running_mode == "macro"
            if r < 1e-5 || (length(dist_lst) >= 5 && maximum(dist_lst[1:5]) < 1e-3)
                running_mode = "micro"
            else
                if ratio < 1
                    r = 0.5 * r
                elseif ratio > 10
                    r = 2 * r
                end
            end
        else
            push!(block_set, code)
            x_tst = x_prev - (alpha / mu_dec) * d
            code_tst, val_tst = f(n, x_tst)
            code2point_update(f_cpt, code2point, code_tst, x_tst, val_tst, x, access_mode)
            push!(block_set, code_tst)
            r = max_dist(code2point, block_set, x, access_mode)
            if r > 1 || length(block_set) > max_cpts || (r > 10 * tol && norm(joint_grad) < 1e2 * tol)
                block_set = Set([code])
            end
            #println("adjusted block size = ", length(block_set))
            #println("max_cpts = ", max_cpts)
        end
        # computing the joint gradient of component functions withint the gap = term_error
        # The norm of which will be used in termination criteria
        if iter % 10 == 0
            if access_mode == 0
                test_grad_norm = norm(joint_gradient_extrapolate_in_gap(f, f_cpt, code2point, x, Inf, 10 * tol)[1])
            else 
                cpts = load_components(f, f_cpt, code2point, x, 10 * tol, Inf, access_mode)
                test_grad_norm = norm(joint_gradient_QP_solver([cpts[c][2] for c in keys(cpts)], nothing)[1])
            end
        end
        end_time = time()
        elapsed_time = end_time - start_time
        flag_opt = test_grad_norm < 1e2 * tol ? true : false 
        flag_obj = (length(obj_dec_list) >= 10 && maximum(obj_dec_list[1:10]) <= 1e-8) ? true : false 
        flag_timing = elapsed_time >= 1200 ? true : false 
        println(file, "$iter     $elapsed_time     $(length(code2point))     $(length(block_set))     $(obj_dec_list[1])     $obj_cur     $flag_opt     $flag_obj     $flag_timing")
        println("running_mode = ", running_mode)
        println("Iter ", iter, ", alpha = ", alpha, ", ratio = ", ratio, ", r = ", r, ", move = ", dist, ", joint_grad = ", norm(joint_grad), ", test_grad_norm = ", test_grad_norm, ", obj = ", val)
    end
    close(file)
end
