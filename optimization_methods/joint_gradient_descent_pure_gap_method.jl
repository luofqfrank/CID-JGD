using LinearAlgebra
using JuMP 
using Ipopt
using DelimitedFiles


function joint_gradient_descent_pure_gap_method(f::Function, f_cpt::Function, x_init, params, file_path)
    file = open(file_path, "w")
    println(file, "Iter     CPU_time     tot_codes     act_codes     obj_dec     obj     flag_opt     flag_obj     flag_timing")
    term_error = params["term_error"]
    gap_to_grad_ratio = params["gap_to_grad_ratio"]
    rho_0 = params["rho_0"]
    mu_inc = params["mu_inc"]
    mu_dec = params["mu_dec"] 
    x = x_init
    n = length(x)
    code, val = f(n, x)
    code_set = Set([code])
    iter = 0
    rho = 0
    gap = 1
    joint_grad = autoGrad(f_cpt, n, code, x)
    obj_prev = Inf 
    obj_cur = val 
    obj_dec_list = [Inf]
    start_time = time() 
    elapsed_time = 0
    flag_opt = flag_obj = flag_timing = false 
    println(file, "0     0     1     1     0     $val     $flag_opt     $flag_obj     $flag_timing")
    while !(gap <= 10 * term_error || norm(joint_grad) <= 1e2 * term_error) && !(length(obj_dec_list) >= 200 && maximum(obj_dec_list[1:10]) <= 1e-8) && !(elapsed_time >= 1200)
    #while iter < 10
        iter += 1
        # implement the line search
        alpha = 0.5
        code1, val1 = f(n, x)
        obj_prev = val1 
        act_code_list = get_act_codes(f_cpt, code_set, gap, code1, x) 
        joint_grad = joint_gradient(f_cpt, act_code_list, x)
        d = joint_grad / norm(joint_grad)
        code2, val2 = f(n, x - alpha * d)  
        rho = (val1 - val2) / (alpha * norm(joint_grad))
        push!(code_set, code1)
        push!(code_set, code2)
        while rho < rho_0
            alpha = mu_dec * alpha
            code2, val2 = f(n, x - alpha * d)
            rho = (val1 - val2) / (alpha * norm(joint_grad)) 
            println("alpha = ", alpha, ", rho = ", rho, ", joint_grad = ", norm(joint_grad))
        end
        x = x - alpha * joint_grad
        # update code_set, joint_grad and gap 
        code_cur, val = f(n, x)
        obj_cur = val 
        pushfirst!(obj_dec_list, obj_prev - obj_cur)
        push!(code_set, code_cur)
        gap = max(10 * term_error, gap_to_grad_ratio * norm(joint_grad))
        end_time = time()
        elapsed_time = end_time - start_time  
        flag_opt = (gap <= 10 * term_error || norm(joint_grad) <= 1e2 * term_error) ? true : false 
        flag_obj = (length(obj_dec_list) >= 200 && maximum(obj_dec_list[1:10]) <= 1e-8) ? true : false 
        flag_timing = elapsed_time >= 1200 ? true : false 
        # output to file
        println(file, "$iter     $elapsed_time     $(length(code_set))     $(length(act_code_list))     $(obj_dec_list[1])     $obj_cur     $flag_opt     $flag_obj     $flag_timing")
        println("Iter ", iter, ", gap = ", gap, ", joint_grad = ", norm(joint_grad), ", obj = ", val)
    end
end


function joint_gradient(f_cpt::Function, cpt_list, x)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_optimizer_attributes(model, "tol" => 1e-16,  # Desired optimality tolerance
        "constr_viol_tol" => 1e-16,  # Desired constraint violation tolerance
        "max_iter" => 100000)  # Maximum number of iterations
    n = length(x)
    k = length(cpt_list)
    @assert k >= 1
    grad_list = []
    for cpt in cpt_list
        push!(grad_list, autoGrad(f_cpt, n, cpt, x))
    end
    #println("k = ", k)
    #println("grad = ", grad_list[1])
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
    return opt_vec
end