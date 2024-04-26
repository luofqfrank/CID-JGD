# encoding rules: n-1 tuple
# (idx1, idx2, ..., idx_{n-1})
# idx_i indicating whether the first or second component in max in the i-th term is active 

function Chained_Crescent_II(n, x)
    res = 0
    code = zeros(Int, n-1)
    for i = 1:(n-1)
        f1 = x[i]^2 + (x[i+1]-1)^2 + (x[i+1]-1)
        f2 = - x[i]^2 - (x[i+1]-1)^2 + (x[i+1]+1)
        if f1 >= f2 
            res += f1 
            code[i] = 1
        else 
            res += f2
            code[i] = 2
        end
    end
    return code, res  
end


function Chained_Crescent_II_cpt(n, code, x)
    res = 0
    for i = 1:(n-1)
        if code[i] == 1
            res += x[i]^2 + (x[i+1]-1)^2 + (x[i+1]-1)
        else 
            res += - x[i]^2 - (x[i+1]-1)^2 + (x[i+1]+1)
        end
    end
    return res
end 


function Chained_Crescent_II_init(n)
    x_init = zeros(Float64, n)
    for i = 1:n
        if i % 2 == 1
            x_init[i] = - 1.5
        else
            x_init[i] = 2.0
        end
    end
    return x_init 
end

# obtain components that are near binding
# tol is the tolerance
function Chained_Crescent_II_near_binding_cpts(n, code, x, tol)
    code_set = Set([code])
    for i = 1:(n-1)
        f = [0.0, 0.0]
        f[1] = x[i]^2 + (x[i+1]-1)^2 + (x[i+1]-1)
        f[2] = - x[i]^2 - (x[i+1]-1)^2 + (x[i+1]+1)
        if abs(f[1] - f[2]) <= tol 
            if code[i] == 1
                new_code = code 
                new_code[i] = 2
                push!(code_set, new_code)
            else
                new_code = code 
                new_code[i] = 1
                push!(code_set, new_code)
            end
        end
    end
    return code_set 
end